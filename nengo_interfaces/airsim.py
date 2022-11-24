import copy
import json
import math
import os
import time
import warnings
from os.path import expanduser

import nengo
import numpy as np
from abr_control.utils import transformations as transform

import airsim
from airsim.types import Pose, Quaternionr, Vector3r

# Only one optional function requires OpenCV, so only print a warning if it is not installed
try:
    import cv2
except ImportError:
    print("WARNING: could not import cv2, AirSim.get_camera_image() will not support resizing")

# https://github.com/microsoft/AirSim/blob/b7a65bb7f7a9471a2ec0ce6f573512b880d3197a/PythonClient/airsim/client.py#L679

# Additional AirSim functions for applying arbitrary forces
# available in this custom AirSim fork
# https://github.com/p3jawors/AirSim/tree/wind2
# self.client.simSetExtForce(np.array([Fx, Fy, Fz]))


class AirSim(nengo.Process):
    """Provides an easy to use API for AirSim in Nengo models.

    NOTES:
    - AirSim returns quaternions in the [x, y, z, w] format. We convert this to
    the [w, x, y, z] format for all output and expect [w, x, y, z] for all input
    - All Euler angles are output / expected to be in Tait-Bryan form.


    Parameters
    ----------
    dt : float, optional (Default: 0.01)
        The time step for the simulation dynamics.
    run_async : boolean, optional (Default: False)
        If true, run AirSim simulation independent of Nengo simulation.
        If false, run the two in lockstep, i.e., one Nengo time step then one
        AirSim time step
    camera_params : dict, optional (Default: None)
        'camera_name' : string or int
        'fps' : float
        'save_name' : string
        'use_physics' : bool, (Default: True if camera_params is None, else False)
            if True we move the drone around and simulate physics
            if False we move the camera around instead, this can speed up the
                sim time for generating datasets that just require visual data
        'capture_settings': dict, Optional (Default: None)
            Optional parameters to adjust in the capture settings
            These are typically manually set in ~/Documents/AirSim/settings.json
            any of the following can be passed in and will update
            Documents/Airsim/settings.json
            If a camera_idx is not passed in, it will be assumed to be the first
            camera in the list of your settings.json
                {
                        #NOTE camera_idx is the index of camera in your
                        settings.json CameraDefaults
                    "camera_idx": 0,
                    "ImageType": 0,
                    "Width": 832,
                    "Height": 832,
                    "FOV_Degrees": 107,
                    "AutoExposureSpeed": 100,
                    "AutoExposureBias": 0,
                    "AutoExposureMaxBrightness": 0.64,
                    "AutoExposureMinBrightness": 0.03,
                    "MotionBlurAmount": 0,
                    "TargetGamma": 0.7,
                    "ProjectionMode": "",
                    "OrthoWidth": 5.12
                }

    seed : int, optional (Default: None)
        Set the seed on the rng.
    takeoff : boolean, optional (Default: True)
        If true, the drone will take off and hover in the air upon connection.
        To prevent this action, set to false.
    show_display: bool, optional (Default: True)
    """

    def __init__(  # noqa: C901
        self,
        dt=0.01,
        run_async=False,
        camera_params=None,
        seed=None,
        takeoff=False,
        show_display=True,
    ):
        self.dt = dt
        # empirically determined function from recorded velocity feedback
        # from various dt the velocity feedback from airsim does not match
        # the derivative of the position feedback. See this github issue:
        # https://github.com/microsoft/AirSim/issues/2914
        self.vel_scale = 1 / (30.55 * self.dt + 0.03)

        self.run_async = run_async
        self.camera_params = camera_params if camera_params is not None else {}
        self.takeoff = takeoff

        self.euler_order = "rxyz"
        # Airsim class for vectors
        self.Vector3r = airsim.Vector3r

        self.use_physics = True

        if self.camera_params:
            # make sure our desired fps isn't greater than the max possible (1/dt)
            max_fps = np.round(1 / self.dt, 4)
            assert (
                self.camera_params["fps"] <= max_fps
            ), f"For dt={self.dt:.3f}sec your max camera fps={1/self.dt:.2f}"
            self.fps_remainder = (1 / camera_params["fps"]) - int(
                1 / camera_params["fps"]
            )
            self.fps_count = 0

            # calculate the possible frame rates we can capture images at given
            # our timestep
            frame_rate_locked = False
            available_frame_rates = []
            fps_multiple = 1
            while max_fps >= self.camera_params["fps"]:
                available_frame_rates.append(max_fps)

                if max_fps == self.camera_params["fps"]:  # pylint: disable=R1723
                    frame_rate_locked = True
                    break
                else:
                    fps_multiple += 1
                    max_fps = np.round(1 / (self.dt * fps_multiple), 4)
            if not frame_rate_locked:
                raise ValueError(
                    "Please select an fps with a time/image that is a multiple"
                    + " of your timestep:",
                    available_frame_rates,
                )

            # TODO remove after all projects updated
            # checks for deprecated parameter name while old projects get changed over
            if "training_mode" in camera_params:
                warnings.warn(
                    "'training_mode' key is deprecated, use 'use_physics' key"
                    + " instead to toggle physics engine"
                )
                if "use_physics" not in camera_params:
                    camera_params["use_physics"] = camera_params["training_mode"]

            self.use_physics = self.camera_params["use_physics"]

            if "capture_settings" not in camera_params:
                camera_params["capture_settings"] = None

            # create the capture save folder if it doesn't exist
            _tmp = camera_params["save_name"]
            if _tmp is not None:
                if not os.path.exists(_tmp):
                    os.makedirs(_tmp)
            _tmp = None

        # update our settings.json that airsim reads in
        home = expanduser("~")
        with open(  # pylint: disable=W1514
            f"{home}/Documents/AirSim/settings.json", "r+"
        ) as fp:
            data = json.load(fp)
            prev_data = copy.deepcopy(data)

            # physics mode, set to multirotor

            CUR_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "."))
            if self.use_physics:
                data["SimMode"] = "Multirotor"
                data["EngineSound"] = True
                print("phys mode")

                if "Vehicles" not in data.keys():
                    # load vehicle params from backup file
                    with open(  # pylint: disable=W1514
                        f"{CUR_DIR}/fly_settings.json", "r+"
                    ) as fp_fly:
                        # save vehicle params to backup file
                        vehicle_data = json.load(fp_fly)["Vehicles"]
                        data["Vehicles"] = vehicle_data

            # computer vision, turn off physics to speed things up
            else:
                data["SimMode"] = "ComputerVision"
                data["EngineSound"] = False

                if "Vehicles" in data.keys():
                    with open(  # pylint: disable=W1514
                        f"{CUR_DIR}/fly_settings.json", "r+"
                    ) as fp_fly:
                        # save vehicle params to backup file
                        vehicle_data = json.load(fp_fly)
                        vehicle_data["Vehicles"] = data["Vehicles"]
                        fp_fly.seek(0)
                        json.dump(vehicle_data, fp_fly, indent=4)
                        fp_fly.truncate()

                    # remove from dict as this will break ComputerVision mode
                    data.pop("Vehicles")

            # can turn display off for speed boost
            with open(  # pylint: disable=W1514
                f"{CUR_DIR}/cv_settings.json", "r+"
            ) as fp_cv:

                cv_data = json.load(fp_cv)

                if not show_display:
                    # back up view mode if one exists
                    if "ViewMode" in data.keys():
                        # if a view mode exists that is not NoDisplay, back it up
                        if data["ViewMode"] != "NoDisplay":
                            cv_data["ViewMode"] = data["ViewMode"]
                    data["ViewMode"] = "NoDisplay"
                else:
                    # get our default view mode that is backed up
                    if cv_data["ViewMode"] == "NoDisplay":
                        # depending on how someone cycles between phys and CV mode
                        # the saved value may be NoDisplay, but intuitively we want
                        # to see something if ViewMode = True
                        view_mode = ""
                    else:
                        # user defined
                        view_mode = cv_data["ViewMode"]

                    # check if we need to change the value in settings
                    if "ViewMode" in data.keys():
                        # set to our backed up view mode if still on NoDisplay
                        # otherwise leave it at the user defined ViewMode
                        if data["ViewMode"] == "NoDisplay":
                            data["ViewMode"] = view_mode
                    else:
                        # use our backed up ViewMode
                        data["ViewMode"] = view_mode

                fp_cv.seek(0)
                json.dump(cv_data, fp_cv, indent=4)
                fp_cv.truncate()

            # if user passes in camera params with capture settings,
            # update the settings.json
            if isinstance(camera_params, dict):
                if isinstance(camera_params["capture_settings"], dict):
                    camera_idx = (
                        camera_params["capture_settings"]["camera_idx"]
                        if "camera_idx" in camera_params["capture_settings"]
                        else 0
                    )
                    # TODO
                    # dict.get(key, return val if missing)
                    if "CameraDefaults" not in data.keys():
                        data["CameraDefaults"] = cv_data["CameraDefaults"]
                    elif "CaptureSettings" not in data["CameraDefaults"]:
                        data["CameraDefaults"]["CaptureSettings"] = cv_data[
                            "CameraDefaults"
                        ]["CaptureSettings"]
                    for key in camera_params["capture_settings"]:
                        data["CameraDefaults"]["CaptureSettings"][camera_idx][
                            key
                        ] = camera_params["capture_settings"][key]

            fp.seek(0)
            json.dump(data, fp, indent=4)
            fp.truncate()

            # The settings.json get loaded in when the UE4 sim starts, which is
            # required for the Airsim API to connect. If our settings have changed
            # throw an error to inform the user that their UE4 sim needs to be
            # restarted for the changed to take effect
            if data != prev_data:
                yellow = "\u001b[33m"
                endc = "\033[0m"
                raise RuntimeError(
                    f"{yellow}You will need to Stop and Start UE4 for setting.json"
                    + f" changes to apply{endc}"
                )

        # instantiate our microsoft airsim client
        self.client = airsim.MultirotorClient()

        # constants required to convert our control rotor velocities
        # to the required pwm signal
        # from airsim/airlib/include/vehicles/multirotor/RotorParams.hpp
        C_T = 0.109919
        air_density = 1.225
        prop_diam = 0.2275
        self.max_thrust = 4.179446268
        self.k = C_T * air_density * prop_diam**4

        if self.use_physics:
            # size_in = the number of rotors on our quadcopter, hardcoded
            size_in = 4
            self.air_density_ratio = (
                self.client.simGetGroundTruthEnvironment().air_density / air_density
            )

        else:
            # controlling camera instead, so use position and orientation
            # of path planner
            size_in = 12
        # size_out = number of parameters in our feedback array,
        # which is of the form...
        # [x, y, z, dx, dy, dz, roll, pitch, yaw, droll, dpitch, dyaw]
        size_out = 12
        super().__init__(size_in, size_out, default_dt=dt, seed=seed)

    def connect(self, pause=False):
        """
        Parameters
        ----------
        pause: boolean, Optional (Default: False)
            whether to pause on connect, sometimes if the initialization
            of your code takes a while the drone will fall to the ground
            before your sim starts. This will instantiate the connection
            and pause until your sim starts
        """
        self.client.confirmConnection()
        self.client.reset()
        if self.use_physics:
            self.client.enableApiControl(True)
            self.client.armDisarm(True)

        # https://microsoft.github.io/AirSim/apis/#async-methods-duration-and-max_wait_seconds
        # NOTE appending .join() will make the call synchronous (wait for completion)
        if self.takeoff:
            # takeoff to 3m above the ground
            if self.run_async:
                self.client.takeoffAsync()
            else:
                self.client.takeoffAsync().join()
        self.is_paused = pause
        self.client.simPause(self.is_paused)

    def disconnect(self, pause=False):
        """
        Parameters
        ----------
        pause: boolean, Optional (Default: False)
            whether to pause on disconnect
        """
        if pause:
            self.client.simPause(False)
        self.client.reset()

        if self.use_physics:
            self.client.enableApiControl(False)
            self.client.armDisarm(False)

        if pause:
            self.client.simPause(True)

    def pause(self, sim_pause=True):
        """
        Parameters
        ----------
        sim_pause: boolean, Optional (Default: True)
            True to pause, False to resume
        """
        # do not call if already in desired state
        if sim_pause is not self.is_paused:
            self.is_paused = not self.is_paused
            self.client.simPause(sim_pause)

    def make_step(self, shape_in, shape_out, dt, rng, state):
        """Create the function for the Airsim interfacing Nengo Node."""

        def step(t, x):
            """Takes in PWM commands for 4 motors. Returns the state of the drone
            after running the command.

            Parameters
            ----------
            u : float
                The control signal to send to AirSim simulation
            """
            # bias and scale the input signals
            u = x

            if self.use_physics:
                self.send_pwm_signal(u)
            else:
                cam_state = [x[0], x[1], x[2], x[6], x[7], x[8]]
                self.set_camera_state(cam_state)

            feedback = self.get_feedback()

            # render camera feedback
            if self.camera_params:
                # subtract dt because we start at dt, not zero, but
                # we want an image on the first step, before the drone starts
                # moving
                if (int(1000 * np.round(t - self.dt, 4))) % int(
                    1000 * np.round(1 / self.camera_params["fps"], 4)
                ) < 1e-5:
                    self.fps_count += 1
                    # scale to seconds and use integers to minimize
                    # rounding issues with floats
                    if self.camera_params["save_name"] is not None:
                        _ = self.get_camera_feedback(
                            camera_name=self.camera_params["camera_name"],
                            save_name=(
                                f"{self.camera_params['save_name']}"
                                + f"/frame_{int(t*1000):08d}"
                            ),
                        )
                    else:
                        _ = self.get_camera_feedback(
                            camera_name=self.camera_params["camera_name"],
                            save_name=None,
                        )

            return np.hstack(
                [
                    feedback["position"],
                    feedback["linear_velocity"],
                    feedback["taitbryan"],
                    feedback["angular_velocity"],
                ]
            )

        return step

    def send_pwm_signal(self, u):
        """
        Send PWM controlled signals to each motor in the order
        [front_right_pwm, rear_left_pwm, front_left_pwm, rear_right_pwm]

        Parameters
        ----------
        u: list of 4 floats
            airsim expected pwm signal to each motor in the range of -1 to 1
            This function accepts rotor velocities in rad/sec and converts them
            to pwm
        dt: float
            the time to run the pwm signal for
        """
        # the pwm output is calculated as...
        # thrust / (max_thrust * air_density_ratio)
        # where thrust is k*w**2
        # https://github.com/microsoft/AirSim/issues/2592
        # NOTE that the k calculated in airsim uses velocity in units of rev/sec
        pwm = np.squeeze(self.k * u / (self.max_thrust * self.air_density_ratio))

        # NOTE appending .join() will make the call synchronous
        # (wait for completion)
        if self.run_async:
            self.client.moveByMotorPWMsAsync(pwm[0], pwm[1], pwm[2], pwm[3], self.dt)
        else:
            self.client.simPause(False)
            self.client.moveByMotorPWMsAsync(
                pwm[0], pwm[1], pwm[2], pwm[3], self.dt
            ).join()
            time.sleep(self.dt)
            self.client.simPause(True)

    def get_feedback(self):
        """
        Calls the simGetGroundTruthKinematics to get system feedback, which is
        then parsed from the airsim custom type to a dict

        returns dict of state in the form:
            {position, quaternion(in format w,x,y,z), taitbryan(euler in
            taitbryan angles)}

        the full state can be accessed from self.state with the main keys:
            - landed_state
            - collision
            - timestamp
            - kinematics estimated
            - rc_data
            - gps_location
        """
        if self.use_physics:
            state = self.client.getMultirotorState().kinematics_estimated
            pos = state.position.to_numpy_array()

            airsim_quat = state.orientation.to_numpy_array()
            # NOTE: The quadcopter feedback does not need to be rotated to be in
            # the expected axes, unlike object orientation feedback.
            # We do need to reorder quaternion to [w, x, y, z] still, though.
            quat = np.array(
                [airsim_quat[3], airsim_quat[0], airsim_quat[1], airsim_quat[2]]
            )

            lin_vel = self.vel_scale * state.linear_velocity.to_numpy_array()
            ang_vel = self.vel_scale * state.angular_velocity.to_numpy_array()

            return {
                "position": pos,
                "linear_velocity": lin_vel,
                "quaternion": quat,
                "taitbryan": self.quat_to_taitbryan(quat),
                "angular_velocity": ang_vel,
            }
        else:
            return {
                "position": [0, 0, 0],
                "linear_velocity": [0, 0, 0],
                "quaternion": [0, 0, 0],
                "taitbryan": [0, 0, 0],
                "angular_velocity": [0, 0, 0],
            }

    def get_camera_feedback(self, camera_name="0", save_name=None):
        """
        Gets an image from the specified camera, and saved to save_name

        Parameters
        ----------
        camera_name: int, Optional (Default: 0)
            the camera index in your settings.json
        save_name: string, Optional (Default: None)
            the filename to save the image to
        """
        responses = self.client.simGetImages(
            [airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)]
        )
        response = responses[0]

        # get numpy array
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)

        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)

        # write to png
        if save_name is not None:
            airsim.write_png(os.path.normpath(save_name + ".png"), img_rgb)

        return img_rgb

    def get_state(self, name):
        """Get the state of an object, return 3D position and orientation
            return quaternion in order [w, x, y, z] and euler angles in
            taitbryan format

        Parameters
        ----------
        name : string
            The name of the object.
        """
        state = self.client.simGetObjectPose(name)
        pos = state.position.to_numpy_array()
        airsim_quat = state.orientation.to_numpy_array()
        quat = np.array(
            [airsim_quat[3], airsim_quat[0], airsim_quat[1], airsim_quat[2]]
        )

        return {
            "position": pos,
            "quaternion": quat,
            "taitbryan": self.quat_to_taitbryan(quat),
        }

    def set_state(self, name, xyz=None, orientation=None):
        """Set the state of object given 3D location and quaternion
            Accepts state position in meters and orientation in euler
            angles in order set by euler_order in the __init__

        Parameters
        ----------
        name: string
            the name of the object
        xyz: 3d array or list, optional (Default: None)
            the desired xyz position in meters
            if you do not want to set the position, set to None
        orientation: 3d array or list, optional (Default: None)
            the desired orientation as euler angles in the format set by
            self.euler_order in the __init__
        """
        xyz = [float(xyz[0]), float(xyz[1]), float(xyz[2])]

        orientation = transform.quaternion_from_euler(
            orientation[0], orientation[1], orientation[2], self.euler_order
        )

        # reorder to xyzw to match the quaternion format of airsim
        quat = [
            float(orientation[1]),
            float(orientation[2]),
            float(orientation[3]),
            float(orientation[0]),
        ]
        pose = Pose(
            position_val=Vector3r(x_val=xyz[0], y_val=xyz[1], z_val=xyz[2]),
            orientation_val=Quaternionr(
                x_val=quat[0], y_val=quat[1], z_val=quat[2], w_val=quat[3]
            ),
        )
        self.client.simSetObjectPose(name, pose, teleport=True)

    def set_camera_state(self, state, name=None):
        """Set the state of object given 3D location and quaternion
            Accepts state position in meters and orientation in euler
            angles in order set by euler_order in the __init__

        Parameters
        ----------
        name: string
            the name of the object
        xyz: 3d array or list, optional (Default: None)
            the desired xyz position in meters
            if you do not want to set the position, set to None
        orientation: 3d array or list, optional (Default: None)
            the desired orientation as euler angles in the format set by
            self.euler_order in the __init__
        """
        if name is None:
            name = self.camera_params["camera_name"]
        xyz = state[:3]
        orientation = state[3:]
        xyz = [float(xyz[0]), float(xyz[1]), float(xyz[2])]

        orientation = transform.quaternion_from_euler(
            orientation[0], orientation[1], orientation[2], self.euler_order
        )

        # reorder to xyzw to match the quaternion format of airsim
        quat = [
            float(orientation[1]),
            float(orientation[2]),
            float(orientation[3]),
            float(orientation[0]),
        ]
        pose = Pose(
            position_val=Vector3r(x_val=xyz[0], y_val=xyz[1], z_val=xyz[2]),
            orientation_val=Quaternionr(
                x_val=quat[0], y_val=quat[1], z_val=quat[2], w_val=quat[3]
            ),
        )
        self.client.simSetCameraPose(name, pose)  # , teleport=True)

    def get_drone_state(self):
        """Get the drone state as a numpy array"""

        state_dict = self.get_feedback()
        return np.hstack(
            [
                state_dict["position"],
                state_dict["linear_velocity"],
                state_dict["taitbryan"],
                state_dict["angular_velocity"],
            ]
        )

    def set_drone_state(self, pose, ignore_collision=True):
        """Teleport the drone to a particular position and orientation

        pose: array-like
            First three dimensions are the x, y, and z positions
            Second three dimensions are roll, pitch, and yaw
        """

        self.client.simSetVehiclePose(
            pose=airsim.Pose(
                airsim.Vector3r(pose[0], pose[1], pose[2]),
                airsim.to_quaternion(pose[3], pose[4], pose[5])
            ),
            ignore_collision=ignore_collision
        )

    def quat_to_taitbryan(self, quat):
        """Convert quaternion to Tait-Bryan Euler angles

        quat : np.array
            The quaternion in [w, x, y, z] format
        """
        euler = np.asarray(transform.euler_from_quaternion(quat, self.euler_order))
        euler = self.ea_xyz_to_zxy(euler)

        return euler

    def ea_xyz_to_zxy(self, ang):
        """Converts Euler angles from x-y-z to z-x-y convention"""

        def b(num):
            """forces magnitude to be 1 or less"""
            if abs(num) > 1.0:
                return np.copysign(1.0, num)
            else:
                return num

        s1 = np.sin(ang[0])
        s2 = np.sin(ang[1])
        s3 = np.sin(ang[2])
        c1 = np.cos(ang[0])
        c3 = np.cos(ang[2])

        pitch = math.asin(b(c1 * c3 * s2 - s1 * s3))
        cp = np.cos(pitch)
        if cp == 0:
            cp = 1e-6

        yaw = math.asin(b((c1 * s3 + c3 * s1 * s2) / cp))  # flipped
        # Fix for getting the quadrants right
        if c3 < 0:
            if yaw > 0:
                yaw = np.pi - yaw
            elif yaw < 0:
                yaw = -np.pi - yaw

        roll = math.asin(b((c3 * s1 + c1 * s2 * s3) / cp))  # flipped
        return [roll, pitch, yaw]

    def get_camera_images(
            self,
            image_shape=(144, 256),
            normalize=False,
            depth_camera=False,
            back_camera=False,
            depth_max=50,
            objects_to_hide=None,
            hidden_depth=100
    ):
        """Returns images from the drone after applying the specified processing options.
        The return format is a dictionary with the names of the images as keys and numpy arrays as values.
        Between 1 and 4 images are returned depending on the parameter settings, the possible names are:
        "front_rgb", "back_rgb", "front_depth", and "back_depth"

        Parameters
        ----------
        image_shape: tuple (Default: (144, 256))
            Height and width of the output images.
            If the AirSim camera settings are different, the images will be resized
        normalize: boolean (Default: False)
            If set to True, normalize the pixel values in the returned images to be between -1 and 1
        depth_camera: boolean (Default: False)
            If set to True, include a depth image along with RGB
        back_camera: boolean (Default: False)
            If set to True, capture images from both a front facing and back facing camera
        depth_max: float (Default: 50)
            The maximum that all depth values are clipped to when a depth camera is used.
            Also influences the logarithmic normalize if normalize is set to True
        objects_to_hide: list of str, or None (Default: None)
            An optional list of object names to move out of the scene before capturing an image
            and then move back again to continue simulation. The main use case are objects such as arrows and spheres
            for visualizing locations and directions that should not be seen by the camera.
        hidden_depth: float (Default: 100)
            When hiding objects, the z-value that they will all get moved to
        """

        camera_names = ["0"]

        if back_camera:
            camera_names.append("4")

        if objects_to_hide is not None:
            object_states = {}
            for object_name in objects_to_hide:
                # store original position
                object_states[object_name] = self.get_state(name=object_name)
                # move under ground
                self.set_state(
                    name=object_name,
                    xyz=(0, 0, hidden_depth),
                    orientation=object_states[object_name]["taitbryan"]
                )

        # the requests to AirSim
        requests = []

        # types of each image, either rgb or depth
        image_types = []

        # the processed images
        images = {}

        for camera_name in camera_names:
            requests.append(airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False))
            if camera_name == "0":
                image_types.append("front_rgb")
            elif camera_name == "4":
                image_types.append("back_rgb")
            else:
                raise NotImplementedError()
            if depth_camera:
                requests.append(airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, True, False))
                if camera_name == "0":
                    image_types.append("front_depth")
                elif camera_name == "4":
                    image_types.append("back_depth")
                else:
                    raise NotImplementedError()

        responses = self.client.simGetImages(requests)

        if objects_to_hide is not None:
            for object_name in objects_to_hide:
                # move back to original position
                self.set_state(
                    name=object_name,
                    xyz=object_states[object_name]["position"],
                    orientation=object_states[object_name]["taitbryan"]
                )

        for response, image_type in zip(responses, image_types):
            images[image_type] = self.process_image_response(
                response, image_shape=image_shape, image_type=image_type, normalize=normalize, depth_max=depth_max
            )

        return images

    @staticmethod
    def process_image_response(response, image_shape=(144, 256), image_type="rgb", normalize=False, depth_max=50):
        """Processes an image response from AirSim. Returns a numpy array.

        Parameters
        ----------
        image_shape: tuple (Default: (144, 256))
            Height and width of the output images.
            If the AirSim camera settings are different, the images will be resized
        image_type: str
            A string containing either "rgb" or "depth" to indicate the type of image being processed
        normalize: boolean (Default: False)
            If set to True, normalize the pixel values in the returned images to be between -1 and 1
        depth_max: float (Default: 50)
            The maximum that all depth values are clipped to when a depth camera is used.
            Also influences the logarithmic normalize if normalize is set to True
        """
        if "rgb" in image_type:
            # get numpy array
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)

            # reshape array to 4 channel image array H X W X 4
            img_rgb = img1d.reshape(response.height, response.width, 3)

            # original image is flipped vertically
            img_rgb = np.flipud(img_rgb)

            # Optional resize
            if response.height != image_shape[0] or response.width != image_shape[1]:
                # Note: shape needs to be reversed in 'dsize' and the output will be changed to BGR from RGB
                img_rgb = cv2.resize(
                    img_rgb,
                    dsize=(image_shape[1], image_shape[0]),
                    interpolation=cv2.INTER_CUBIC
                )

            # Change from BGR to RGB
            img_rgb = img_rgb[:, :, ::-1]

            if normalize:
                img_rgb = img_rgb.astype("float32")
                img_rgb /= (255 / 2.)
                img_rgb -= 1

            return img_rgb
        elif "depth" in image_type:
            img_depth = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
            img_depth = img_depth.reshape(response.height, response.width, 1)

            # original image is flipped vertically
            img_depth = np.flipud(img_depth)

            # Optional resize
            if response.height != image_shape[0] or response.width != image_shape[1]:
                # Note: shape needs to be reversed in 'dsize' and the output will be changed to BGR from RGB
                img_depth = cv2.resize(
                    img_depth,
                    dsize=(image_shape[1], image_shape[0]),
                    interpolation=cv2.INTER_CUBIC
                )

            img_depth = np.clip(np.abs(img_depth), 0, depth_max)

            if normalize:
                img_depth = img_depth.astype("float32")
                img_depth = (2 * np.log(img_depth + 1) / np.log(depth_max)) - 1

            # resize removes the last dimension, need to reshape again to get it back
            img_depth = img_depth.reshape(image_shape[0], image_shape[1], 1)

            return img_depth
        else:
            raise ValueError(f"Image type must contain 'rgb' or 'depth', but received: {image_type}")
