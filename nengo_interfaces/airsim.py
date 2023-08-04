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
    sleep_dt: bool, Optional (Default: False)
        adds a sleep of length dt to run control in lock step. When true the sim
        will unpause, run the pwm (or move camera) command, sleep for dt seconds,
        and then pause the sim. This allows for slower simulations to be run in the
        background while maintaining a constant ue4 sim step.
        NOTE: run_async has to be False, otherwise the sleep is ignored and commands
        are run asynchronously
    """

    def __init__(  # noqa: C901
        self,
        dt=0.01,
        run_async=False,
        camera_params=None,
        seed=None,
        takeoff=False,
        show_display=True,
        sleep_dt=False
    ):
        assert not (sleep_dt and run_async), "Cannot run asynchronously and sleep for dt seconds"
        self.dt = dt
        self.sleep_dt = sleep_dt
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
            if self.sleep_dt:
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

    def set_scale(self, name, scale):
        scale_vec = Vector3r(scale[0], scale[1], scale[2])
        self.client.simSetObjectScale(name, scale_vec)

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

        if self.run_async:
            self.client.simSetCameraPose(name, pose)  # , teleport=True)
        else:
            self.client.simPause(False)
            self.client.simSetCameraPose(name, pose)  # , teleport=True)
            if self.sleep_dt:
                time.sleep(self.dt)
            self.client.simPause(True)

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

# obtained from https://microsoft.github.io/AirSim/seg_rgbs.txt
# color map used for setting segmentation colors and IDs in Airsim
ID_TO_COLOR_MAP = {100: [185, 243, 231]}

def retry(img_func, n_tries=3):
    """
    Wrapper function to retry Airsim image feedback function 3 times in the event
    of a client error. Should be able to handle erroneous situation where Airsim returns
    empty image.

    Parameters:
    -----------
    img_func : function
        A function that returns an image of shape (H,W,C) obtained by airsim.client.simGetImages()
    n_tries : int
        Number of attemps to run the function

    Returns:
    --------
    wrapper : function
        A wrapped function that checks if the returned image is empty and retries if that
        is the case
    """

    def wrapper(*args):
        for _ in range(n_tries):
            img_return = img_func(*args)
            if img_return.shape[0] != 0 and img_return.shape[1] != 0:
                return img_return
        return None

    return wrapper


class CaptureSingleTargetXYDepth(AirSim):
    """
    Extend the AirSim interface to capture pixel location and depth camera feedback
    for a single UE4 character/object target. The character/object is chosen
    at random from objects tagged with a specific wildcard (i.e. ABR).

    To be used inside of an Airsim simulation loop.

    Attributes:
    ----------
    seg_id : int
        An integer corresponding to the Airsim segmentation ID of interest
    """

    def __init__(self, seg_id=100, **kwargs):
        super(CaptureSingleTargetXYDepth, self).__init__(**kwargs)
        self.seg_id = seg_id
        self.reset()

    def reset(self):
        self.images = []
        self.ground_truth = []

    def set_target_tag(self, target_tag):
        """
        Sets the class variable for the object's UE4 nametag.
        Also collects the initial pose (location) to return
        the object to that location when it's no longer needed.
        """

        self.target_tag = target_tag
        self.reset_pose = self.client.simGetObjectPose(self.target_tag)

    def prepare_segmentation(self):
        """Sets the segmentation ID for the specified target"""

        success = self.client.simSetSegmentationObjectID(
            self.target_tag, self.seg_id, is_name_regex=True
        )
        assert success

    def test_line_of_sight(self, pixel_threshold=20):
        """
        Utility function that checks if the camera, at its current position,
        can identify the target object within it's field of view from a
        segmentation image.
        """

        # Get a segmentation image from Airsim
        seg_img_bgr = self.segmentation_feedback()
        seg_img_rgb = seg_img_bgr[:, :, ::-1]

        # Check if any pixels in the segmentation image match the
        # target objects segmention ID/colour
        mask = np.all(
            seg_img_rgb == np.expand_dims(ID_TO_COLOR_MAP[self.seg_id], axis=(0, 1)),
            axis=2,
        )

        # The number of target object pixels located in the segmentation image
        # can be found by summing the mask
        n_pixels = np.sum(mask)

        # If the number of pixels is greater than the threshold value, we say
        # the target object is in frame
        if n_pixels > pixel_threshold:
            return True

        return False

    @retry
    def segmentation_feedback(self, camera_name="0"):
        """
        Function to obtain segmentation image feedback from the Airsim
        client
        """
        # obtain segmentation image at current step
        seg_img = self.client.simGetImages(
            [
                airsim.ImageRequest(
                    camera_name, airsim.ImageType.Segmentation, False, False
                )
            ]
        )[0]

        # reshape segmented rgb image to 3 channel image array H X W X 3
        seg_img1d = np.fromstring(seg_img.image_data_uint8, dtype=np.uint8)
        seg_img_rgb = seg_img1d.reshape(seg_img.height, seg_img.width, 3)
        return seg_img_rgb

    @retry
    def depth_feedback(self, camera_name="0"):
        """
        Function to obtain depth image feedback from the Airsim
        client
        """
        # obtain planar depth information at current step
        dep_img = self.client.simGetImages(
            [
                airsim.ImageRequest(
                    camera_name, airsim.ImageType.DepthPlanar, True, False
                )
            ]
        )[0]

        # reshape planar depth images into float array H X W
        depth_img1d = np.array(dep_img.image_data_float).astype(np.float32)
        depth_img = depth_img1d.reshape(dep_img.height, dep_img.width)
        return depth_img

    def capture_step(self, segmentation_threshold=16):
        """
        Funcation that will collect rgb camera, segmentation camera, and
        depth camera feedback utilizing the configured nengo Airsim interface.

        This function should be called inside an Airsim simulation loop, where
        image and ground truth information is captured every time this function
        is called.

        Parameters:
        -----------
        segmentation_threshold : int
            Number of pixels returned by segmentation ID that
            is accetable for identifying the ground truth
            location.

        """
        # obtain default camera image at current step
        camera_feedback = self.get_camera_feedback()

        # get segmentation feedback
        seg_img_bgr = self.segmentation_feedback()

        # get depth feedback
        depth_img = self.depth_feedback()

        # Check if any pixels in the segmentation image match the
        # target objects segmention ID/colour
        mask = np.all(
            seg_img_bgr[:, :, ::-1]
            == np.expand_dims(ID_TO_COLOR_MAP[self.seg_id], axis=(0, 1)),
            axis=2,
        )
        object_in_view = np.any(mask)

        # If no pixels match or the number of pixels is below the threshold,
        # set the ground truth location to sentinel value of -1
        if not object_in_view or np.sum(mask) < segmentation_threshold:
            # if the object is not in the camera capture, then set the center
            # to be some dummy value
            true_loc = np.array([-1, -1, -1])
        else:
            # calculate centre of mass of pixels in segmentation image
            center = np.mean(np.argwhere(mask), axis=0).astype(int)
            if depth_img is None:
                depth = -1
            else:
                depth = depth_img[center[0], center[1]]
            true_loc = np.array([center[0], center[1], depth]).astype(np.float32)

        self.images.append(camera_feedback)
        self.ground_truth.append(true_loc)
