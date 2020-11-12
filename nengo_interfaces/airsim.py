import pprint
import ast
import numpy as np
import time

from abr_control.utils import transformations as transform
import airsim
from airsim.types import Pose, Vector3r, Quaternionr
import math
import nengo

# https://github.com/microsoft/AirSim/blob/b7a65bb7f7a9471a2ec0ce6f573512b880d3197a/PythonClient/airsim/client.py#L679


class AirSim(nengo.Process):
    """Provides an easy to use API for AirSim in Nengo models.

    NOTES:
    - AirSim returns quaternions in the [x, y, z, w] format. We convert this to the
    [w, x, y, z] format for all output and expect [w, x, y, z] for all input.
    - All Euler angles are output / expected to be in Tait-Bryan form.


    Parameters
    ----------
    dt : float, optional (Default: 0.01)
        The time step for the simulation dynamics.
    run_async : boolean, optional (Default: False)
        If true, run AirSim simulation independent of Nengo simulation. If false,
        run the two in lockstep, i.e., one Nengo time step then one AirSim time step
    render_params : dict, optional (Default: None)
        'cameras' : list
        'resolution' : list
        'update_frequency' : int, optional (Default: 1)
        'plot_frequency' : int, optional (Default: None)
    track_input : bool, option (Default: False)
        If True, store the input that is passed in to the Node in self.input_track.
    input_bias : int, optional (Default: 0)
        Added to the input signal to the Mujoco environment, after scaling.
    input_scale : int, optional (Default: 1)
        Multiplies the input signal to the Mujoco environment, before biasing.
    seed : int, optional (Default: None)
        Set the seed on the rng.
    takeoff : boolean, optional (Default: True)
        If true, the drone will take off and hover in the air upon connection.
        To prevent this action, set to false.
    """

    def __init__(
        self,
        dt=0.01,
        run_async=False,
        render_params=None,
        # track_input=False,
        input_bias=0,
        input_scale=1,
        seed=None,
        takeoff=False
    ):
        self.dt = dt
        # empirically determined function from recorded velocity feedback from various dt
        self.vel_scale = 1/(30.55 * self.dt + 0.03)
        # self.vel_scale = 2

        self.run_async = run_async
        self.render_params = render_params if render_params is not None else {}
        # self.track_input = track_input
        self.input_bias = input_bias
        self.input_scale = input_scale
        self.takeoff = takeoff

        # self.input_track = []
        self.euler_order = "rxyz"
        self.Vector3r = airsim.Vector3r

        self.image = np.array([])
        # if self.render_params:
        #     # if not empty, create array for storing rendered camera feedback
        #     self.camera_feedback = np.zeros(
        #         (
        #             self.render_params["resolution"][0],
        #             self.render_params["resolution"][1]
        #             * len(self.render_params["cameras"]),
        #             3,
        #         )
        #     )
        #     self.subpixels = np.product(self.camera_feedback.shape)
        #     self.image = np.zeros(self.subpixels)
        #     if "frequency" not in self.render_params.keys():
        #         self.render_params["frequency"] = 1
        #     if "plot_frequency" not in self.render_params.keys():
        #         self.render_params["plot_frequency"] = None

        self.client = airsim.MultirotorClient()

        # from airsim/airlib/include/vehicles/multirotor/RotorParams.hpp
        C_T = 0.109919
        air_density = 1.225
        prop_diam = 0.2275
        self.max_thrust = 4.179446268
        self.k = C_T * air_density * prop_diam ** 4
        self.air_density_ratio = (
            self.client.simGetGroundTruthEnvironment().air_density / air_density
        )

        # size_in = the number of rotors on our quadcopter, hardcoded
        size_in = 4
        # size_out = number of parameters in our feedback array, which is of the form
        # [x, y, z, dx, dy, dz, roll, pitch, yaw, droll, dpitch, dyaw]
        size_out = 12
        super().__init__(size_in, size_out, default_dt=dt, seed=seed)

    def connect(self):
        self.client.confirmConnection()
        self.client.reset()
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
        # self.is_paused = False
        self.is_paused = True
        self.client.simPause(self.is_paused)
        # initialize wind and payload to zero
        zeros = self.Vector3r(0, 0, 0)
        self.client.simSetWind(zeros)
        # self.client.simSetExtForce(zeros)

    def disconnect(self):

        self.client.simPause(False)
        ext_force = self.Vector3r(0.0, 0.0, 0.0)
        # self.client.simSetExtForce(ext_force)
        self.client.simSetWind(ext_force)

        self.client.reset()
        self.client.enableApiControl(False)
        self.client.armDisarm(False)
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

        # self.connect()

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
            u += self.input_bias
            u *= self.input_scale

            # print("u: ", [float("%.3f" % val) for val in u])
            self.send_pwm_signal(u)
            feedback = self.get_feedback()

            # render camera feedback
            if self.render_params:
                raise NotImplementedError

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
            This function accepts rotor velocities in rad/sec and converts them to pwm
        dt: float
            the time to run the pwm signal for
        """
        # the pwm output is calculated as thrust / (max_thrust * air_density_ratio)
        # where thrust is k*w**2 -  https://github.com/microsoft/AirSim/issues/2592
        # NOTE that the k calculated in airsim uses velocity in units of rev/sec
        pwm = np.squeeze(self.k * u / (self.max_thrust * self.air_density_ratio))

        # NOTE appending .join() will make the call synchronous (wait for completion)
        if self.run_async:
            self.client.moveByMotorPWMsAsync(pwm[0], pwm[1], pwm[2], pwm[3], self.dt)
        else:
            self.client.simPause(False)
            # print("pwm: ", pwm)
            # print("dt: ", self.dt)
            self.client.moveByMotorPWMsAsync(
                pwm[0], pwm[1], pwm[2], pwm[3], self.dt
            ).join()
            time.sleep(self.dt)
            self.client.simPause(True)

    def get_feedback(self):
        """
        Calls the simGetGroundTruthKinematics to get system feedback, which is then
        parsed from the airsim custom type to a dict

        returns dict of state in the form:
            {position, quaternion(in format w,x,y,z), taitbryan(euler in taitbryan angles)}

        the full state can be accessed from self.state with the main keys:
            - landed_state
            - collision
            - timestamp
            - kinematics estimated
            - rc_data
            - gps_location
        """
        state = self.client.getMultirotorState().kinematics_estimated
        pos = state.position.to_numpy_array()

        airsim_quat = state.orientation.to_numpy_array()
        # NOTE: The quadcopter feedback does not need to be rotated to be in the
        # expected axes, unlike object orientation feedback.
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

    def get_camera_feedback(self, name):
        NotImplementedError

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

        orientation = transform.quaternion_from_euler(orientation[0], orientation[1], orientation[2], self.euler_order)

        # reorder to xyzw to match the quaternion format of airsim
        quat = [float(orientation[1]), float(orientation[2]), float(orientation[3]), float(orientation[0])]
        pose = Pose(
                position_val=Vector3r(
                    x_val=xyz[0], y_val=xyz[1], z_val=xyz[2]
                ),
                orientation_val=Quaternionr(
                    x_val=quat[0], y_val=quat[1], z_val=quat[2], w_val=quat[3]
                )

        )
        self.client.simSetObjectPose(name, pose, teleport=True)

    def quat_to_taitbryan(self, quat):
        """Convert quaternion to Tait-Bryan Euler angles

        quat : np.array
            The quaternion in [w, x, y, z] format
        """
        euler = np.asarray(transform.euler_from_quaternion(quat, self.euler_order))
        euler = self.ea_xyz_to_zxy(euler)

        return euler

        # return np.array(
        #     [
        #         np.arctan2(
        #             quat[2] * quat[3] + quat[0] * quat[1],
        #             1 / 2 - (quat[1] ** 2 + quat[2] ** 2),
        #         ),
        #         np.arcsin(-2 * (quat[1] * quat[3] - quat[0] * quat[2])),
        #         np.arctan2(
        #             quat[1] * quat[2] + quat[0] * quat[3],
        #             1 / 2 - (quat[2] ** 2 + quat[3] ** 2),
        #         ),
        #     ]
        # )
        #
    def ea_xyz_to_zxy(self, ang):
        """ Converts Euler angles from x-y-z to z-x-y convention """

        def b(num):
            """ forces magnitude to be 1 or less """
            if abs(num) > 1.0:
                return np.copysign(1.0, num)
            else:
                return num

        s1 = np.sin(ang[0])
        s2 = np.sin(ang[1])
        s3 = np.sin(ang[2])
        c1 = np.cos(ang[0])
        c2 = np.cos(ang[1])
        c3 = np.cos(ang[2])

        pitch = math.asin(b(c1 * c3 * s2 - s1 * s3))
        cp = np.cos(pitch)
        if cp == 0:
            cp = 1e-6

        yaw = math.asin(b((c1 * s3 + c3 * s1 * s2) / cp))  # flipped
        # Fix for getting the quadrants right
        if c3 < 0 and yaw > 0:
            yaw = np.pi - yaw
        elif c3 < 0 and yaw < 0:
            yaw = -np.pi - yaw

        roll = math.asin(b((c3 * s1 + c1 * s2 * s3) / cp))  # flipped
        return [roll, pitch, yaw]

if __name__ == '__main__':
    import nengo
    import matplotlib.pyplot as plt

    dt = 0.01
    time = 3
    control_u = np.array([6800, 6800, 7000, 6800])

    # RUN THROUGH NENGO NODE
    interface = AirSim(dt=dt)
    interface.connect()
    net = nengo.Network(seed=0)

    with net:
        interface_node = nengo.Node(interface)
        u = nengo.Node(lambda x:control_u, size_in=0, size_out=4)
        nengo.Connection(u, interface_node)
        state_probe = nengo.Probe(interface_node)
    sim = nengo.Simulator(net, dt=0.01, seed=0)
    sim.run(time)

    interface.disconnect()

    # NENGO NODE CLASS WITHOUT USING NENGO
    interface = AirSim(dt=dt)
    interface.connect()
    state = []

    labs = ['x', 'y', 'z', 'dx', 'dy', 'dz', 'a', 'b', 'g', 'da', 'db', 'dg']
    for ii in range(0, int(time/dt)):
        output = interface.get_feedback()
        output = np.hstack((
            np.hstack((output['position'], output['linear_velocity'])),
            np.hstack((output['taitbryan'], output['angular_velocity']))
        ))

        state.append(output)
        interface.send_pwm_signal(control_u)

    state = np.asarray(state).T
    interface.disconnect()

    # OLD INTERFACE WITHOUT NENGO
    # from airsim_interface import airsimInterface
    # interface = airsimInterface(dt=dt)
    # interface.connect()
    # state_old = []
    #
    # labs = ['x', 'y', 'z', 'dx', 'dy', 'dz', 'a', 'b', 'g', 'da', 'db', 'dg']
    # for ii in range(0, int(time/dt)):
    #     output = interface.get_feedback()
    #     output = np.hstack((
    #         np.hstack((output['pos'], output['lin_vel'])),
    #         np.hstack((output['ori'], output['ang_vel']))
    #     ))
    #
    #     state_old.append(output)
    #     interface.send_motor_commands(control_u)
    #
    # state_old = np.asarray(state_old).T
    # interface.disconnect()
    #
    #
    #
    # COMPARE
    plt.figure()
    data = sim.data[state_probe].T
    print(data.shape)
    for ii in range(0, len(data)):
        plt.subplot(len(data), 1, ii+1)
        plt.plot(data[ii], label='%s Node' % labs[ii])
        plt.plot(state[ii], label='%s Node No Nengo' % labs[ii])
        # plt.plot(state_old[ii], label='%s Old Airsim' % labs[ii])
        plt.legend()
    plt.show()
