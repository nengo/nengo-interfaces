import airsim
# import setup_path
import pprint
import time
import ast
import re
import numpy as np
import math
from abr_control.utils import transformations as transform
from airsim.types import Pose, Vector3r, Quaternionr
# https://github.com/microsoft/AirSim/blob/b7a65bb7f7a9471a2ec0ce6f573512b880d3197a/PythonClient/airsim/client.py#L679

class airsimInterface():
    def __init__(self, dt):
        # self.euler_order = 'rzxy'
        self.euler_order = 'rxyz'
        self.client = airsim.MultirotorClient()
        self.Vector3r = airsim.Vector3r
        self.dt = dt
        self.prev_pos = np.zeros(3)
        self.prev_ori = np.zeros(3)
        self.q_prev = [0, 0, 0, 1]


    def connect(self, takeoff=False, run_async=False):
        self.run_async = run_async
        self.client.confirmConnection()
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        # https://microsoft.github.io/AirSim/apis/#async-methods-duration-and-max_wait_seconds
        # NOTE appending .join() will make the call synchronous (wait for completion)
        if takeoff:
            # takeoff to 3m above the ground
            if self.run_async:
                self.client.takeoffAsync()
            else:
                self.client.takeoffAsync().join()
        self.is_paused = False
        self.client.simPause(self.is_paused)

        # from airsim/airlib/include/vehicles/multirotor/RotorParams.hpp
        C_T = 0.109919
        air_density = 1.225
        # prop diameter
        # prop_diam = 0.2286
        # arm length
        prop_diam = 0.2275
        self.max_thrust = 4.179446268
        self.k = C_T * air_density * prop_diam**4
        self.air_density_ratio = self.client.simGetGroundTruthEnvironment().air_density/air_density
        # print('air density ratio: ', self.air_density_ratio)
        # print('k: ', self.k)
        # time.sleep(1)


    def disconnect(self):

        ext_force = self.Vector3r(0.0, 0.0, 0.0)
        # self.client.simSetExtForce(ext_force)
        self.client.simSetWind(ext_force)
        self.client.reset()
        self.client.enableApiControl(False)
        self.client.armDisarm(False)

    # def get_scene_objects(self, name_regex=''):
    #     return self.client.simListSceneObjects(name_regex)

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



    # def convert_vrep_rotor_order(self, u):
    #     # IN VREP assuming +x is 0 yaw and considered forward: FL, BL, BR, FR
    #     airsim_u = [float(u[3]), float(u[1]), float(u[0]), float(u[2])]
    #     return airsim_u

    def send_motor_commands(self, u):#:, dt):
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
        #NOTE #TODO look into capping the amount of error by setting a max positional error
        # using our max error calculate our max pd output and use this to scale our signal
        # between 0 and 1

        #NOTE not sure how and where we want to handle these type of differences between
        # sim interfaces, leaving it all here for now for simplicity

        # convert our rotor order from vrep
        # print('incoming u: ', u)
        # u = np.asarray(self.convert_vrep_rotor_order(u))
        # print('converted u: ', u)

        # convert from rad/sec to rev/sec
        # u /= 2*np.pi
        # print('rev/sec: ', u)

        # the pwm output is calculated as thrust / (max_thrust * air_density_ratio)
        # where thrust is k*w**2
        # https://github.com/microsoft/AirSim/issues/2592
        # NOTE that the k calculated in airsim uses velocity in units of rev/sec
        # print('max thrust*air density ratio: ', self.max_thrust*self.air_density_ratio)
        # pwm = np.squeeze(self.k*u**2 / (self.max_thrust * self.air_density_ratio))

        pwm = np.squeeze(self.k*u / (self.max_thrust * self.air_density_ratio))
        # pwm = np.squeeze(u)

        # print('final pwm: ', pwm)

        if self.run_async:
            self.client.moveByMotorPWMsAsync(pwm[0], pwm[1], pwm[2], pwm[3], self.dt)
        else:
            self.client.moveByMotorPWMsAsync(pwm[0], pwm[1], pwm[2], pwm[3], self.dt).join()

    def _convert_to_airsim_quat(self, euler, rotate=False):
        """
        if rotate is false the nwe just convert our euler to quat and reorder to xyzw
        """
        # https://github.com/microsoft/AirSim/blob/b7a65bb7f7a9471a2ec0ce6f573512b880d3197a/PythonClient/airsim/client.py#L285
        quat = transform.quaternion_from_euler(euler[0], euler[1], euler[2], self.euler_order)

        # rotate targets to match airsim state, this should be false for set_state
        if rotate:
            # rotate by y to add airsims weird rotation

            ang = np.pi/2
            #Y
            rot_matrix = np.array([
                [np.cos(ang), 0, np.sin(ang)],
                [0, 1, 0],
                [-np.sin(ang), 0, np.cos(ang)]
            ])

            # quaternion to rotate airsim feedback
            quat_rotation = transform.quaternion_from_matrix(rot_matrix)

            # rotate our target quat by yaw to get our rotation of our drone
            quat = transform.quaternion_multiply(quat, quat_rotation)

        final_quat = [float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])]
        return final_quat


    def _convert_from_airsim_quat(self, airsim_quat):
        # airsim has quaternion order xyzw, reorder to match our math to wxyz
        quat = np.array([airsim_quat[3], airsim_quat[0], airsim_quat[1], airsim_quat[2]])

        # airsim rotate by y for some reason, fix this
        # get our current orientation as a quaternion
        ang = np.pi/2

        #Y
        rot_matrix = np.array([
            [np.cos(ang), 0, np.sin(ang)],
            [0, 1, 0],
            [-np.sin(ang), 0, np.cos(ang)]
        ])


        # quaternion to rotate airsim feedback
        quat_rotation = transform.quaternion_from_matrix(rot_matrix)

        # rotate our target quat by yaw to get our rotation of our drone
        final_quat = transform.quaternion_multiply(quat, quat_rotation)
        return final_quat

    def airsim_ang_vel(self, q, q_prev):
        """
        z-y-x rotation convention (Tait-Bryan angles) 
        Apply yaw, pitch and roll in order to front vector (+X)
        http://www.sedris.org/wg8home/Documents/WG80485.pdf
        http://www.ctralie.com/Teaching/COMPSCI290/Materials/EulerAnglesViz/
        """

        # def _to_euler(quat):
        #     ysqr = quat[1] * quat[1]
        #
        #     # roll (x-axis rotation)
        #     t0 = 2.0 * (quat[3] * quat[0] + quat[1] * quat[2])
        #     t1 = 1.0 - 2.0 * (quat[0] * quat[0] + ysqr)
        #     roll = math.atan2(t0, t1)
        #
        #     # pitch (y-axis rotation)
        #     t2 = 2.0 * (quat[3] * quat[1] - quat[2] * quat[0])
        #     if t2 > 1.0:
        #         t2 = 1.0
        #     elif t2 < -1.0:
        #         t2 = -1.0
        #
        #     pitch = np.arcsin(t2)
        #
        #     # yaw (z-axis rotation)
        #     t3 = 2.0 * (quat[3] * quat[2] + quat[0] * quat[1])
        #     t4 = 1.0 - 2.0 * (ysqr + quat[2] * quat[2])
        #     yaw = math.atan2(t3, t4)
        #
        #     return pitch, roll, yaw
        def _to_euler(quat):
            quat = np.array([quat[3], quat[0], quat[1], quat[2]])
            to_euler = transform.euler_from_quaternion(quat, self.euler_order)
            euler = self.convert_angles(to_euler)
            return euler

        r_s, p_s, y_s = _to_euler(q_prev)
        r_e, p_e, y_e = _to_euler(q)

        p_rate = (p_e - p_s) / self.dt
        r_rate = (r_e - r_s) / self.dt
        y_rate = (y_e - y_s) / self.dt

        #TODO: optimize below
        #Sec 1.3, https://ocw.mit.edu/courses/mechanical-engineering/2-154-maneuvering-and-control-of-surface-and-underwater-vehicles-13-49-fall-2004/lecture-notes/lec1.pdf
        wx = r_rate + 0 - y_rate * np.sin(p_e)
        wy = 0 + p_rate * np.cos(r_e) + y_rate * np.sin(r_e) * np.cos(p_e)
        wz = 0 - p_rate * np.sin(r_e) + y_rate * np.cos(r_e) * np.cos(p_e);

        return np.array([wx, wy, wz])

    def get_feedback(self, axes='rxyz'):
        """
        Calls the simGetGroundTruthKinematics to get system feedback, which is then
        parsed from the airsim custom type to a dict

        returns dict of state in the form:
            [x, y, z, dx, dy, dz, roll, pitch, yaw, droll, dpitch, dyaw]

        the full state can be accessed from self.state with the main keys:
            - landed_state
            - collision
            - timestamp
            - kinematics estimated
            - rc_data
            - gps_location
        """
        state = self.client.getMultirotorState().kinematics_estimated
        # convert to numpy array
        pos = state.position.to_numpy_array()
        # NOTE use this to go through the quaternion rotation
        # quat = self._convert_from_airsim_quat(state.orientation.to_numpy_array())

        airsim_quat = state.orientation.to_numpy_array()
        quat = np.array([airsim_quat[3], airsim_quat[0], airsim_quat[1], airsim_quat[2]])

        euler = np.asarray(transform.euler_from_quaternion(quat, self.euler_order))

        # lin_vel = (pos-self.prev_pos)/self.dt
        lin_vel = 2*state.linear_velocity.to_numpy_array()
        # self.prev_pos = np.copy(pos)

        ang_vel = 2*state.angular_velocity.to_numpy_array()

        # https://phys.libretexts.org/Bookshelves/Classical_Mechanics/Book%3A_Classical_Mechanics_(Tatum)/04%3A_Rigid_Body_Rotation/4.02%3A_Angular_Velocity_and_Eulerian_Angles
        # dphi = (euler[0] - self.prev_ori[0]) / self.dt
        # dtheta = (euler[1] - self.prev_ori[1]) / self.dt
        # dpsi = (euler[2] - self.prev_ori[2]) / self.dt
        # theta = euler[1]
        # phi = euler[0]
        # psi = euler[2]
        # w1 = dphi * np.sin(theta) * np.sin(psi) + dtheta * np.cos(psi)
        # w2 = dphi * np.sin(theta) * np.cos(psi) - dtheta * np.sin(psi)
        # w3 = dphi * np.cos(theta) + dpsi
        # ang_vel = [w1, w2, w3]

        # Backward Difference
        # ang_vel_mine = (euler-self.prev_ori)/self.dt

        # https://math.stackexchange.com/questions/1422258/derive-euler-angles-derivative-from-angular-velocity
        # ang_vel_mine = (airsim_quat[:3] - self.q_prev[:3])/self.dt

        # https://answers.unity.com/questions/49082/rotation-quaternion-to-angular-velocity.html#:~:text=The%20angular%20velocity%20is%20indeed,and%20the%20angle%20IN%20DEGREES.
        # ang_vel_mine = 2*np.log(airsim_quat * np.conj(self.q_prev))

        # https://answers.unity.com/questions/49082/rotation-quaternion-to-angular-velocity.html#:~:text=The%20angular%20velocity%20is%20indeed,and%20the%20angle%20IN%20DEGREES.
        # ang_vel_mine = 2 * airsim_quat[:3] / abs(airsim_quat[:3]) * np.arccos(airsim_quat[3])

        # Airsim calculation
        # ang_vel_mine = self.airsim_ang_vel(q=airsim_quat, q_prev=self.q_prev)
        # ang_vel = ang_vel_mine

        # https://physics.stackexchange.com/questions/420695/how-to-derive-the-relation-between-euler-angles-and-angular-velocity-and-get-the
        # dphi = (euler[0] - self.prev_ori[0]) / self.dt
        # dtheta = (euler[1] - self.prev_ori[1]) / self.dt
        # dpsi = (euler[2] - self.prev_ori[2]) / self.dt
        # theta = euler[1]
        # phi = euler[0]
        # psi = euler[2]
        # T1 = np.array([1, 0, 0]).T
        # T2 = np.array([0, np.cos(phi), np.sin(phi)]).T
        # T3 = np.array([
        #     np.sin(theta),
        #     -np.sin(phi)*np.cos(theta),
        #     np.cos(phi)*np.cos(theta)]).T
        #
        # ang_vel_mine = T1 * dphi + T2*dtheta + T3*dpsi

        # self.q_prev = np.copy(airsim_quat)
        # self.prev_ori = np.copy(euler)
        euler = self.convert_angles(euler)

        state = {
            'pos': pos, 'lin_vel': lin_vel, 'quat': quat, 'ang_vel': ang_vel,
            'ori': euler#, 'ang_vel_mine': ang_vel_mine
        }

        return state

    def get_camera_feedback(self, name):
        NotImplementedError

    # def convert_angles(self, ang):
    #     #TODO compare this to abr_control.utils.transformations and use that
    #     # if we can since that isn't hardcoded to just zxy conversion
    #     """ Converts Euler angles from x-y-z to z-x-y convention """
    #
    #     def b(num):
    #         """ forces magnitude to be 1 or less """
    #         if abs( num ) > 1.0:
    #             return math.copysign( 1.0, num )
    #         else:
    #             return num
    #
    #     s1 = math.sin(ang[0])
    #     s2 = math.sin(ang[1])
    #     s3 = math.sin(ang[2])
    #     c1 = math.cos(ang[0])
    #     c2 = math.cos(ang[1])
    #     c3 = math.cos(ang[2])
    #
    #     pitch = math.asin(b(c1*c3*s2-s1*s3) )
    #     cp = math.cos(pitch)
    #     # just in case
    #     if cp == 0:
    #         cp = 0.000001
    #
    #     yaw = math.asin(b((c1*s3+c3*s1*s2)/cp) ) #flipped
    #     # Fix for getting the quadrants right
    #     if c3 < 0 and yaw > 0:
    #         yaw = math.pi - yaw
    #     elif c3 < 0 and yaw < 0:
    #         yaw = -math.pi - yaw
    #
    #     roll = math.asin(b((c3*s1+c1*s2*s3)/cp) ) #flipped
    #     return [roll, pitch, yaw]


    # def rot_quat_about_x(self, quat):
    #     rot_matrix = np.array([
    #         [1, 0, 0],
    #         [0, np.cos(np.pi), -np.sin(np.pi)],
    #         [0, np.sin(np.pi), np.cos(np.pi)]
    #     ])
    #     rot_quat = transform.quaternion_from_matrix(rot_matrix)
    #     rotated_quat = transform.quaternion_multiply(quat, rot_quat)
    #     return rotated_quat

    def get_state(self, name):
        raw_pose = self.client.simGetObjectPose(name)
        pos = raw_pose.position.to_numpy_array()
        quat = self._convert_from_airsim_quat(raw_pose.orientation.to_numpy_array())
        euler = transform.euler_from_quaternion(quat, self.euler_order)
        pose = {'pos': pos, 'quat': quat, 'ori': euler}

        return pose

    def set_state(self, name, xyz=None, orientation=None):
        """
        Parameters
        ----------
        name: string
            the name of the object
        xyz: 3d array or list
            the desired xyz position in meters
            if you do not want to set the position, set to None
        orientation: 4d array or list
            the desired quaternion orientation of the object in order [xyzw]
            if you do not want to set the orientation, set to None
        """
        # https://github.com/microsoft/AirSim/blob/b7a65bb7f7a9471a2ec0ce6f573512b880d3197a/PythonClient/airsim/client.py#L285
        # xyz = [float(xyz[0]), -1*float(xyz[1]), -1*float(xyz[2])]
        xyz = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        # rotate about x by pi to return to NED coordinates
        # orientation = self.rot_quat_about_x(orientation)
        # airsim quaternions are in the order xyzw
        # orientation = transform.quaternion_from_euler(orientation[0], orientation[1], orientation[2], self.euler_order)
        # orientation = [float(orientation[1]), float(orientation[2]), float(orientation[3]), float(orientation[0])]
        quat = self._convert_to_airsim_quat(orientation, rotate=False)
        # orientation = np.asarray(orientation, 'float32')
        pose = Pose(
                position_val=Vector3r(
                    x_val=xyz[0], y_val=xyz[1], z_val=xyz[2]
                ),
                orientation_val=Quaternionr(
                    x_val=quat[0], y_val=quat[1], z_val=quat[2], w_val=quat[3]
                )

        )
        self.client.simSetObjectPose(name, pose, teleport=True)

    def send_external_forces(self, xyz):
        raise NotImplementedError

    def set_weather(self):
        # def simEnableWeather(self, enable):
        #     """
        #     Enable Weather effects. Needs to be called before using `simSetWeatherParameter` API
        #     Args:
        #         enable (bool): True to enable, False to disable
        #     """
        #     self.client.call('simEnableWeather', enable)
        #
        # def simSetWeatherParameter(self, param, val):
        #     """
        #     Enable various weather effects
        #     Args:
        #         param (WeatherParameter): Weather effect to be enabled
        #         val (float): Intensity of the effect, Range 0-1
        #     """
        #     self.client.call('simSetWeatherParameter', param, val)

        raise NotImplementedError

    def convert_angles(self, ang):
        """ Converts Euler angles from x-y-z to z-x-y convention """

        def b(num):
            """ forces magnitude to be 1 or less """
            if abs( num ) > 1.0:
                return math.copysign( 1.0, num )
            else:
                return num

        s1 = math.sin(ang[0])
        s2 = math.sin(ang[1])
        s3 = math.sin(ang[2])
        c1 = math.cos(ang[0])
        c2 = math.cos(ang[1])
        c3 = math.cos(ang[2])

        pitch = math.asin(b(c1*c3*s2-s1*s3) )
        cp = math.cos(pitch)
        # just in case
        if cp == 0:
            cp = 0.000001

        yaw = math.asin(b((c1*s3+c3*s1*s2)/cp) ) #flipped
        # Fix for getting the quadrants right
        if c3 < 0 and yaw > 0:
            yaw = math.pi - yaw
        elif c3 < 0 and yaw < 0:
            yaw = -math.pi - yaw

        roll = math.asin(b((c3*s1+c1*s2*s3)/cp) ) #flipped
        return [roll, pitch, yaw]



# if __name__ == '__main__':
#     import setup_path
#     import airsim
#     import pprint
#
#     try:
#         interface = airsimInterface()
#         interface.connect()
#
#         # FR, RL, FL, RR
#         l = 0.69
#         h = 0.7
#         up = [h, h, h, h]
#         down = [0.5, 0.5, 0.5, 0.5]
#         forward = [l, h, l, h]
#         backward = [h, l, h, l]
#         bend_left = [h, l, l, h]
#         bend_right = [l, h, h, l]
#         rot_left = [h, h, l, l]
#         rot_right = [l, l, h, h]
#
#         commands = [up, forward,
#                     # up, bend_right,
#                     # up, rot_left,
#                     # up, rot_right,
#                     up, backward,
#                     up, backward,
#                     down, down,
#                     down, down,
#                     down, down,
#                     # up, up
#                     ]
#
#         # state = interface.get_feedback()
#         # print(state)
#         import time
#         # print('drone feedback: ', interface.get_feedback())
#         state = interface.get_state('filtered_target')
#         print('filtered_target pose: ', state)
#         # time.sleep(1)
#         new_pos = state['pos'] + np.array([0, 0, 2])
#         interface.set_state(xyz=new_pos, orientation=state['quat'], name='filtered_target')
#         print('filtered_target pose new: ', interface.get_state('filtered_target'))
#         steps = [4, 1]
#         for ii, command in enumerate(commands):
#             step = steps[ii%2]
#             print('Command: ', command)
#             state = interface.get_feedback()
#             interface.send_motor_commands(command, step)
#             print('pos: ', state['pos'])
#             print('orient: ', state['quat'])
#     finally:
#         interface.disconnect()
