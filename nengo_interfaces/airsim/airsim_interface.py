import airsim
import setup_path
import pprint
import ast
import re
import numpy as np

class AirsimInterface():
    def connect(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        # not sure if we need this yet
        # self.client.takeoffAsync().join()
        self.is_paused = False


    def disconnect(self):
        self.client.reset()
        self.client.enableApiControl(False)
        self.client.armDisarm(False)

    def pause(self, sim_pause=True):
        """
        Parameters
        ----------
        sim_pause: boolean, Optional (Default: True)
            True to pause, False to resume
        """
        # do not call if already in desired state
        if sim_pause is not self.is_paused:
            self.client.simPause(sim_pause)



    def send_pwm(self, pwm, dt):
        """
        Send PWM controlled signals to each motor in the order
        [front_right_pwm, rear_left_pwm, front_left_pwm, rear_right_pwm]

        Parameters
        ----------
        pwm: list of 4 floats
            pwm signal to each motor in the range of -1 to 1
        dt: float
            the time to run the pwm signal for
        """
        # wait for command to execute
        # self.client.takeoffAsync()
        self.client.moveByMotorPWMsAsync(pwm[0], pwm[1], pwm[2], pwm[3], dt).join()

    def get_feedback(self):
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
        # state = self.client.simGetGroundTruthKinematics()
        # print(state)
        state = self.client.getMultirotorState()
        # print(state)
        # print(state)
        # state is of type <class 'airsim.types.KinematicsState'>
        # convert to string and parse
        state = pprint.pformat(state)
        # remove extra descriptive terms that break dict formatting
        state = re.sub('<[^>]*> ', '', state)
        self.state = ast.literal_eval(state)

        # # angular vel and accel
        # w_a = state['angular_acceleration']
        # w_v = state['angular_velocity']
        # # linear vel and accel
        # a = state['linear_acceleration']
        # v = state['linear_velocity']

        # # position and orientation
        # pos = np.array([
        #         state['position']['x_val'],
        #         state['position']['y_val'],
        #         state['position']['z_val']
        #       ])
        #
        # orient = np.array([
        #             state['orientation']['w_val'],
        #             state['orientation']['x_val'],
        #             state['orientation']['y_val'],
        #             state['orientation']['z_val']
        #          ])
        #NOTE orientation is given in quaternion, not certain about what frame to use
        # for conversion to euluer angles, grabbing roll pitch yaw instead
        state = {
            'pos': np.array([
                self.state['kinematics_estimated']['position']['x_val'],
                self.state['kinematics_estimated']['position']['y_val'],
                -self.state['kinematics_estimated']['position']['z_val']]),
            'lin_vel': np.array([
                self.state['kinematics_estimated']['linear_velocity']['x_val'],
                self.state['kinematics_estimated']['linear_velocity']['y_val'],
                -self.state['kinematics_estimated']['linear_velocity']['z_val']]),
            'ang_vel': np.array([
                self.state['kinematics_estimated']['angular_velocity']['x_val'],
                self.state['kinematics_estimated']['angular_velocity']['y_val'],
                -self.state['kinematics_estimated']['angular_velocity']['z_val']]),
            'quaternion': np.array([
                self.state['kinematics_estimated']['orientation']['w_val'],
                self.state['kinematics_estimated']['orientation']['x_val'],
                self.state['kinematics_estimated']['orientation']['y_val'],
                self.state['kinematics_estimated']['orientation']['z_val']])
        }

        return state

    # NOTE see client.simGetObjectPose and client.sim.SetObjectPose for the following
    def get_orientation(self, name):
        raise NotImplementedError

    def set_orientation(self, name, angles):
        raise NotImplementedError

    def get_xyz(self, name):
        raise NotImplementedError

    def set_xyz(self, name, xyz):
        raise NotImplementedError

if __name__ == '__main__':
    import setup_path
    import airsim
    import pprint

    interface = AirsimInterface()
    interface.connect()

    # FR, RL, FL, RR
    l = 0.69
    h = 0.7
    up = [h, h, h, h]
    down = [0.5, 0.5, 0.5, 0.5]
    forward = [l, h, l, h]
    backward = [h, l, h, l]
    bend_left = [h, l, l, h]
    bend_right = [l, h, h, l]
    rot_left = [h, h, l, l]
    rot_right = [l, l, h, h]

    commands = [up, forward,
                # up, bend_right,
                # up, rot_left,
                # up, rot_right,
                up, backward,
                up, backward,
                down, down,
                down, down,
                down, down,
                # up, up
                ]

    state = interface.get_feedback()
    print(interface.state)
    # steps = [40, 10]
    # for ii, command in enumerate(commands):
    #     step = steps[ii%2]
    #     print('Command: ', command)
    #     pos, orient = interface.get_feedback()
    #     print('pos: ', pos)
    #     print('orient: ', orient)
