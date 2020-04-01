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
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        # not sure if we need this yet
        # self.client.takeoffAsync().join()


    def disconnect(self):
        raise NotImplementedError


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
        self.client.moveByMotorPWMsAsync(pwm[0], pwm[1], pwm[2], pwm[3], dt).join()

    def get_feedback(self):
        """
        Calls the simGetGroundTruthKinematics to get system feedback, which is then
        parsed from the airsim custom type to a dict

        Returns cartesian position and quaternion orientation
        """
        state = self.client.simGetGroundTruthKinematics()
        # state is of type <class 'airsim.types.KinematicsState'>
        # convert to string and parse
        state = pprint.pformat(state)
        state = re.sub('<KinematicsState> ', '', state)
        state = re.sub('<Vector3r> ', '', state)
        state = re.sub('<Quaternionr> ', '', state)
        state = ast.literal_eval(state)

        # # angular vel and accel
        # w_a = state['angular_acceleration']
        # w_v = state['angular_velocity']
        # # linear vel and accel
        # a = state['linear_acceleration']
        # v = state['linear_velocity']

        # position and orientation
        pos = np.array([
                state['position']['x_val'],
                state['position']['y_val'],
                state['position']['z_val']
              ])

        orient = np.array([
                    state['orientation']['w_val'],
                    state['orientation']['x_val'],
                    state['orientation']['y_val'],
                    state['orientation']['z_val']
                 ])

        return pos, orient

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

    print('moving up')
    steps = 10
    for ii in range(0,steps):
        scale = 0.8
        pwm = np.array([1, 1, 1, 1])
        interface.send_pwm(pwm*scale, dt=0.001)
    for ii in range(0,steps):
        scale = 0.8
        pwm = np.array([-1, 1, -1, 1])
        interface.send_pwm(pwm*scale, dt=0.001)

        pos, orient = interface.get_feedback()
        print('pos: ', pos)
        print('orient: ', orient)
