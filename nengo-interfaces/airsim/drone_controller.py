import numpy as np
import math
import airsim_interface
import matplotlib.pyplot as plt
from abr_control.utils import transformations as t

class DroneController():
    def __init__(self):
        #TODO should move these constants to a drone config

        #TODO what are k and b?
        # drag force constants
        self.A = np.array([
            [k, k, k, k, ],
            [0, -l*k, 0, l*k],
            [-l*k, 0, l*k, 0],
            [-b, b, -b, b]
        ])

        # lift constant
        self.K = np.array([
            [0, 0, k2, 0, 0, -k4, 0, 0, 0, 0, 0, 0],
            [0, k1, 0, 0, -k3, 0, -k5, 0, 0, k7, 0, 0],
            [-k1, 0, 0, k3, 0, 0, 0, -k5, 0, 0, k7, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -k6, 0, 0, k8]
        ])

        # transform to rotor vel space
        self.TR = np.array([
            [1, -1, 1, 1],
            [1, -1, -1, -1],
            [1, 1, -1, 1],
            [1, 1, 1, -1]
        ])

        # drag constant
        self.b = [1, 1, 1, 1]
        # distance from rotor to COM
        self.l = 1
        # drone inertia matrix
        self.Ixx = None
        self.Iyy = None
        self.Izz = None
        self.I = np.array([
            [self.Ixx, 0, 0],
            [0, self.Iyy, 0],
            [0, 0, self.Izz]
        ])
        # rotor moment of inertia
        self.Mi = None
        # mass
        self.m = 1

        # gravity term
        self.g = np.array([0, 0, -9.81])


    # TODO are any of the dynamics equations even used?
    # dynamics equations
    def _R(self, q):
        """
        transform from body frame to inertial frame

        Parameters
        ----------
        q: list of floats
            pitch, roll, yaw
        """
        #NOTE may need to change the order of q depending on airsim order
        cos = np.cos
        sin = np.sin
        #TODO change the equation to directly reference these
        # using the extra step for now to make sure there are no mistakes
        phi = q[0] # roll
        theta =  q[1] # pitch
        psi = q[2] # yaw

        R = np.array([
            [
                cos(psi) * cos(theta),
                cos(psi) * sin(theta) * sin(phi) - sin(psi) * cos(phi),
                cos(psi) * sin(theta) * cos(phi) + sin(psi) * sin(phi)],
            [
                sin(psi) * cos(theta),
                sin(psi) * sin(theta) * sin(phi) + cos(psi) * cos(phi),
                sin(psi) * sin(theta) * cos(phi) - cos(psi) * sin(phi)],
            [
                -sin(theta),
                cos(theta) * sin(phi),
                cos(theta) * cos(phi)]
        ])

        return R

    def _T(self, w):
        """
        rotor forces

        Parameters
        ----------
        w: np.array of floats
            rotor angular velocities
        """
        return self.k * np.asarray(w)**2

    def _TB(self, w):
        """
        rotor thrust

        Parameters
        ----------
        w: np.array of floats
            rotor angular velocities
        """
        return np.array([
                0,
                0,
                np.sum(self._T(w))
            ])

    def _TauMi(self, w, dw):
        """
        rotor torques

        Parameters
        ----------
        w: np.array of floats
            rotor angular velocities
        dw: np.array of floats
            rotor angular accelerations
        """
        return self.b * np.asarray(w)**2 + self.Mi * np.asarray(dw)

    def _TauB(self, w, dw):
        """
        body torques

        returns torque along principle body axes [roll, pitch, yaw]

        Parameters
        ----------
        w: np.array of floats
            rotor angular velocities
        """

        return np.array([
                self.l * self.k * (-w[1]**2 + w[3]**2),
                self.l * self.k * (-w[0]**2 + w[2]**2),
                np.sum(self._TauMi(w, dw))
            ])

    def _A(self):
        """
        drag force matrix
        """
        A = np.array([
            [self.k, self.k, self.k, self.k],
            [0, -self.l * self.k, 0, self.l * self.k],
            [-self.l * self.k, 0, self.l * self.k, 0],
            [-self.b, self.b, -self.b, self.f]
        ])

        return A

    def _a(self, v):
        """
        cartesian linear acceleration

        Parameters
        ----------
        v: np.array of floats
            linear velocity of drone
        """
        return np.array(1/self.m * (
                    (self._TB * self._R)
                    + (self._A * v)
                    - self.g)
                )

    def _aw(self, w):
        """
        cartesian angular acceleration

        Parameters
        ----------
        w: list of floats
            angular velocity [pitch, roll, yaw] of drone
        """
        dphi = w[0] # roll
        dtheta =  w[1] # pitch
        dpsi = w[2] # yaw

        dw = (
              np.array([
                [(self.Iyy-self.Izz) * dtheta * dpsi / self.Ixx],
                [(self.Izz-self.Ixx) * dphi * dpsi / self.Iyy],
                [(self.Ixx-self.Iyy) * dphi * dtheta / self.Izz]])
              + (self._TauB/np.array([self.Ixx, self.Iyy, self.Izz]))
        )

    # Adaptive control equations
    def calc_modelled_forces(self, X):

    # Base pd equation
    def calc_speeds(self, X, target):
        """
        accepts 12D drone state and returns 4D velocity for rotors

        Parameters
        ----------
        X: np.array of 12 floats
            0th and 1st derivative of position and orientation of drone state
            np.array([x, y, z, dx, dy, dz, theta, phi, psi, dtheta, dphi, dpsi])
        target: np.array of 12 floats
            target position and velocity or drone (location and orientation)
            np.array([x, y, z, dx, dy, dz, theta, phi, psi, dtheta, dphi, dpsi])
        """

        # u == w**2
        Y = calc_modelled_forces(X)
        # TODO is self.K multiply by state or error?
        u = Y + self.TR * self.K * (target - X)
        w = np.sqrt(u)

        return w

    def calc_alt_speeds(self, X, target):
        # shorthand for equations so we maintain the same variables as Brent's paper
        x = X[0]
        y = X[1]
        z = X[2]
        dx = X[3]
        dy = X[4]
        dz = X[5]
        theta =  X[6] # pitch
        phi = X[7] # roll
        psi = X[8] # yaw
        dtheta =  X[9] # pitch
        dphi = X[10] # roll
        dpsi = X[11] # yaw

        cos = np.cos
        sin = np.sin


        # Error from PD term
        e_phi = self.kp*(target[7]-phi) + self.kd*(target[10]-dphi)
        e_theta = self.kp*(target[6]-theta) + self.kd*(target[9]-dtheta)
        e_psi = self.kp*(target[8]-psi) + self.kd*(target[11]-dpsi)

        # gamma == w**2
        gamma1 = (self.m * self.g / (4*k*cos(theta)*cos(phi))
                  - ((2*b*e_phi*self.Ixx + e_psi*self.Izz*k*l) / 4*b*k*l)
                 )
        gamma2 = (self.m * self.g / (4*k*cos(theta)*cos(phi))
                  + e_psi*self.Izz/(4*b) - e_theta*self.Iyy/(2*k*l)
                 )
        gamma3 = (self.m * self.g / (4*k*cos(theta)*cos(phi))
                  - ((-2*b*e_phi*self.Ixx + e_psi*self.Izz*k*l) / 4*b*k*l)
                 )
        gamma4 = (self.m * self.g / (4*k*cos(theta)*cos(phi))
                  + e_psi*self.Izz/(4*b) + e_theta*self.Iyy/(2*k*l)
                 )
        w = np.array([
                np.sqrt(gamma1),
                np.sqrt(gamma2),
                np.sqrt(gamma3),
                np.sqrt(gamma4)
            ])

        return w


# Default airsim params in AirSim/AirLib/include/vehicles/multirotor/firmwares/simple_flight/SimpleFlightQuadXParams.hpp
interface = airsim_interface.AirsimInterface()
interface.connect()

k1 = 0.43352026190263104
k2 = 2.0 *4
k3 = 0.5388202808181405
k4 = 1.65 * 4
k5 = 2.5995452450850185
k6 = 0.802872750102059 * 8
k7 = 0.5990281657438163
k8 = 2.8897310746350824 * 4

gains = 20 * np.array([[ 0,  0, k2,  0,  0,-k4,  0,  0,  0,  0,  0,  0],
                    [  0, k1,  0,  0,-k3,  0,-k5,  0,  0, k7,  0,  0],
                    [-k1,  0,  0, k3,  0,  0,  0,-k5,  0,  0, k7,  0],
                    [  0,  0,  0,  0,  0,  0,  0,  0,-k6,  0,  0, k8] ])

rotor_transform = np.array([
    [ 1,-1, 1, 1],
    [ 1,-1,-1,-1],
    [ 1, 1,-1, 1],
    [ 1, 1, 1,-1] ])

# NOTE: default mass is 1kg
m = 1.9

gravity_compensation = m * np.array([[9.81], [9.81], [9.81], [9.81]])
# gravity_compensation = np.array([[5.6535],
#                                         [5.6535],
#                                         [5.6535],
#                                         [5.6535],
#                                         ])

# feedback = interface.get_feedback()
# target_pos = feedback['pos'] + np.array([0, 0, 6])
target_pos = np.array([0, 0, 6])
target_lin_vel = np.array([0, 0, 0])
target_ori = np.array([0, 0, 0])
target_ang_vel = np.array([0, 0, 0])

at_target = 0
cnt = 0
data = {'pos': [], 'target': [], 'pwm': [], 'u': []}

try:
    while 1:
        cnt += 1
        feedback = interface.get_feedback()
        # convert quaternion to tait-bryan angles
        orientation = t.euler_from_quaternion(feedback['quaternion'], 'szxy')

        # Find the error
        ori_err = [target_ori[0] - orientation[0],
                        target_ori[1] - orientation[1],
                        target_ori[2] - orientation[2]]
        cz = math.cos(orientation[2])
        sz = math.sin(orientation[2])
        x_err = target_pos[0] - feedback['pos'][0]
        y_err = target_pos[1] - feedback['pos'][1]
        pos_err = [ x_err * cz + y_err * sz,
                        -x_err * sz + y_err * cz,
                         target_pos[2] - feedback['pos'][2]]

        lin_vel = [
                feedback['lin_vel'][0]*cz+feedback['lin_vel'][1]*sz,
                -feedback['lin_vel'][0]*sz+feedback['lin_vel'][1]*cz,
                feedback['lin_vel'][2]]

        ang_vel = [
                feedback['ang_vel'][0]*cz+feedback['ang_vel'][1]*sz,
                -feedback['ang_vel'][0]*sz+feedback['ang_vel'][1]*cz,
                feedback['ang_vel'][2]]

        for ii in range(3):
            if ori_err[ii] > math.pi:
                ori_err[ii] -= 2 * math.pi
            elif ori_err[ii] < -math.pi:
                ori_err[ii] += 2 * math.pi

        error = np.array([
            [pos_err[0]],
            [pos_err[1]],
            [pos_err[2]],
            [lin_vel[0]],
            [lin_vel[1]],
            [lin_vel[2]],
            [ori_err[0]],
            [ori_err[1]],
            [ori_err[2]],
            [ang_vel[0]],
            [ang_vel[1]],
            [ang_vel[2]],
        ])

        # TODO is this not w**2?
        # TODO gravity comp is in N, how do units work out?
        u = np.dot(rotor_transform, np.dot(gains, error)) + gravity_compensation

        # NOTE: in vrep motor starts at front left and goes ccw
        # in airsim we go FR, RL, FL, RR
        u = np.array([u[3], u[1], u[0], u[2]])

        # NOTE brent clipped u out to be 0-30
        max_u = 30
        pwm = np.squeeze(u/max_u)
        pwm = np.clip(pwm, 0, 1)
        if cnt < 200:
            pwm = np.array([0.6, 0.6, 0.6, 0.6])
        else:
            pwm = np.array([0.5, 0.5, 0.5, 0.5])

        interface.send_pwm(pwm, dt=0.001)

        print('u: ', u)
        print('pwm: ', pwm)
        # print('state: ', state)
        # print('error: ', error)

        # state = np.hstack((
        #     np.hstack((feedback['pos'], feedback['lin_vel'])),
        #     np.hstack((orientation, feedback['ang_vel'])))
        # )

        data['pos'].append(feedback['pos'])
        data['target'].append(target_pos)
        data['pwm'].append(pwm)
        data['u'].append(u)

        if np.sum(error) < 0.2:
            at_target += 1
        else:
            at_target = 0

        if at_target > 100:
            break
except:
    plt.figure()
    plt.subplot(311)
    plt.title('pos')
    plt.plot(np.array(data['pos'])[:, :3], label='pos')
    plt.plot(np.array(data['target'])[:, :3], linestyle='--', label='target')
    plt.legend()
    plt.subplot(312)
    plt.title('pwm')
    plt.plot(data['pwm'])
    plt.legend()
    plt.subplot(313)
    plt.title('u')
    plt.plot(np.squeeze(data['u']))
    plt.legend()
    plt.show()

    interface.disconnect()

else:
    plt.figure()
    plt.subplot(311)
    plt.title('pos')
    plt.plot(np.array(data['pos'])[:, :3], label='pos')
    plt.plot(np.array(data['target'])[:, :3], linestyle='--', label='target')
    plt.legend()
    plt.subplot(312)
    plt.title('pwm')
    plt.plot(data['pwm'])
    plt.legend()
    plt.subplot(313)
    plt.title('u')
    plt.plot(np.squeeze(data['u']))
    plt.legend()
    plt.show()


    interface.disconnect()
