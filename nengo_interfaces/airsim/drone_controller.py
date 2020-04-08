import numpy as np
import math
import airsim_interface
import matplotlib.pyplot as plt
from abr_control.utils import transformations as t

# class DroneController():
#     def __init__(self):
#         #TODO should move these constants to a drone config
#
#         #TODO what are k and b?
#         # drag force constants
#         self.A = np.array([
#             [k, k, k, k, ],
#             [0, -l*k, 0, l*k],
#             [-l*k, 0, l*k, 0],
#             [-b, b, -b, b]
#         ])
#
#         # lift constant
#         self.K = np.array([
#             [0, 0, k2, 0, 0, -k4, 0, 0, 0, 0, 0, 0],
#             [0, k1, 0, 0, -k3, 0, -k5, 0, 0, k7, 0, 0],
#             [-k1, 0, 0, k3, 0, 0, 0, -k5, 0, 0, k7, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, -k6, 0, 0, k8]
#         ])
#
#         # transform to rotor vel space
#         self.TR = np.array([
#             [1, -1, 1, 1],
#             [1, -1, -1, -1],
#             [1, 1, -1, 1],
#             [1, 1, 1, -1]
#         ])
#
#         # drag constant
#         self.b = [1, 1, 1, 1]
#         # distance from rotor to COM
#         self.l = 1
#         # drone inertia matrix
#         self.Ixx = None
#         self.Iyy = None
#         self.Izz = None
#         self.I = np.array([
#             [self.Ixx, 0, 0],
#             [0, self.Iyy, 0],
#             [0, 0, self.Izz]
#         ])
#         # rotor moment of inertia
#         self.Mi = None
#         # mass
#         self.m = 1
#
#         # gravity term
#         self.g = np.array([0, 0, -9.81])
#
#
#     # TODO are any of the dynamics equations even used?
#     # dynamics equations
#     def _R(self, q):
#         """
#         transform from body frame to inertial frame
#
#         Parameters
#         ----------
#         q: list of floats
#             pitch, roll, yaw
#         """
#         #NOTE may need to change the order of q depending on airsim order
#         cos = np.cos
#         sin = np.sin
#         #TODO change the equation to directly reference these
#         # using the extra step for now to make sure there are no mistakes
#         phi = q[0] # roll
#         theta =  q[1] # pitch
#         psi = q[2] # yaw
#
#         R = np.array([
#             [
#                 cos(psi) * cos(theta),
#                 cos(psi) * sin(theta) * sin(phi) - sin(psi) * cos(phi),
#                 cos(psi) * sin(theta) * cos(phi) + sin(psi) * sin(phi)],
#             [
#                 sin(psi) * cos(theta),
#                 sin(psi) * sin(theta) * sin(phi) + cos(psi) * cos(phi),
#                 sin(psi) * sin(theta) * cos(phi) - cos(psi) * sin(phi)],
#             [
#                 -sin(theta),
#                 cos(theta) * sin(phi),
#                 cos(theta) * cos(phi)]
#         ])
#
#         return R
#
#     def _T(self, w):
#         """
#         rotor forces
#
#         Parameters
#         ----------
#         w: np.array of floats
#             rotor angular velocities
#         """
#         return self.k * np.asarray(w)**2
#
#     def _TB(self, w):
#         """
#         rotor thrust
#
#         Parameters
#         ----------
#         w: np.array of floats
#             rotor angular velocities
#         """
#         return np.array([
#                 0,
#                 0,
#                 np.sum(self._T(w))
#             ])
#
#     def _TauMi(self, w, dw):
#         """
#         rotor torques
#
#         Parameters
#         ----------
#         w: np.array of floats
#             rotor angular velocities
#         dw: np.array of floats
#             rotor angular accelerations
#         """
#         return self.b * np.asarray(w)**2 + self.Mi * np.asarray(dw)
#
#     def _TauB(self, w, dw):
#         """
#         body torques
#
#         returns torque along principle body axes [roll, pitch, yaw]
#
#         Parameters
#         ----------
#         w: np.array of floats
#             rotor angular velocities
#         """
#
#         return np.array([
#                 self.l * self.k * (-w[1]**2 + w[3]**2),
#                 self.l * self.k * (-w[0]**2 + w[2]**2),
#                 np.sum(self._TauMi(w, dw))
#             ])
#
#     def _A(self):
#         """
#         drag force matrix
#         """
#         #TODO find out what this should be
#         raise NotImplementedError
#
#     def _a(self, v):
#         """
#         cartesian linear acceleration
#
#         Parameters
#         ----------
#         v: np.array of floats
#             linear velocity of drone
#         """
#         return np.array(1/self.m * (
#                     (self._TB * self._R)
#                     + (self._A * v)
#                     - self.g)
#                 )
#
#     def _aw(self, w):
#         """
#         cartesian angular acceleration
#
#         Parameters
#         ----------
#         w: list of floats
#             angular velocity [pitch, roll, yaw] of drone
#         """
#         dphi = w[0] # roll
#         dtheta =  w[1] # pitch
#         dpsi = w[2] # yaw
#
#         dw = (
#               np.array([
#                 [(self.Iyy-self.Izz) * dtheta * dpsi / self.Ixx],
#                 [(self.Izz-self.Ixx) * dphi * dpsi / self.Iyy],
#                 [(self.Ixx-self.Iyy) * dphi * dtheta / self.Izz]])
#               + (self._TauB/np.array([self.Ixx, self.Iyy, self.Izz]))
#         )
#
#     # Adaptive control equations
#     def calc_modelled_forces(self, X):
#         """
#         Parameters
#         ----------
#         X: np.array of 12 floats
#             0th and 1st derivative of position and orientation of drone state
#             np.array([x, y, z, dx, dy, dz, theta, phi, psi, dtheta, dphi, dpsi])
#         """
#         # shorthand for equations so we maintain the same variables as Brent's paper
#         x = X[0]
#         y = X[1]
#         z = X[2]
#         dx = X[3]
#         dy = X[4]
#         dz = X[5]
#         theta =  X[6] # pitch
#         phi = X[7] # roll
#         psi = X[8] # yaw
#         dtheta =  X[9] # pitch
#         dphi = X[10] # roll
#         dpsi = X[11] # yaw
#
#         cos = np.cos
#         sin = np.sin
#
#         d = (
#                 (cos(phi) * sin(theta) * sin(psi) + sin(theta) * sin(psi))**2
#                 + (cos(phi) * sin(theta) * sin(psi) - sin(theta) * cos(psi))**2
#                 + (cos(phi) * cos(theta))**2
#             )
#
#         a = (cos(phi) * sin(theta) * cos(psi) + sin(theta) * sin(psi)) / d
#
#         b = (cos(phi) * sin(theta) * sin(psi) - sin(theta) * cos(psi)) / d
#
#         c = cos(phi) * cos(theta) / d
#
#         #NOTE Y in eqn 4.9
#         #NOTE first 3 columns are constant, can precalculate one row
#         # for optimization
#         state_dynamics = np.array([
#             [a*abs(dx)*dx, b*abs(dy)*dy, c*abs(dz)*dz, c,
#                 0, -dphi*dpsi, -dphi*dtheta],
#             [a*abs(dx)*dx, b*abs(dy)*dy, c*abs(dz)*dz, c,
#                 -dtheta*dpsi, 0, dphi*dtheta],
#             [a*abs(dx)*dx, b*abs(dy)*dy, c*abs(dz)*dz, c,
#                 0, dphi*dpsi, -dphi*dtheta]
#             [a*abs(dx)*dx, b*abs(dy)*dy, c*abs(dz)*dz, c,
#                 dtheta*dpsi, 0 , dphi*dtheta]
#         ])
#
#         #NOTE Theta in eqn 4.11
#         #TODO are Ax,y,z the first three rows of self.A defined above?
#         #TODO what is k?
#         drone_dynamics = np_array([
#             [Ax/4k],
#             [Ay/4k],
#             [Az/4k],
#             [mg/4k],
#             [(self.Izz-self.Iyy)/(2*k*l)],
#             [(self.Ixx-self.Izz)/(2*k*l)],
#             [(self.Iyy-self.Ixx)/(2*k*l)]
#         ])
#
#         return state_dynamics * drone_dynamics
#
#     # Base pd equation
#     def calc_speeds(self, X, target):
#         """
#         accepts 12D drone state and returns 4D velocity for rotors
#
#         Parameters
#         ----------
#         X: np.array of 12 floats
#             0th and 1st derivative of position and orientation of drone state
#             np.array([x, y, z, dx, dy, dz, theta, phi, psi, dtheta, dphi, dpsi])
#         target: np.array of 12 floats
#             target position and velocity or drone (location and orientation)
#             np.array([x, y, z, dx, dy, dz, theta, phi, psi, dtheta, dphi, dpsi])
#         """
#
#         # u == w**2
#         Y = calc_modelled_forces(X)
#         # TODO is self.K multiply by state or error?
#         u = Y + self.TR * self.K * (target - X)
#         w = np.sqrt(u)
#
#         return w
#
#     def calc_alt_speeds(self, X, target):
#         # shorthand for equations so we maintain the same variables as Brent's paper
#         x = X[0]
#         y = X[1]
#         z = X[2]
#         dx = X[3]
#         dy = X[4]
#         dz = X[5]
#         theta =  X[6] # pitch
#         phi = X[7] # roll
#         psi = X[8] # yaw
#         dtheta =  X[9] # pitch
#         dphi = X[10] # roll
#         dpsi = X[11] # yaw
#
#         cos = np.cos
#         sin = np.sin
#
#
#         # Error from PD term
#         e_phi = self.kp*(target[7]-phi) + self.kd*(target[10]-dphi)
#         e_theta = self.kp*(target[6]-theta) + self.kd*(target[9]-dtheta)
#         e_psi = self.kp*(target[8]-psi) + self.kd*(target[11]-dpsi)
#
#         # gamma == w**2
#         gamma1 = (self.m * self.g / (4*k*cos(theta)*cos(phi))
#                   - ((2*b*e_phi*self.Ixx + e_psi*self.Izz*k*l) / 4*b*k*l)
#                  )
#         gamma2 = (self.m * self.g / (4*k*cos(theta)*cos(phi))
#                   + e_psi*self.Izz/(4*b) - e_theta*self.Iyy/(2*k*l)
#                  )
#         gamma3 = (self.m * self.g / (4*k*cos(theta)*cos(phi))
#                   - ((-2*b*e_phi*self.Ixx + e_psi*self.Izz*k*l) / 4*b*k*l)
#                  )
#         gamma4 = (self.m * self.g / (4*k*cos(theta)*cos(phi))
#                   + e_psi*self.Izz/(4*b) + e_theta*self.Iyy/(2*k*l)
#                  )
#         w = np.array([
#                 np.sqrt(gamma1),
#                 np.sqrt(gamma2),
#                 np.sqrt(gamma3),
#                 np.sqrt(gamma4)
#             ])
#
#         return w
#

# Default airsim params in AirSim/AirLib/include/vehicles/multirotor/firmwares/simple_flight/SimpleFlightQuadXParams.hpp
interface = airsim_interface.AirsimInterface()
interface.connect()

k1 = 0.43352026190263104
k2 = 2.0 *2
k3 = 0.5388202808181405
k4 = 1.65 * 2
k5 = 2.5995452450850185
k6 = 0.802872750102059 * 4
k7 = 0.5990281657438163
k8 = 2.8897310746350824 * 2

gains = np.array([[ 0,  0, k2,  0,  0,-k4,  0,  0,  0,  0,  0,  0],
                    [  0, k1,  0,  0,-k3,  0,-k5,  0,  0, k7,  0,  0],
                    [-k1,  0,  0, k3,  0,  0,  0,-k5,  0,  0, k7,  0],
                    [  0,  0,  0,  0,  0,  0,  0,  0,-k6,  0,  0, k8] ])

rotor_transform = np.array([[ 1,-1, 1, 1],
                                    [ 1,-1,-1,-1],
                                    [ 1, 1,-1, 1],
                                    [ 1, 1, 1,-1] ])

# NOTE: default mass is 1kg
gravity_compensation = np.array([[9.81], [9.81], [9.81], [9.81]])
# gravity_compensation = np.array([[5.6535],
#                                         [5.6535],
#                                         [5.6535],
#                                         [5.6535],
#                                         ])



state, orientation = interface.get_feedback()
orientation = t.euler_from_quaternion(orientation, 'rzyx')
state[6] = orientation[0]
state[7] = orientation[1]
state[8] = orientation[2]
# state = np.array([state[0], state[1], state[2], 0, 0, 0, 0, 0, 0, 0, 0, 0])
target = np.array([0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0])

def calc_brent_error(target, state):
    # Find the error
    # ori_err = [t_ori[0] - ori[0],
    #                 t_ori[1] - ori[1],
    #                 t_ori[2] - ori[2]]
    ori_err = [target[6] - state[6],
               target[7] - state[7],
               target[8] - state[8]
               ]
    # cz = math.cos(ori[2])
    cz = math.cos(state[4])
    # sz = math.sin(ori[2])
    sz = math.sin(state[4])
    # x_err = t_pos[0] - pos[0]
    x_err = target[0] - state[0]
    # y_err = t_pos[1] - pos[1]
    y_err = target[1] - state[1]
    # pos_err = [ x_err * cz + y_err * sz,
    #                 -x_err * sz + y_err * cz,
    #                     t_pos[2] - pos[2]]

    # NOTE: ignoring velocity errors for initial implementation
    error = np.array([[x_err * cz + y_err * sz,
             -x_err * sz + y_err * cz,
             target[2] - state[2],
             0, 0, 0,
             ori_err[0],
             ori_err[1],
             ori_err[2],
             0,0,0]]).T
    return error

def calc_norm_error(target, state):
    return np.linalg.norm(target**2-state**2)

# NOTE brent clipped u out to be 0-30
def convert_to_pwm(u, max_u):
    pwm = np.squeeze(u/max_u)
    pwm = np.clip(pwm, 0, 1)
    return pwm

# from brent's repo
def b( num ):
    """ forces magnitude to be 1 or less """
    if abs( num ) > 1.0:
        return math.copysign( 1.0, num )
    else:
        return num


def convert_angles( ang ):
    """ Converts Euler angles from x-y-z to z-x-y convention """

    s1 = math.sin(ang[0])
    s2 = math.sin(ang[1])
    s3 = math.sin(ang[2])
    c1 = math.cos(ang[0])
    c2 = math.cos(ang[1])
    c3 = math.cos(ang[2])

    pitch = math.asin( b(c1*c3*s2-s1*s3) )
    cp = math.cos(pitch)
    # just in case
    if cp == 0:
        cp = 0.000001

    yaw = math.asin( b((c1*s3+c3*s1*s2)/cp) ) #flipped
    # Fix for getting the quadrants right
    if c3 < 0 and yaw > 0:
        yaw = math.pi - yaw
    elif c3 < 0 and yaw < 0:
        yaw = -math.pi - yaw

    roll = math.asin( b((c3*s1+c1*s2*s3)/cp) ) #flipped
    return [roll, pitch, yaw]


def plot(data):
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

error = calc_brent_error(target, state)

at_target = 0
data = {'pos': [], 'target': [], 'pwm': [], 'u': []}
try:
    while 1:
        print('---------')
        error = calc_brent_error(target=target, state=state)
        # TODO is this not w**2?
        # TODO gravity comp is in N, how do units work out?
        u = np.dot(rotor_transform, np.dot(gains, error)) + gravity_compensation
        # print(u.shape)
        # print(rotor_transform.shape)
        # print(gains.shape)
        # print(error.shape)

        # u = np.array([u[2], u[0], u[1], u[3]])
        pwm = convert_to_pwm(u=u, max_u=30)
        # pwm = [1, 1, 1, 1]
        interface.send_pwm(pwm, dt=0.001)

        # hacked in to pass quaternion separately
        state, orientation = interface.get_feedback()
        print('pre: ', state)
        # convert quaternion to tait-bryan angles
        print(orientation)
        orientation = t.euler_from_quaternion(orientation, 'szyx')
        print(orientation)
        # replace roll pitch yaw with this, which I think should be the same?
        state[6] = orientation[0]
        state[7] = orientation[1]
        state[8] = orientation[2]
        print('post: ', state)

        # print('u: ', u)
        # print('pwm: ', pwm)
        # print('state: ', state)
        # print('error: ', error)
        data['pos'].append(state)
        data['target'].append(target)
        data['pwm'].append(pwm)
        data['u'].append(u)
        # if cnt > 10:
        #     break
        if np.sum(error) < 0.2:
            at_target += 1
        else:
            at_target = 0

        if at_target > 100:
            break
except:
    plot(data)
    interface.disconnect()



# print('AT TARGET!')
# interface.disconnect()
# plot(data)
