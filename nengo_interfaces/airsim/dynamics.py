import numpy as np
import traceback
import math
import airsim_interface
import matplotlib.pyplot as plt
from abr_control.utils import transformations as t

plot = False
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


# Dynamics
# lift coeff (airlib/include/vehicles/multirotor/rotorparams.hpp)
# k = 0.109919
# drag coeff (airlib/include/vehicles/multirotor/multirotorparams.hpp)
# b = 1.3/4
# drag coeff (airlib/include/vehicles/multirotor/rotorparams.hpp)
# b = 0.040164
# arm lengths (airlib/include/vehicles/multirotor/firmwares/simple_flight/simpleflightquadxparams.hpp)
l = 0.2275
# from airsim params
max_rpm = 6396.667;
revolutions_per_second = max_rpm / 60
# max_w = revolutions_per_second
max_w = revolutions_per_second * 2 * np.pi  # radians / sec

# from api docs, Thrust = k * w**2
# at max_w we have max thrust
max_thrust = 4.179446268
k = max_thrust / (max_w**2)
# same for max torque
max_torque = 0.055562
b = max_torque / (max_w**2)
print('k: ', k)
print('b: ', b)

A = np.array([
    [k, k, k, k],
    [0, -l * k, 0, l * k],
    [-l * k, 0, l * k, 0],
    [-b, b, -b, b]
])

A_inv = np.linalg.inv(A)

def calc_B(phi, theta, psi):
    """
    phi == roll
    theta == pitch
    psi == yaw
    """
    cos = np.cos
    sin = np.sin
    B = np.array([
        [cos(phi) * sin(theta) * cos(psi) + sin(theta) * sin(psi), 0, 0, 0],
        [cos(phi) * sin(theta) * sin(psi) - sin(theta) * cos(psi), 0, 0, 0],
        [cos(phi) * cos(theta), 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    # print('calc_B: ', B)
    # print('shape: ', B.shape)

    return B

def calc_B_inv(phi, theta, psi):
    B_inv = np.linalg.pinv(calc_B(phi, theta, psi))
    # print('calc_B_inv: ', B_inv)
    return B_inv

#TODO what should g be? Is it 3d in state space?
# g = np.array([[9.81], [9.81], [9.81], [9.81]])
m = 1 - 0.055*4
g = m * np.array([[0, 0, 9.81, 0, 0, 0]])
# g = np.array([[9.81, 9.81, 9.81, 9.81, 9.81, 9.81]])

def calc_w(phi, theta, psi):
    # w = np.dot(A_inv, np.dot(calc_B_inv(phi, theta, psi), g.T))
    w = np.dot(np.dot(A_inv, calc_B_inv(phi, theta, psi)), g.T)
    w = np.sqrt(w)
    # print('calc_w: ', w)
    return w

# m = 1.9
# gravity_compensation = m * np.array([[9.81], [9.81], [9.81], [9.81]])

# feedback = interface.get_feedback()
# target_pos = feedback['pos'] + np.array([0, 0, 6])
target_pos = np.array([0, 0, 6])
target_lin_vel = np.array([0, 0, 0])
target_ori = np.array([0, 0, 0])
target_ang_vel = np.array([0, 0, 0])

at_target = 0
cnt = 0
data = {'pos': [], 'target': [], 'pwm': [], 'u': []}

# Brent's function
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


try:
    while 1:
        cnt += 1
        # ======================== PD CONTROLLER
        # feedback = interface.get_feedback()
        # convert quaternion to tait-bryan angles
        # orientation = t.euler_from_quaternion(feedback['quaternion'], 'rzxy')
        # # Find the error
        # ori_err = [target_ori[0] - orientation[0],
        #                 target_ori[1] - orientation[1],
        #                 target_ori[2] - orientation[2]]
        # cz = math.cos(orientation[2])
        # sz = math.sin(orientation[2])
        # x_err = target_pos[0] - feedback['pos'][0]
        # y_err = target_pos[1] - feedback['pos'][1]
        # pos_err = [ x_err * cz + y_err * sz,
        #                 -x_err * sz + y_err * cz,
        #                  target_pos[2] - feedback['pos'][2]]
        #
        # lin_vel = [
        #         feedback['lin_vel'][0]*cz+feedback['lin_vel'][1]*sz,
        #         -feedback['lin_vel'][0]*sz+feedback['lin_vel'][1]*cz,
        #         feedback['lin_vel'][2]]
        #
        # ang_vel = [
        #         feedback['ang_vel'][0]*cz+feedback['ang_vel'][1]*sz,
        #         -feedback['ang_vel'][0]*sz+feedback['ang_vel'][1]*cz,
        #         feedback['ang_vel'][2]]
        #
        # for ii in range(3):
        #     if ori_err[ii] > math.pi:
        #         ori_err[ii] -= 2 * math.pi
        #     elif ori_err[ii] < -math.pi:
        #         ori_err[ii] += 2 * math.pi
        #
        # error = np.array([
        #     [pos_err[0]],
        #     [pos_err[1]],
        #     [pos_err[2]],
        #     [lin_vel[0]],
        #     [lin_vel[1]],
        #     [lin_vel[2]],
        #     [ori_err[0]],
        #     [ori_err[1]],
        #     [ori_err[2]],
        #     [ang_vel[0]],
        #     [ang_vel[1]],
        #     [ang_vel[2]],
        # ])
        #
        # # TODO is this not w**2?
        # # TODO gravity comp is in N, how do units work out?
        # u = np.dot(rotor_transform, np.dot(gains, error)) + gravity_compensation
        #
        # # NOTE: in vrep motor starts at front left and goes ccw
        # # in airsim we go FR, RL, FL, RR
        # u = np.array([u[3], u[1], u[0], u[2]])
        #
        # # NOTE brent clipped u out to be 0-30
        # max_u = 30
        # pwm = np.squeeze(u/max_u)
        # pwm = np.clip(pwm, 0, 1)
        # ======================================

        # ===================== AIRSIM DYNAMICS
        feedback = interface.get_feedback()
        # convert quaternion to tait-bryan angles
        # orientation = t.euler_from_quaternion(feedback['quaternion'], 'rzxy')
        orientation = t.euler_from_quaternion(feedback['quaternion'], 'rzyx')
        # brent's conversion returns the angles in reverse order for some reason
        # flipping here to avoid having to change the transforms
        orientation = np.array([orientation[2], orientation[1], orientation[0]])
        """
        phi == roll
        theta == pitch
        psi == yaw

        converting angles to z-x-y (yaw, roll, pitch)
        in brents code the conversion returns roll,pitch,yaw, in that order
        """
        if cnt > 20:
            # w = calc_w(phi=orientation[1], theta=orientation[2], psi=orientation[0])
            w = calc_w(phi=orientation[0], theta=orientation[1], psi=orientation[2])
            # print('w: ', w)
            u = np.array([w[3], w[1], w[0], w[2]])
            # print('u: ', u)
            pwm = np.squeeze(u/max_w)
        # elif cnt > 30 and cnt < 50:
        #     u = [0, 0, 0, 0]
        #     pwm = u
        else:
            u = [1, 1, 1, 1]
            pwm = [1, 1, 1, 1]

        interface.send_pwm(pwm, dt=0.001)

        print('u: ', u)
        print('pwm: ', pwm)
        # print('state: ', state)
        # print('error: ', error)

        # data['pos'].append(feedback['pos'])
        # data['target'].append(target_pos)
        # data['pwm'].append(pwm)
        # data['u'].append(u)

        # if np.sum(error) < 0.2:
        #     at_target += 1
        # else:
        #     at_target = 0
        #
        # if at_target > 100:
        #     break
except Exception as e:
    interface.disconnect()
    print(traceback.format_exc())

else:
    interface.disconnect()
    print('disconnected')

    if plot:
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


# else:
#     if plot:
#         plt.figure()
#         plt.subplot(311)
#         plt.title('pos')
#         plt.plot(np.array(data['pos'])[:, :3], label='pos')
#         plt.plot(np.array(data['target'])[:, :3], linestyle='--', label='target')
#         plt.legend()
#         plt.subplot(312)
#         plt.title('pwm')
#         plt.plot(data['pwm'])
#         plt.legend()
#         plt.subplot(313)
#         plt.title('u')
#         plt.plot(np.squeeze(data['u']))
#         plt.legend()
#         plt.show()
#
#
#     interface.disconnect()
