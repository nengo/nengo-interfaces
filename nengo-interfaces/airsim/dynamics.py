import numpy as np
import traceback
import math
import airsim_interface
import matplotlib.pyplot as plt
from abr_control.utils import transformations as t

plot = True
use_pd = True
use_adapt = True
# shorthand
cos = np.cos
sin = np.sin
interface = airsim_interface.AirsimInterface()
interface.connect()


# PD gains
k1 = 0.43352026190263104
k2 = 2.0 *4
k3 = 0.5388202808181405
k4 = 1.65 * 4
k5 = 2.5995452450850185
k6 = 0.802872750102059 * 8
k7 = 0.5990281657438163
k8 = 2.8897310746350824 * 4

gains = np.array([[ 0,  0, k2,  0,  0,-k4,  0,  0,  0,  0,  0,  0],
                    [  0, k1,  0,  0,-k3,  0,-k5,  0,  0, k7,  0,  0],
                    [-k1,  0,  0, k3,  0,  0,  0,-k5,  0,  0, k7,  0],
                    [  0,  0,  0,  0,  0,  0,  0,  0,-k6,  0,  0, k8] ])

rotor_transform = np.array([
    [ 1,-1, 1, 1],
    [ 1,-1,-1,-1],
    [ 1, 1,-1, 1],
    [ 1, 1, 1,-1] ])

# mass from airsim code
m = 1 - 0.055*4
# gravity force
g = m * np.array([[0, 0, 9.81, 0, 0, 0]])

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
rpss = revolutions_per_second**2
max_w = revolutions_per_second * 2 * np.pi  # radians / sec
# NOTE should be using rad/sec, but airsim max thrust and torque calculations use rev/sec (check RotorParams.hpp)
# max_w = revolutions_per_second

# from api docs, Thrust = k * w**2
# at max_w we have max thrust
max_thrust = 4.179446268
# assuming in foot-pound-second imperial units
# convert to N metric
# max_thrust = 0.5776
# max_thrust *= 4.448221
# k = max_thrust / (max_w**2)
k = max_thrust / rpss
print('k: ', k)

# same for max torque
max_torque = 0.055562
# assuming in foot-pounds imperial units
# convert to Nm metric
# max_torque *= 1.3558
# b = max_torque / (max_w**2)
b = max_torque / rpss
print('b: ', b)

A = np.array([
    [k, k, k, k],
    [0, -l * k, 0, l * k],
    [-l * k, 0, l * k, 0],
    [-b, b, -b, b]
])

A_inv = np.linalg.inv(A)

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
        # ======================= FEEDBACK
        feedback = interface.get_feedback()
        # convert quaternion to tait-bryan angles
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

        if use_pd:
            # ======================== PD CONTROLLER
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
            pd = np.dot(rotor_transform, np.dot(gains, error))
        else:
            pd = np.zeros(4)

        if use_adapt:
            # ===================== AIRSIM DYNAMICS
            # shorthandto avoid confusion and mistakes in equations
            phi=orientation[0]
            theta=orientation[1]
            psi=orientation[2]

            B = np.array([
                [cos(phi) * sin(theta) * cos(psi) + sin(theta) * sin(psi), 0, 0, 0],
                [cos(phi) * sin(theta) * sin(psi) - sin(theta) * cos(psi), 0, 0, 0],
                [cos(phi) * cos(theta), 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

            B_inv = np.linalg.pinv(B)

            # gravity_w = np.dot(A_inv, np.dot(calc_B_inv(phi, theta, psi), g.T))
            gravity_w = np.dot(np.dot(A_inv, B_inv), g.T)
            # gravity_w = np.sqrt(w)
        else:
            gravity_w = np.zeros(4)

        # ==================== COMBINING CONTROLLER
        # transforms rotors from vrep to airsim ordering
        # TODO is this not w**2?
        # TODO gravity comp is in N, how do units work out?
        u = pd + gravity_w
        # u = gravity_w

        # convert from w**2 to w
        u = np.sqrt(u)

        # NOTE: in vrep motor starts at front left and goes ccw
        # in airsim we go FR, RL, FL, RR
        u = np.array([u[3], u[1], u[0], u[2]])

        # convert to pwm 0-1
        pwm = np.squeeze(u/max_w)
        interface.send_pwm(pwm, dt=0.001)

        data['pos'].append(feedback['pos'])
        data['target'].append(target_pos)
        data['pwm'].append(pwm)
        data['u'].append(u)

        pos_norm_error = np.linalg.norm(target_pos - feedback['pos'])
        if cnt % 100 == 0:
            print('pos error: ', pos_norm_error)
            print('pos: ', feedback['pos'])
            print('target: ', target_pos)
            print('u: ', u)
            print('pwm: ', pwm)

        if np.sum(pos_norm_error) < 0.2:
            print(pos_norm_error)
            at_target += 1
        else:
            at_target = 0

        if at_target > 100:
            print(pos_norm_error)
            break

except Exception as e:
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
    interface.disconnect()
    print(traceback.format_exc())

else:
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
    interface.disconnect()
    print('disconnected')
