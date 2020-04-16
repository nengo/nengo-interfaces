import numpy as np
import time
# import nni
import ctypes
import traceback
import vrep
import math
import airsim_interface
import matplotlib.pyplot as plt
from abr_control.utils import transformations as t

# NOTE not used atm
def gen_u_adapt(params, orientation):
    # shorthand
    cos = np.cos
    sin = np.sin

    # Dynamics
    l = params['l']
    k = params['k']
    b = params['b']
    m = params['m']
    # gravity force
    g = m * np.array([[0, 0, 9.81, 0, 0, 0]])

    A = np.array([
        [k, k, k, k],
        [0, -l * k, 0, l * k],
        [-l * k, 0, l * k, 0],
        [-b, b, -b, b]
    ])

    A_inv = np.linalg.inv(A)

    # shorthand to avoid confusion and mistakes in equations
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

    gravity_w = np.sqrt(np.dot(np.dot(A_inv, B_inv), g.T))

    return gravity_w

class vrepInterface():
    def connect(self, cid=None):
        self.vrep_mode = vrep.simx_opmode_oneshot
        SYNC = True
        # setup vrep connection
        if cid is None:
            vrep.simxFinish(-1) # just in case, close all opened connections
            self.cid = vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
        else:
            self.cid = cid

        if self.cid != -1:
            print ('Connected to V-REP remote API server, client id: %s' % self.cid)
            vrep.simxStartSimulation( self.cid, vrep.simx_opmode_oneshot )
            if SYNC:
                vrep.simxSynchronous( self.cid, True )
        else:
            print ('Failed connecting to V-REP remote API server')
            self.exit()


        err, self.copter = vrep.simxGetObjectHandle(self.cid, "Quadricopter_base",
                                                vrep.simx_opmode_oneshot_wait )
        err, self.target = vrep.simxGetObjectHandle(self.cid, "Quadricopter_target",
                                                vrep.simx_opmode_oneshot_wait )

        # Reset the motor commands to zero
        packedData=vrep.simxPackFloats([0,0,0,0])
        raw_bytes = (ctypes.c_ubyte * len(packedData)).from_buffer_copy(packedData)

        err = vrep.simxSetStringSignal(self.cid, "rotorTargetVelocities",
                                        raw_bytes,
                                        self.vrep_mode)

    def disconnect(self):
        err = vrep.simxStopSimulation(self.cid, vrep.simx_opmode_oneshot_wait )
        time.sleep(0.01) # Maybe this will prevent V-REP from crashing as often

    def get_feedback(self):
        # Return the state variables
        err, ori = vrep.simxGetObjectOrientation(self.cid, self.copter, -1,
                                                self.vrep_mode )
        err, pos = vrep.simxGetObjectPosition(self.cid, self.copter, -1,
                                            self.vrep_mode )
        err, lin, ang = vrep.simxGetObjectVelocity(self.cid, self.copter,
                                                            self.vrep_mode )

        ori = self.convert_angles(ori)

        feedback = {'pos': pos, 'ori': ori, 'lin_vel': lin, 'ang_vel': ang}

        return feedback

    def b(self, num):
        """ forces magnitude to be 1 or less """
        if abs( num ) > 1.0:
            return math.copysign( 1.0, num )
        else:
            return num


    def convert_angles(self, ang):
        """ Converts Euler angles from x-y-z to z-x-y convention """
        s1 = math.sin(ang[0])
        s2 = math.sin(ang[1])
        s3 = math.sin(ang[2])
        c1 = math.cos(ang[0])
        c2 = math.cos(ang[1])
        c3 = math.cos(ang[2])

        pitch = math.asin( self.b(c1*c3*s2-s1*s3) )
        cp = math.cos(pitch)
        # just in case
        if cp == 0:
            cp = 0.000001

        yaw = math.asin( self.b((c1*s3+c3*s1*s2)/cp) ) #flipped
        # Fix for getting the quadrants right
        if c3 < 0 and yaw > 0:
            yaw = math.pi - yaw
        elif c3 < 0 and yaw < 0:
            yaw = -math.pi - yaw

        roll = math.asin( self.b((c3*s1+c1*s2*s3)/cp) ) #flipped
        return [roll, pitch, yaw]

    def send_motor_commands(self, values):

        # Limit motors by max and min values
        motor_values = np.zeros(4)
        for i in range(4):
          motor_values[i] = values[i]
        # print(motor_values)
        packedData=vrep.simxPackFloats(motor_values.flatten())
        raw_bytes = (ctypes.c_ubyte * len(packedData)).from_buffer_copy(packedData) 
        err = vrep.simxSetStringSignal(self.cid, "rotorTargetVelocities",
                                        raw_bytes,
                                        self.vrep_mode)
        vrep.simxSynchronousTrigger( self.cid )

def fly_session():
    """
    phi == roll
    theta == pitch
    psi == yaw
    """
    plot = False
    step_limit = 10000
    pos_track = []
    u_track = []

    interface = vrepInterface()
    interface.connect()

    # params = nni.get_next_parameter()
    # params = {
    #             'k1': 0.43352026190263104,
    #             'k2' : 2.0 *2,
    #             'k3' : 0.5388202808181405,
    #             'k4' : 1.65 * 2,
    #             'k5' : 2.5995452450850185,
    #             'k6' : 0.802872750102059 * 4,
    #             'k7' : 0.5990281657438163,
    #             'k8' : 2.8897310746350824 * 2
    #         }

    targets = [
        {'target_pos': np.array([0, 0, 1]),
         'target_lin_vel': np.array([0, 0, 0]),
         'target_ori': np.array([0, 0, 0]),
         'target_ang_vel': np.array([0, 0, 0])
        },
    ]


    # PD gains
    k1 = params['k1']
    k2 = params['k2']
    k3 = params['k3']
    k4 = params['k4']
    k5 = params['k5']
    k6 = params['k6']
    k7 = params['k7']
    k8 = params['k8']

    gains = np.array([[ 0,  0, k2,  0,  0,-k4,  0,  0,  0,  0,  0,  0],
                        [  0, k1,  0,  0,-k3,  0,-k5,  0,  0, k7,  0,  0],
                        [-k1,  0,  0, k3,  0,  0,  0,-k5,  0,  0, k7,  0],
                        [  0,  0,  0,  0,  0,  0,  0,  0,-k6,  0,  0, k8] ])

    rotor_transform = np.array([
        [ 1,-1, 1, 1],
        [ 1,-1,-1,-1],
        [ 1, 1,-1, 1],
        [ 1, 1, 1,-1] ])

    for target in targets:
        cnt = 0
        while cnt < step_limit:

            # brent only sent commands every 10 steps
            if cnt % 10 == 0:
                # get feedback
                feedback = interface.get_feedback()

                # Find the error
                ori_err = [target['target_ori'][0] - feedback['ori'][0],
                                target['target_ori'][1] - feedback['ori'][1],
                                target['target_ori'][2] - feedback['ori'][2]]
                cz = math.cos(feedback['ori'][2])
                sz = math.sin(feedback['ori'][2])
                x_err = target['target_pos'][0] - feedback['pos'][0]
                y_err = target['target_pos'][1] - feedback['pos'][1]
                pos_err = [ x_err * cz + y_err * sz,
                                -x_err * sz + y_err * cz,
                                target['target_pos'][2] - feedback['pos'][2]]

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

                # calculate pd signal
                # pd = np.sqrt(np.dot(rotor_transform, np.dot(gains, error)))

                # gravity_compensation = np.matrix([[5.6535],
                #                                     [5.6535],
                #                                     [5.6535],
                #                                     [5.6535],
                #                                     ])
                # this should be sqrt but will lead to errors if force is negative
                pd = np.dot(rotor_transform, np.dot(gains, error)) #+ gravity_compensation

                interface.send_motor_commands(pd)

                if plot:
                    pos_track.append(feedback['pos'])
                    u_track.append(pd)

                # if cnt % 100 == 0:
                #     print('target: ', target['target_pos'])
                #     print('pos: ', feedback['pos'])

            cnt += 1

        #TODO calculate error signal
        # test_accuracy =

    # nni.report_final_results(test_accuracy)

    # disconnect from vrep
    interface.disconnect()
    if plot:
        u_track = np.squeeze(np.array(u_track))
        pos_track = np.squeeze(np.array(pos_track))
        plt.figure()
        for ii, u in enumerate(u_track.T):
            plt.subplot(4,1,ii+1)
            plt.plot(u)
        plt.show()
        for ii, pos in enumerate(pos_track.T):
            plt.subplot(3,1,ii+1)
            plt.plot(pos)
        plt.show()
        np.savez_compressed('mine.npz', pos=pos_track, u=u_track)

fly_session()
