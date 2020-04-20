import subprocess
import signal
import os
import numpy as np
import time
import nni
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
    def connect(self, cid=0, gui=False):

        self.vrep_mode = vrep.simx_opmode_oneshot
        SYNC = True

        remoteAPIConnections_loc = '/home/pawel/Downloads/V-REP_PRO_V3_5_0_Linux/remoteApiConnections.txt'
        # setup vrep connection
        # open the remote connection file and change the port that vrep looks for
        conn_file = open(remoteAPIConnections_loc, 'r')
        new_file = ''
        for line in conn_file:
            stripped_line = line.strip()
            if 'portIndex1_port' in stripped_line:
                stripped_line = 'portIndex1_port             = %i' % (19997 + cid)
            new_file += stripped_line + '\n'
        conn_file.close()

        write_file = open(remoteAPIConnections_loc, 'w')
        write_file.write(new_file)
        write_file.close()

        # if cid is None:
        # vrep.simxFinish(-1) # just in case, close all opened connections
        # else:
        # self.cid = cid

        if gui:
            command = [
             '/home/pawel/Downloads/V-REP_PRO_V3_5_0_Linux/./vrep.sh',
             ' /home/pawel/src/masters-thesis/quadcopter_experiments_simple.ttt']
        else:
            command = [
             'uxterm',
             '-e',
             '/home/pawel/Downloads/V-REP_PRO_V3_5_0_Linux/./vrep.sh',
             ' -h',
             ' /home/pawel/src/masters-thesis/quadcopter_experiments_simple.ttt']

        # open coppeliasim instance
        self.vrep_instance = subprocess.Popen(
                command,
            stdout=subprocess.PIPE)

        if gui:
            time.sleep(5)
        else:
            time.sleep(5)

        self.cid = vrep.simxStart('127.0.0.1',19997+cid,True,True,5000,5)
        # self.cid = vrep.simxStart('127.0.0.1',19997,True,True,5000,5)

        if self.cid == -1:
            raise Exception("Failed connecting to CoppeliaSim remote API server")

        print ('Connected to V-REP remote API server, client id: %s' % self.cid)
        # vrep.simxLoadScene(
        #     self.cid,
        #     '/home/pawel/src/masters-thesis/quadcopter_experiments_simple.ttt',
        #     0,
        #     self.vrep_mode
        # )

        if SYNC:
            vrep.simxSynchronous( self.cid, True )

        vrep.simxStartSimulation( self.cid, vrep.simx_opmode_oneshot )

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
        # end simulation
        err = vrep.simxStopSimulation(self.cid, vrep.simx_opmode_oneshot_wait )
        # end communication thread
        vrep.simxFinish(self.cid)
        # NOTE:this doesn't seem to do anything
        # kill bash instance
        # self.vrep_instance.kill()
        # sys.exit(0)
        os.kill(self.vrep_instance.pid, signal.SIGTERM)
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

def fly_session(use_nni=True, plot=False, step_limit=1000, gui=False):
    """
    phi == roll
    theta == pitch
    psi == yaw
    """
    pos_track = []
    target_track = []
    u_track = []

    interface = vrepInterface()

    if use_nni:
        print('SEQUENCE ID: %i' % nni.get_sequence_id())
        interface.connect(cid=nni.get_sequence_id(), gui=gui)
    else:
        interface.connect(cid=np.random.randint(1, 10), gui=gui)

    # n_attemps = 10
    # for ii in range(n_attemps):
    #     try:
    #         connected = True
    #         if use_nni:
    #             print('SEQUENCE ID: %i' % nni.get_sequence_id())
    #             interface.connect(cid=nni.get_sequence_id())
    #         else:
    #             interface.connect(cid=np.random.randint(1, 10))
    #             # interface.connect(gui=True)
    #     except: # ConnectionError:
    #         connected = False
    #         print('retrying connection...')
    #
    #     if connected:
    #         break

    # TODO fix this bug, need to send one feedback and one motor command before we can get feedback
    interface.get_feedback()
    interface.send_motor_commands(np.zeros(4))

    if use_nni:
        params = nni.get_next_parameter()
    else:
        # gravity comp
        # params = {
        #     'k7':3.5507965061299425,
        #     'k5':5.8918387400151335,
        #     'k4':9.984104311343183,
        #     'k2':9.9975624442988,
        #     'k3':8.351443088454126,
        #     'k8':7.793104952437566,
        #     'k1':7.13115856837296,
        #     'k6':5.332358267734109
        #  }
        params = {
        'k5':4.123516422497108,
        'k1':2.4495316824399342,
        'k6':5.628947290025429,
        'k7':3.2333984852224065,
        'k3':9.024561841578501,
        'k2':5.34314868270615,
        'k4':4.393081669969126,
        'k8':3.9650105222558802}

    targets = [
        {'target_pos': interface.get_feedback()['pos'], # + np.array([1, 1, 1]), #np.array([0, 0, 1]),
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
        target_state = np.array([
                target['target_pos'][0],
                target['target_pos'][1],
                target['target_pos'][2],
                target['target_ori'][0],
                target['target_ori'][1],
                target['target_ori'][2],
                target['target_lin_vel'][0],
                target['target_lin_vel'][1],
                target['target_lin_vel'][2],
                target['target_ang_vel'][0],
                target['target_ang_vel'][1],
                target['target_ang_vel'][2]
            ])

        test_accuracy = 0

        while cnt < step_limit:

            # brent only sent commands every 10 steps
            # if cnt % 10 == 0:
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
            gravity_compensation = np.array([[0.574], [0.571], [0.574], [0.571]])

            # this should be sqrt but will lead to errors if force is negative
            pd = np.dot(rotor_transform, np.dot(gains, error)) #+ gravity_compensation

            interface.send_motor_commands(pd)

            drone_state = np.array([
                    feedback['pos'][0],
                    feedback['pos'][1],
                    feedback['pos'][2],
                    feedback['ori'][0],
                    feedback['ori'][1],
                    feedback['ori'][2],
                    feedback['lin_vel'][0],
                    feedback['lin_vel'][1],
                    feedback['lin_vel'][2],
                    feedback['ang_vel'][0],
                    feedback['ang_vel'][1],
                    feedback['ang_vel'][2]
                ])

            test_accuracy += np.linalg.norm(drone_state - target_state)

            if plot:
                # pos_track.append(feedback['pos'])
                pos_track.append(drone_state)
                target_track.append(target_state)
                u_track.append(pd)

            if cnt % 500 == 0:
                print('target: ', target['target_pos'])
                print('pos: ', feedback['pos'])

            cnt += 1

        #TODO calculate error signal
        # test_accuracy =

    if use_nni:
        nni.report_final_result(test_accuracy)

    # disconnect from vrep
    interface.disconnect()
    if plot:
        u_track = np.squeeze(np.array(u_track))
        pos_track = np.squeeze(np.array(pos_track))
        target_track = np.squeeze(np.array(target_track))
        plt.figure()
        for ii, u in enumerate(u_track.T):
            plt.subplot(4,1,ii+1)
            plt.plot(u, label='avg: %.3f' % np.mean(u))
            plt.ylabel('u%i' % ii)
            plt.legend()
        plt.savefig('u_track')
        plt.show()
        titles = ['x', 'y', 'z', 'a', 'b', 'g', 'dx', 'dy', 'dz', 'da', 'db', 'dg']
        ylabs = ['pos', 'orientation', 'lin vel', 'ang vel']
        for ii, pos in enumerate(pos_track.T):
            plt.subplot(4, 3,ii+1)
            plt.plot(pos, linestyle='--')
            plt.plot(target_track.T[ii])
            plt.title(titles[ii])
            if ii % 3 == 0:
                plt.ylabel(ylabs[int(ii/3)])
        plt.tight_layout()
        plt.savefig('state_track')
        plt.show()
        np.savez_compressed('mine.npz', pos=pos_track, u=u_track)

fly_session(use_nni=True, plot=False, step_limit=1000, gui=False)
