# runs a simplified path planner and pd controller to test the motor control portion of the interface
import math

import nengo
import numpy as np

from nengo_interfaces.airsim import AirSim


def convert_angles(ang):
    """Converts Euler angles from x-y-z to z-x-y convention"""

    def b(num):
        """forces magnitude to be 1 or less"""
        if abs(num) > 1.0:
            return math.copysign(1.0, num)
        else:
            return num

    s1 = math.sin(ang[0])
    s2 = math.sin(ang[1])
    s3 = math.sin(ang[2])
    c1 = math.cos(ang[0])
    c3 = math.cos(ang[2])

    pitch = math.asin(b(c1 * c3 * s2 - s1 * s3))
    cp = math.cos(pitch)
    if cp == 0:  # just in case
        cp = 0.000001

    yaw = math.asin(b((c1 * s3 + c3 * s1 * s2) / cp))  # flipped
    # Fix for getting the quadrants right
    if c3 < 0 and yaw > 0:
        yaw = math.pi - yaw
    elif c3 < 0 and yaw < 0:
        yaw = -math.pi - yaw

    roll = math.asin(b((c3 * s1 + c1 * s2 * s3) / cp))  # flipped
    return [roll, pitch, yaw]


class PD:
    def __init__(self, gains):
        """
        PD controller for quadrotor without any dynamics compensation

        Parameters
        ----------

        gains: list of 8 floats
            the PD control gains for pos, lin_vel, rotation, ang_vel [k1-k8]
        """
        self.rotor_transform = np.array(
            [[-1, -1, -1, 1], [-1, 1, 1, 1], [-1, -1, 1, -1], [-1, 1, -1, -1]]
        )

        self.gains = np.array(
            [
                [0, 0, gains[1], 0, 0, gains[3], 0, 0, 0, 0, 0, 0],
                [gains[0], 0, 0, gains[2], 0, 0, 0, -gains[4], 0, 0, -gains[6], 0],
                [0, gains[0], 0, 0, gains[2], 0, gains[4], 0, 0, gains[6], 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, gains[5], 0, 0, gains[7]],
            ]
        )

    def calc_error(self, state, target):
        """
        Parameters
        ----------
        state: 12D list or array
            drone state in the form of
            np.array([
                x[m], y[m], z[m],
                lin_vel_x[m/s], lin_vel_y[m/s], lin_vel_z[m/s],
                roll[rad], pitch[rad], yaw[rad],
                ang_vel_roll[rad/sec], ang_vel_pitch[rad/s], ang_vel_yaw[rad/s]
            ]
        target: 12D list or array
            same format as state vector
        """
        # convert our targets to zxy orientation
        target[6:9] = convert_angles(target[6:9])
        # Find the error

        ori_err = [target[6] - state[6], target[7] - state[7], target[8] - state[8]]

        for ii in range(3):
            if ori_err[ii] > math.pi:
                ori_err[ii] -= 2 * math.pi
            elif ori_err[ii] < -math.pi:
                ori_err[ii] += 2 * math.pi

        cz = math.cos(state[8])
        sz = math.sin(state[8])
        x_err = target[0] - state[0]
        y_err = target[1] - state[1]

        pos_err = [
            x_err * cz + y_err * sz,
            -x_err * sz + y_err * cz,
            target[2] - state[2],
        ]

        dx_err = target[3] - state[3]
        dy_err = target[4] - state[4]
        dz_err = target[5] - state[5]

        lin_vel = [dx_err * cz + dy_err * sz, -dx_err * sz + dy_err * cz, dz_err]

        da_err = target[9] - state[9]
        db_err = target[10] - state[10]
        dg_err = target[11] - state[11]

        ang_vel = [da_err, db_err, dg_err]

        error = np.array(
            [
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
            ]
        )

        return error

    def generate(self, state, target):
        """
        Parameters
        ----------
        state: 12D list or array
            drone state in the form of
            np.array([
                x[m], y[m], z[m],
                lin_vel_x[m/s], lin_vel_y[m/s], lin_vel_z[m/s],
                roll[rad], pitch[rad], yaw[rad],
                ang_vel_roll[rad/sec], ang_vel_pitch[rad/s], ang_vel_yaw[rad/s]
            ]
        target: 12D list or array
            same format as state vector
        """

        self.error = self.calc_error(state=state, target=target)
        u_pd = np.dot(self.rotor_transform, np.dot(self.gains, self.error))
        # add gravity compensation
        u_pd += 6800

        return u_pd


# Test begins here
airsim_dt = 0.01
steps = 500

pd_ctrl = PD(
    gains=np.array(
        [
            8950.827941754635,
            5396.8148923228555,
            3797.2396183387336,
            2838.8455160747803,
            5817.333354627463,
            10763.75342891863,
            415.04893487790997,
            500.1385252571632,
        ]
    )
)

interface = AirSim(dt=airsim_dt)
interface.connect()

target = np.array([2, 1, -3, 0, 0, 0, 0, 0, 1.57, 0, 0, 0])

model = nengo.Network()
with model:
    state = interface.get_feedback()
    interface.set_state("target", target[:3], target[6:9])
    model.filtered_target = np.hstack(
        (
            np.hstack((state["position"], np.zeros(3))),
            np.hstack((state["taitbryan"], np.zeros(3))),
        )
    )

    start_xyz = state["position"]
    start_ang = state["taitbryan"]
    difference = target[:3] - start_xyz
    dist = np.linalg.norm(difference)
    xyz_step = dist / steps
    ang_step = (target[6:9] - start_ang) / steps
    direction = difference / dist

    def target_func(t):
        model.filtered_target[:3] += direction * xyz_step
        model.filtered_target[6:9] += ang_step
        interface.set_state(
            "filtered_target",
            xyz=model.filtered_target[:3],
            orientation=model.filtered_target[6:9],
        )
        return list(model.filtered_target)

    target_node = nengo.Node(target_func, size_out=12)

    def ctrl_func(t, x):
        return pd_ctrl.generate(x[:12], x[12:]).flatten()

    ctrl = nengo.Node(ctrl_func, size_in=24, size_out=4)

    interface_node = nengo.Node(interface, label="Airsim")

    nengo.Connection(interface_node, ctrl[:12], synapse=None)
    nengo.Connection(target_node, ctrl[12:], synapse=None)
    nengo.Connection(ctrl, interface_node, synapse=0)

with nengo.Simulator(model, dt=airsim_dt) as sim:
    sim.run(steps * airsim_dt)

interface.disconnect()
