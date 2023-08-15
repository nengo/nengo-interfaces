"""
Move the jao2 Mujoco arm to a target position.

The simulation ends after 1500 time steps, and the trajectory of the end-effector
is plotted in 3D.
"""
import sys
import traceback

import glfw
import nengo
import numpy as np
from abr_control.arms.mujoco_config import MujocoConfig
from abr_control.controllers import OSC, Damping
from abr_control.utils import transformations

from nengo_interfaces.mujoco import ExitSim, Mujoco

if len(sys.argv) > 1:
    arm_model = sys.argv[1]
else:
    arm_model = "jaco2"
# initialize our robot config for the jaco2
robot_config = MujocoConfig(arm_model)

# create our Mujoco interface
dt = 0.001

interface = Mujoco(
    robot_config=robot_config,
    dt=dt,
)
interface.connect(
    joint_names=[f"joint{ii}" for ii in range(len(robot_config.START_ANGLES))]
)
# interface.connect()
interface.send_target_angles(robot_config.START_ANGLES)

# damp the movements of the arm
damping = Damping(robot_config, kv=10)
# instantiate controller
ctrlr = OSC(
    robot_config,
    kp=200,
    null_controllers=[damping],
    vmax=[0.5, 0],  # [m/s, rad/s]
    # control (x, y, z) out of [x, y, z, alpha, beta, gamma]
    ctrlr_dof=[True, True, True, False, False, False],
)

# set up lists for tracking data
ee_track = []
target_track = []

target_geom = "target"
green = [0, 0.9, 0, 0.5]
red = [0.9, 0, 0, 0.5]

seed = 0
np.random.seed(seed)
net = nengo.Network(seed=seed)
net.config[nengo.Connection].synapse = None
with net:
    net.count = 0
    net.targets_reached = 0
    net.targets_to_reach = 5

    def gen_target():
        target_xyz = (np.random.rand(3) + np.array([-0.5, -0.5, 0.5])) * np.array(
            [1, 1, 0.5]
        )
        return target_xyz

    net.gen_target = gen_target
    net.target = net.gen_target()
    interface.set_mocap_xyz(name="target", xyz=net.target)

    mujoco_node = interface.make_node()
    # print(f"{interface.joint_dyn_addrs=}")
    # raise Exception

    def control_func(t, x):
        """Takes in joint angles and a target, returns the control signal."""
        # get the end-effector's initial position
        feedback = {}
        feedback["q"] = x[: robot_config.N_JOINTS]
        feedback["dq"] = x[robot_config.N_JOINTS :]

        target = np.hstack(
            [
                interface.get_xyz("target"),
                transformations.euler_from_quaternion(
                    interface.get_orientation("target"), "rxyz"
                ),
            ]
        )

        # calculate the control signal
        u = ctrlr.generate(
            q=feedback["q"],
            dq=feedback["dq"],
            target=target,
        )
        # u_grip = np.ones(3) * 0.2
        # return np.hstack((u, u_grip)).tolist()
        # print(f"{len(u)=}")
        # print(u)
        # print(f"{x=}")
        # print(f"{u=}")
        # raise Exception
        return u

    ctrlr_node = nengo.Node(
        size_in=robot_config.N_JOINTS * 2,
        size_out=robot_config.N_JOINTS,
        output=control_func,
    )
    nengo.Connection(mujoco_node, ctrlr_node, label="mujoco>control")
    # Need a synapse somewhere with the looping connection
    nengo.Connection(ctrlr_node, mujoco_node, synapse=0, label="control>mujoco")

    def vis_func(t, x):
        # Update visualizations and targets
        # calculate end-effector position
        feedback = {}
        feedback["q"] = x[: robot_config.N_JOINTS]
        feedback["dq"] = x[robot_config.N_JOINTS :]

        ee_xyz = robot_config.Tx("EE", q=feedback["q"])
        # track data
        ee_track.append(np.copy(ee_xyz))
        target_track.append(np.copy(net.target[:3]))

        error = np.linalg.norm(ee_xyz - net.target[:3])
        if error < 0.02:
            interface.model.geom(target_geom).rgba = green
            net.count += 1
        else:
            net.count = 0
            interface.model.geom(target_geom).rgba = red

        if net.count >= 50:
            print("Generating a new target")
            net.target = net.gen_target()
            interface.set_mocap_xyz(name="target", xyz=net.target)
            net.count = 0
            net.targets_reached += 1
            if net.targets_reached + 1 == net.targets_to_reach:
                raise ExitSim

    vis_node = nengo.Node(size_in=robot_config.N_JOINTS * 2, output=vis_func)
    nengo.Connection(mujoco_node, vis_node)

try:
    sim = nengo.Simulator(net, dt=dt)
    with sim:
        sim.run(100)

except Exception:
    print(traceback.format_exc())

finally:
    # stop and reset the Mujoco simulation
    interface.disconnect()

    print("Simulation terminated...")

    ee_track = np.array(ee_track)
    target_track = np.array(target_track)

    if ee_track.shape[0] > 0:
        # plot distance from target and 3D trajectory
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d  # pylint: disable=W0611

        fig = plt.figure(figsize=(8, 12))
        ax1 = fig.add_subplot(211)
        ax1.set_ylabel("Distance (m)")
        ax1.set_xlabel("Time (ms)")
        ax1.set_title("Distance to target")
        ax1.plot(
            np.sqrt(np.sum((np.array(target_track) - np.array(ee_track)) ** 2, axis=1))
        )

        ax2 = fig.add_subplot(212, projection="3d")
        ax2.set_title("End-Effector Trajectory")
        ax2.plot(ee_track[:, 0], ee_track[:, 1], ee_track[:, 2], label="ee_xyz")
        ax2.scatter(
            target_track[1, 0],
            target_track[1, 1],
            target_track[1, 2],
            label="target",
            c="r",
        )
        ax2.scatter(
            ee_track[0, 0],
            ee_track[0, 1],
            ee_track[0, 2],
            label="start",
            c="g",
        )
        ax2.legend()
        plt.show()
