# runs a simplified path planner and pd controller to test the motor control and feedback portion of the airsim interface
import math

import nengo
import numpy as np
import matplotlib.pyplot as plt

from nengo_interfaces.airsim import AirSim
from nengo_control.controllers.quadrotor import PD

# Test begins here
airsim_dt = 0.01
steps = 500

# Accepts 12D state as input and outputs a 4D control signal in radians/second
# in the rotor order: [front_right, rear_left, front_left, rear_right]
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
    # Set the position of an object called `target` in the UE4 env
    # to the final target position
    interface.set_state("target", target[:3], target[6:9])

    # Set our starting state to the current drone position and zero velocity
    state = interface.get_feedback()
    model.filtered_target = np.hstack(
        (
            np.hstack((state["position"], np.zeros(3))),
            np.hstack((state["taitbryan"], np.zeros(3))),
        )
    )

    # Define parameters and a Node function to output a controller target
    start_xyz = state["position"]
    start_ang = state["taitbryan"]
    difference = target[:3] - start_xyz
    dist = np.linalg.norm(difference)
    xyz_step = dist / steps
    ang_step = (target[6:9] - start_ang) / steps
    direction = difference / dist

    # Simple function that steps towards our target
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

    # wrap our non-neural controller in a node that takes
    # in target and drone state, and outputs rotor velocities
    def ctrl_func(t, x):
        return pd_ctrl.generate(x[:12], x[12:]).flatten()

    ctrl = nengo.Node(ctrl_func, size_in=24, size_out=4)

    # wrap our interface in a node that takes in control signals
    # and outputs drone state
    interface_node = nengo.Node(interface, label="Airsim")

    # connect our sim state to the controller
    nengo.Connection(interface_node, ctrl[:12], synapse=None)
    # connect our target state to our controller
    nengo.Connection(target_node, ctrl[12:], synapse=None)
    # connect our controller ouput to the interface to close the loop
    nengo.Connection(ctrl, interface_node, synapse=0)

    # add probes for plotting
    state_p = nengo.Probe(interface_node, synapse=0)
    target_p = nengo.Probe(target_node, synapse=0)
    ctrl_p = nengo.Probe(ctrl, synapse=0)

with nengo.Simulator(model, dt=airsim_dt) as sim:
    sim.run(steps * airsim_dt)

interface.disconnect()

# Plot results
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plt.title('Flight Path in NED Coordinates')
ax.plot(
    sim.data[target_p].T[0],
    sim.data[target_p].T[1],
    sim.data[target_p].T[2],
    label='target'
)
ax.plot(
    sim.data[state_p].T[0],
    sim.data[state_p].T[1],
    sim.data[state_p].T[2],
    label='state',
    linestyle='--'
)
plt.legend()

plt.figure()
plt.title('Control Commands')
plt.ylabel('Rotor Velocities [rad/sec]')
plt.xlabel('Time [sec]')
plt.plot(sim.trange(), sim.data[ctrl_p])
plt.legend(["front_right", "rear_left", "front_left", "rear_right"])
plt.show()
