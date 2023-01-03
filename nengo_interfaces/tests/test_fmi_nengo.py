"""
Example of how to run a controller with the FMU interface in the loop.

NOTE: If you don't know the variables for your FMU, set debug=True, and the list of
variables will be printed on instantiation.
"""
import os
import sys

import matplotlib.pyplot as plt
import nengo

from nengo_interfaces.fmi import FMI

# Simulation parameters
dt = 0.001  # time step
T = 30  # sim time (s)

# Instantiate FMI interface
# get the path to the FMU
if len(sys.argv) < 2:
    raise Exception("Provide the path to the fmu as a command line argument")
else:
    path = sys.argv[1]

interface = FMI(
    path=path,
    dt=dt,
    init_dict={"enable_full_steering_control": 0},
    input_keys=["tgt_residual_steering_angle"],
    feedback_keys=[
        "act_position_x",
        "act_position_y",
        "act_cross_track_error",
    ],
    debug=True,
)

model = nengo.Network(seed=12)
with model:

    def u_fun(t, x):
        return 0

    control_node = nengo.Node(u_fun, size_in=3, label="Controller")
    fmu_node = nengo.Node(interface, size_in=1, size_out=3, label="FMU")
    output_node = nengo.Node(size_in=3, label="Output")

    nengo.Connection(control_node, fmu_node, synapse=None)
    nengo.Connection(fmu_node, output_node, synapse=None)
    probe = nengo.Probe(output_node)

print(f"Running {T}s with dt={dt}")
sim = nengo.Simulator(network=model, seed=12, dt=dt)
with sim:
    sim.run(T)
print("Sim Finished.")


data = sim.data[probe]
# data is [T, len(feedback_keys)], stored in same order feedback keys are specified
x = data[:, 0]
y = data[:, 1]
error = data[:, 2]

fig, axs = plt.subplots(2, 1)
axs[0].plot(x, y)
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
axs[1].plot(error)
axs[1].set_xlabel("Time (ms)")
axs[1].set_ylabel("Error")

plt.tight_layout()
plt.show()
