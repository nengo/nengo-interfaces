"""Example of how to interface a controller with an FMU.

NOTE: If you don't know the variables for your FMU, set debug=True, and the list of
variables will be printed on instantiation.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from nengo_interfaces.fmi import FMI

# Simulation parameters
dt = 0.001  # time step
T = 30  # sim time (s)

# get the path to the FMU
if len(sys.argv) < 2:
    raise Exception("Provide the path to the fmu as a command line argument")
else:
    path = sys.argv[1]

# Instantiate FMU interface
interface = FMI(
    debug=True,
    path=path,
    dt=dt,
    init_dict={"enable_full_steering_control": 0},
    input_keys=["tgt_residual_steering_angle"],
    feedback_keys=["act_position_x", "act_position_y", "act_cross_track_error"],
)

x_track = []
y_track = []
error_track = []
print(f"Running {T}s with dt={dt}")
for t in tqdm(range(0, int(T / dt))):
    # calculate your control signal here given feedback
    u_res = np.array([0])
    # send motor control and step sim by dt
    interface.send_control(u=[u_res])
    # get the state feedback as a dict
    feedback = interface.get_feedback()

    x_track.append(feedback["act_position_x"])
    y_track.append(feedback["act_position_y"])
    error_track.append(feedback["act_cross_track_error"])
print("Sim Finished.")

fig, axs = plt.subplots(2, 1)
axs[0].plot(x_track, y_track)
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
axs[1].plot(error_track)
axs[1].set_xlabel("Time (ms)")
axs[1].set_ylabel("Error")

plt.tight_layout()
plt.show()
