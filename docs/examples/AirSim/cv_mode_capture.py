# Example how to capture drone POV in CV mode
# we turn off physics and move the camera around instead of the drone
# this simplifies our task to just using a path planner for the camera
# to follow exactly, no controller required
import math

import nengo
import numpy as np
import matplotlib.pyplot as plt

from nengo_interfaces.airsim import AirSim
from nengo_control.controllers.quadrotor import PD

# Test begins here
airsim_dt = 0.01
steps = 500

interface = AirSim(
    dt=airsim_dt,
    camera_params={
        "use_physics": False,
        "fps": 10,
        "save_name": "test_figure",
        "camera_name": 0,
        "Width": 960,
        "Height": 832
    },
    show_display=False,
)

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

    # wrap our interface in a node that takes in control signals
    # and outputs drone state
    interface_node = nengo.Node(interface, label="Airsim")

    # connect our target state to our controller
    nengo.Connection(target_node, interface_node, synapse=0)

with nengo.Simulator(model, dt=airsim_dt) as sim:
    sim.run(steps * airsim_dt)

interface.disconnect()
