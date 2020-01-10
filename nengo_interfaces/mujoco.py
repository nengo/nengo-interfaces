import glfw
import nengo
import numpy as np
import sys

from abr_control.arms.mujoco_config import MujocoConfig
from abr_control.interfaces.mujoco import Mujoco as Interface


class MujocoProcess(nengo.Process):
    def __init__(self, robot_name, dt=0.001, update_display=1, **kwargs):

        self.dt = dt
        self.update_display = update_display

        self.config = MujocoConfig(robot_name)
        # need to connect the interface to fill in the config variables
        # connect with visualize = False so that nothing is rendered yet
        self.interface = Interface(self.config, dt=dt, visualize=False)
        self.interface.connect()

        size_in = self.config.N_JOINTS
        size_out = size_in * 2

        super().__init__(size_in, size_out, **kwargs)

    def make_step(self, shape_in, shape_out, dt, rng, state):

        self.interface.disconnect()
        self.interface = Interface(self.config, dt=self.dt)
        self.interface.connect()

        def step(t, u):
            update_display = not int(t / dt) % self.update_display
            self.interface.send_forces(u, update_display=update_display)
            feedback = self.interface.get_feedback()
            return np.hstack([feedback["q"], feedback["dq"]])

        return step


def Mujoco(robot_name, **kwargs):
    process = MujocoProcess(robot_name, **kwargs)
    return nengo.Node(
        process, process.default_size_in, process.default_size_out, label="Mujoco"
    )
