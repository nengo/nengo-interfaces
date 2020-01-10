import importlib
import nengo
import numpy as np

from abr_control.interfaces.coppeliasim import CoppeliaSim as Interface


class CoppeliaSimProcess(nengo.Process):
    def __init__(self, robot_name, dt=0.001, **kwargs):

        self.dt = dt

        arm = importlib.import_module('abr_control.arms.' + robot_name)
        self.config = arm.Config()
        print(self.config)
        self.interface = None

        size_in = self.config.N_JOINTS
        size_out = size_in * 2

        super().__init__(size_in, size_out, **kwargs)


    def make_step(self, shape_in, shape_out, dt, rng, state):

        if self.interface is not None:
            self.interface.disconnect()

        self.interface = Interface(self.config, dt=self.dt)
        self.interface.connect()

        def step(t, u):
            self.interface.send_forces(u)
            feedback = self.interface.get_feedback()
            return np.hstack([feedback['q'], feedback['dq']])

        return step


def CoppeliaSim(robot_name, **kwargs):
    process = CoppeliaSimProcess(robot_name, **kwargs)
    return nengo.Node(
        process,
        process.default_size_in,
        process.default_size_out,
        label='CoppeliaSim'
    )
