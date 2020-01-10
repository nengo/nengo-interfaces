import importlib
import nengo
import numpy as np

from abr_control.interfaces.pygame import PyGame as Interface


class PyGameProcess(nengo.Process):
    def __init__(self, robot_name, dt=0.001, update_display=20,
                 on_click=None, on_keypress=None, **kwargs):

        self.dt = dt
        self.update_display = update_display
        self.on_click = on_click
        self.on_keypress = on_keypress

        arm = importlib.import_module('abr_control.arms.' + robot_name)
        self.config = arm.Config()
        self.arm_sim = arm.ArmSim(self.config)
        self.interface = None

        size_in = self.config.N_JOINTS
        size_out = size_in * 2

        super().__init__(size_in, size_out, **kwargs)


    def make_step(self, shape_in, shape_out, dt, rng, state):

        if self.interface is not None:
            self.interface.disconnect()

        self.interface = Interface(
            self.config,
            self.arm_sim,
            dt=self.dt,
            on_click=self.on_click,
            on_keypress=self.on_keypress
        )
        self.interface.connect()

        def step(t, u):
            update_display = not int(t / dt) % self.update_display
            self.interface.send_forces(u, dt=self.dt, update_display=update_display)
            feedback = self.interface.get_feedback()
            return np.hstack([feedback['q'], feedback['dq']])

        return step


def PyGame(robot_name, **kwargs):
    process = PyGameProcess(robot_name, **kwargs)
    return nengo.Node(
        process,
        process.default_size_in,
        process.default_size_out,
        label='PyGame'
    )
