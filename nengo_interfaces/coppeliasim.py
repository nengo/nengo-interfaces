import importlib
import nengo
import numpy as np

from abr_control.interfaces.coppeliasim import CoppeliaSim as Interface


class CoppeliaSimProcess(nengo.Process):
    """Wraps CoppeliaSim interface in a Process for use in a Node.

    Parameters
    ----------
    robot_name : string
        The filename of the robot to load into CoppeliaSim.
    dt : float
        The time step for the simulation dynamics.
    """

    def __init__(self, robot_name, dt=0.001, **kwargs):

        self.dt = dt

        arm = importlib.import_module("abr_control.arms." + robot_name)
        self.config = arm.Config()
        self.interface = None

        size_in = self.config.N_JOINTS
        size_out = size_in * 2

        super().__init__(size_in, size_out, **kwargs)

    def make_step(self, shape_in, shape_out, dt, rng, state):
        """End the previously instantiated interface, create and connect to a new
        CoppeliaSim session.
        """
        if self.interface is not None:
            self.interface.disconnect()

        self.interface = Interface(self.config, dt=self.dt)
        self.interface.connect()

        def step(t, u):
            """Takes in a set of forces. Returns the state of the robot after
            advancing the simulation one time step.
            """
            self.interface.send_forces(u)
            feedback = self.interface.get_feedback()
            return np.hstack([feedback["q"], feedback["dq"]])

        return step


def CoppeliaSim(robot_name, **kwargs):
    """Create a Node that wraps the CoppeliaSimProcess for interacting with a
    CoppeliaSim simulation.
    """
    process = CoppeliaSimProcess(robot_name, **kwargs)
    return nengo.Node(
        process, process.default_size_in, process.default_size_out, label="CoppeliaSim"
    )
