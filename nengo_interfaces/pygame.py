import importlib
import nengo
import numpy as np

from abr_control.interfaces.pygame import PyGame as Interface


class PyGameProcess(nengo.Process):
    """Wraps PyGame interface in a Process for use in a Node.

    Parameters
    ----------
    robot_name : string
        The filename of the robot to load into PyGame.
    dt : float
        The time step for the simulation dynamics.
    update_display : int
        How often to render the graphics when running.
    on_click : function
        Takes in (self, mouse_x, mouse_y), where self is the PyGame viewer instance.
    on_keypress : function
        Takes in (self, key), where self is the PyGame viewer instance.
    """

    def __init__(
        self,
        robot_name,
        dt=0.001,
        update_display=20,
        on_click=None,
        on_keypress=None,
        **kwargs
    ):

        self.dt = dt
        self.update_display = update_display
        self.on_click = on_click
        self.on_keypress = on_keypress

        arm = importlib.import_module("abr_control.arms." + robot_name)
        self.config = arm.Config()
        self.arm_sim = arm.ArmSim(self.config)
        self.interface = None

        size_in = self.config.N_JOINTS
        size_out = size_in * 2

        super().__init__(size_in, size_out, **kwargs)

    def make_step(self, shape_in, shape_out, dt, rng, state):
        """End the previously instantiated interface, create and connect to a new
        PyGame session.
        """
        if self.interface is not None:
            self.interface.disconnect()

        self.interface = Interface(
            self.config,
            self.arm_sim,
            dt=self.dt,
            on_click=self.on_click,
            on_keypress=self.on_keypress,
        )
        self.interface.connect()

        def step(t, u):
            """Takes in a set of forces. Returns the state of the robot after
            advancing the simulation one time step.
            """
            update_display = not int(t / dt) % self.update_display
            self.interface.send_forces(u, dt=self.dt, update_display=update_display)
            feedback = self.interface.get_feedback()
            return np.hstack([feedback["q"], feedback["dq"]])

        return step


def PyGame(robot_name, **kwargs):
    """Create a Node that wraps the PyGameProcess for interacting with a
    PyGame simulation.
    """
    process = PyGameProcess(robot_name, **kwargs)
    return nengo.Node(
        process, process.default_size_in, process.default_size_out, label="PyGame"
    )
