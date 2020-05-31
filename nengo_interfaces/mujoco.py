import glfw
import matplotlib.pyplot as plt

import nengo
import numpy as np

from abr_control.arms.mujoco_config import MujocoConfig
from abr_control.interfaces.mujoco import Mujoco as Interface


class Mujoco(nengo.Process):
    """Wraps Mujoco interface in a Process for use in a Node.

    Parameters
    ----------
    xml_file: string
        The filename of the Mujoco XML to load.
    dt : float
        The time step for the simulation dynamics.
    update_display : int
        How often to render the graphics when running.
    render_params : dict
        cameras : list
        resolution : list
        update_frequency : int, optional (Default: 1)
        plot_frequency : int, optional (Default: None)
    joint_names : list, optional (Default: None)
        which joints from Mujoco to collect feedback from, in what order
        Default used are joints that connect the end-effector to the world
    track_input : bool, option (Default: False)
        if True, store the input that is passed in to the Node in self.input_track
    """

    def __init__(
        self,
        xml_file,
        dt=0.001,
        update_display=1,
        seed=None,
        render_params={},
        joint_names=None,
        track_input=False,
        **kwargs
    ):

        self.dt = dt
        self.update_display = update_display
        self.render_params = render_params
        self.joint_names = joint_names
        self.track_input = track_input
        self.exit = False

        self.input_track = []
        self.image = np.array([])
        if self.render_params:
            # if not empty, create array for storing rendered camera feedback
            self.camera_feedback = np.zeros(
                (
                    self.render_params["resolution"][0],
                    self.render_params["resolution"][1] * len(self.render_params["cameras"]),
                    3,
                )
            )
            self.subpixels = np.product(self.camera_feedback.shape)
            self.image = np.zeros(self.subpixels)
            if "frequency" not in self.render_params.keys():
                self.render_params["frequency"] = 1
            if "plot_frequency" not in self.render_params.keys():
                self.render_params["plot_frequency"] = None

        self.config = MujocoConfig(xml_file, **kwargs)
        # need to connect the interface to fill in the config variables
        # connect with visualize = False so that nothing is rendered yet
        self.interface = Interface(
            self.config,
            dt=self.dt,
            create_offscreen_rendercontext=bool(self.render_params),
        )
        self.connect()

        # size in = the number of actuators defined
        size_in = self.interface.sim.model.nu
        # size out = the number of joints we're getting feedback from x2 (pos, vel)
        size_out = len(self.interface.joint_pos_addrs) * 2 + int(
            np.product(self.image.shape)
        )

        super().__init__(size_in, size_out, default_dt=dt, seed=seed)

    def connect(self):
        """ Connect to the interface """
        self.interface.connect(joint_names=self.joint_names)

    def make_step(self, shape_in, shape_out, dt, rng, state):
        """End the previously instantiated interface, create and connect to a new
        Mujoco session.
        """
        if self.render_params:
            cameras = self.render_params['cameras']
            frequency = self.render_params['frequency']
            plot_frequency = self.render_params['plot_frequency']

            read_pixels = self.interface.offscreen.read_pixels
            render = self.interface.offscreen.render

        def step(t, u):
            """Takes in a set of forces. Returns the state of the robot after
            advancing the simulation one time step.
            """
            if self.track_input:
                self.input_track.append(np.copy(u))

            update_display = not int(t / self.dt) % self.update_display
            self.interface.send_forces(u, update_display=update_display)
            feedback = self.interface.get_feedback()
            timestep  = int(t / self.dt)

            # render camera feedback
            if self.render_params:
                if timestep % frequency == 0:

                    for ii, jj in enumerate(cameras):
                        render(
                            self.render_params["resolution"][0],
                            self.render_params["resolution"][1],
                            camera_id=jj,
                        )

                        self.camera_feedback[
                                :, ii * 32: 32 * (ii+1)] = read_pixels(
                            self.render_params["resolution"][0],
                            self.render_params["resolution"][1],
                        )[0]
                    self.image[:] = self.camera_feedback.flatten()

                if plot_frequency is not None:
                    if (timestep % plot_frequency) == 0:
                        plt.figure()
                        a = plt.subplot(1, 1, 1)
                        a.imshow(self.image.reshape((res[0], res[1], 3)) / 255)
                        plt.show()

            self.exit = self.interface.viewer.exit
            if self.exit:
                glfw.destroy_window(self.interface.viewer.window)

            return np.hstack([feedback["q"], feedback["dq"], self.image])

        return step

    def make_node(self):
        """Create a Node that wraps the MujocoProcess for interacting with a Mujoco
        simulation.
        """
        return nengo.Node(
            self,
            size_in=self.default_size_in,
            size_out=self.default_size_out,
            label="Mujoco",
        )

    def get_position(self, name, object_type="body"):
        """ Returns Cartesion position of object in world"""
        return self.config.Tx(name, object_type=object_type)

    def get_orientation(self, name, object_type="body"):
        """ Returns Cartesian orientation of object in world"""
        if object_type == "body":
            body_id = self.interface.sim.model.body_name2id(name)
            xmat = self.interface.sim.data.body_xmat[body_id]
        return xmat.reshape(3, 3)

    def set_color(self, name, color, object_type="geom"):
        if object_type == "geom":
            geom_id = self.interface.sim.model.geom_name2id(name)
            self.interface.sim.model.geom_rgba[geom_id] = color
        else:
            raise NotImplementedError

    def set_mocap_xyz(self, name, xyz):
        """ Set the position of a Mocap object in the environment """
        self.interface.set_mocap_xyz(name, xyz)
