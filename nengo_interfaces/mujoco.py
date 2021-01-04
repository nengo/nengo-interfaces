from xml.etree import ElementTree

import glfw
import matplotlib.pyplot as plt
import mujoco_py as mjp
import nengo
import numpy as np
from mujoco_py.generated import const


class Mujoco(nengo.Process):
    """Provides an easy to use API for Mujoco in Nengo models.

    Parameters
    ----------
    xml_file : string
        The filename of the Mujoco XML to load.
    joint_names : list
        Which joints from Mujoco to collect feedback from, in what order.
    dt : float
        The time step for the simulation dynamics.
    update_display : int
        How often to render the graphics when running.
    render_params : dict
        'cameras' : list
        'resolution' : list
        'update_frequency' : int, optional (Default: 1)
        'plot_frequency' : int, optional (Default: None)
    track_input : bool, option (Default: False)
        If True, store the input that is passed in to the Node in self.input_track.
    visualize : bool, optional (Default: True)
        If True, render the environment every update_display time steps.
    input_bias : int, optional (Default: 0)
        Added to the input signal to the Mujoco environment, after scaling.
    input_scale : int, optional (Default: 1)
        Multiplies the input signal to the Mujoco environment, before biasing.
    seed : int, optional (Default: None)
        Set the seed on the rng.
    """

    def __init__(
        self,
        xml_file,
        joint_names,
        dt=0.001,
        update_display=1,
        render_params=None,
        track_input=False,
        visualize=True,
        input_bias=0,
        input_scale=1,
        seed=None,
    ):

        self.xml_file = xml_file
        self.dt = dt
        self.update_display = update_display
        self.render_params = render_params if render_params is not None else {}
        self.joint_names = joint_names
        self.track_input = track_input
        self.visualize = visualize
        self.input_bias = input_bias
        self.input_scale = input_scale
        self.exit = False

        self.input_track = []
        self.image = np.array([])
        if self.render_params:
            # if not empty, create array for storing rendered camera feedback
            self.camera_feedback = np.zeros(
                (
                    self.render_params["resolution"][0],
                    self.render_params["resolution"][1]
                    * len(self.render_params["cameras"]),
                    3,
                )
            )
            self.subpixels = np.product(self.camera_feedback.shape)
            self.image = np.zeros(self.subpixels)
            if "frequency" not in self.render_params.keys():
                self.render_params["frequency"] = 1
            if "plot_frequency" not in self.render_params.keys():
                self.render_params["plot_frequency"] = None

        self.create_offscreen_rendercontext = bool(self.render_params)

        self.model = mjp.load_model_from_path(self.xml_file)
        # set the time step for simulation
        self.model.opt.timestep = self.dt

        self._connect(joint_names=joint_names)

        # get access to some of our custom arm parameters from the xml definition
        tree = ElementTree.parse(self.xml_file)
        root = tree.getroot()
        for custom in root.findall("custom/numeric"):
            name = custom.get("name")
            if name == "START_ANGLES":
                START_ANGLES = custom.get("data").split(" ")
                self.START_ANGLES = np.array([float(angle) for angle in START_ANGLES])
            elif name == "N_GRIPPER_JOINTS":
                self.N_GRIPPER_JOINTS = int(custom.get("data"))

        # size in = the number of actuators defined
        size_in = self.sim.model.nu
        # size out = the number of joints we're getting feedback from x2 (pos, vel)
        size_out = len(self.joint_pos_addrs) * 2 + int(np.product(self.image.shape))

        super().__init__(size_in, size_out, default_dt=dt, seed=seed)

    def _connect(self, joint_names, camera_id=-1):
        """Connect to the interface.

        Parameters
        ----------
        joint_names : list
            The set of joints to gather feedback from.
        camera_id : int, optional (Default: -1)
            The ID of the camera to use when visualizing the environment.
            Can also be changed in-simulation with the Tab key.
        """
        self.sim = mjp.MjSim(self.model)
        self.sim.forward()  # run forward to fill in sim.data
        model = self.sim.model
        self.model = model

        joint_ids = [model.joint_name2id(name) for name in joint_names]
        self.joint_pos_addrs = [model.get_joint_qpos_addr(name) for name in joint_names]
        self.joint_vel_addrs = [model.get_joint_qvel_addr(name) for name in joint_names]

        # Need to also get the joint rows of the Jacobian, inertia matrix, and
        # gravity vector. This is trickier because if there's a quaternion in
        # the joint (e.g. a free joint or a ball joint) then the joint position
        # address will be different than the joint Jacobian row. This is because
        # the quaternion joint will have a 4D position and a 3D derivative. So
        # we go through all the joints, and find out what type they are, then
        # calculate the Jacobian position based on their order and type.
        index = 0
        self.joint_dyn_addrs = []
        for ii, joint_type in enumerate(model.jnt_type):
            if ii in joint_ids:
                self.joint_dyn_addrs.append(index)
            if joint_type == 0:  # free joint
                index += 6  # derivative has 6 dimensions
            elif joint_type == 1:  # ball joint
                index += 3  # derivative has 3 dimensions
            else:  # slide or hinge joint
                index += 1  # derivative has 1 dimensions

        # if we want to use the offscreen render context create it before the
        # viewer so the corresponding window is behind the viewer
        if self.create_offscreen_rendercontext:
            self.offscreen = mjp.MjRenderContextOffscreen(self.sim, 0)

        # create the visualizer
        if self.visualize:
            self.viewer = mjp.MjViewer(self.sim)
            # if specified, set the camera
            if camera_id > -1:
                self.viewer.cam.type = const.CAMERA_FIXED
                self.viewer.cam.fixedcamid = camera_id

        print("MuJoCo session created")

    def make_step(self, shape_in, shape_out, dt, rng, state):
        """Create the function for the Mujoco interfacing Nengo Node."""
        res = self.render_params["resolution"]

        def step(t, u):
            """Takes in a set of forces. Returns the state of the robot after
            advancing the simulation one time step.

            Parameters
            ----------
            u : float
                The control signal to send to Mujoco simulation.
            """
            # bias and scale the input signal
            u += self.input_bias
            u *= self.input_scale

            if self.track_input:
                self.input_track.append(np.copy(u))

            timestep = int(t / self.dt)
            update_display = not timestep % self.update_display
            self._send_forces(u, update_display=update_display)
            feedback = self.get_feedback()

            # render camera feedback
            if self.render_params:
                if timestep % self.render_params["frequency"] == 0:

                    for ii, jj in enumerate(self.render_params["cameras"]):
                        self.offscreen.render(res[0], res[1], camera_id=jj)

                        self.camera_feedback[
                            :, ii * res[1] : (ii + 1) * res[1]
                        ] = self.offscreen.read_pixels(res[0], res[1], depth=False)
                    self.image[:] = self.camera_feedback.flatten()

                if self.render_params["plot_frequency"] is not None:
                    if (timestep % self.render_params["plot_frequency"]) == 0:
                        plt.figure()
                        a = plt.subplot(1, 1, 1)
                        a.imshow(self.image.reshape((res[0], res[1], 3)) / 255)
                        plt.show()

            if self.visualize:
                self.exit = self.viewer.exit
                if self.exit:
                    glfw.destroy_window(self.viewer.window)

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

    def _send_forces(self, u, update_display=True):
        """Apply the specified torque to the robot joints

        Apply the specified torque to the robot joints, move the simulation
        one time step forward, and update the position of the hand object.

        Parameters
        ----------
        u : np.array
            The torques to apply to the robot [Nm].
        update_display : boolean, optional (Default:True)
            Toggle for rendering the visualization.
        """

        # NOTE: the qpos_addr's are unrelated to the order of the motors
        self.sim.data.ctrl[:] = u[:]

        # move simulation ahead one time step
        self.sim.step()

        if self.visualize and update_display:
            self.viewer.render()

    def get_feedback(self):
        """Return a dictionary of information needed by the controller.

        Returns the joint angles and joint velocities in [rad] and [rad/sec],
        respectively.
        """

        q = np.copy(self.sim.data.qpos[self.joint_pos_addrs])
        dq = np.copy(self.sim.data.qvel[self.joint_vel_addrs])

        return {"q": q, "dq": dq}

    def get_position(self, name, object_type="body"):
        """Returns Cartesion position of object in world.

        Parameters
        ----------
        name : string
            The name of the object.
        object_type : string, optional (Default: 'body')
            The type of the object.
        """

        if object_type == "body":
            position = self.sim.data.get_body_xpos(name)
        elif object_type == "geom":
            position = self.sim.data.get_geom_xpos(name)
        elif object_type == "joint":
            position = self.sim.data.get_joint_xanchor(name)
        elif object_type == "site":
            position = self.sim.data.get_site_xpos(name)
        elif object_type == "camera":
            position = self.sim.data.get_cam_xpos(name)
        elif object_type == "light":
            position = self.sim.data.get_light_xpos(name)
        elif object_type == "mocap":
            position = self.sim.data.get_mocap_pos(name)
        else:
            raise Exception("Invalid object type specified: ", object_type)

        return np.copy(position)

    def get_orientation(self, name, object_type="body"):
        """Returns Cartesian orientation of object in world as a rotation matrix.

        Parameters
        ----------
        name : string
            The name of the object.
        object_type : string
            The type of object.
        """
        if object_type == "body":
            body_id = self.sim.model.body_name2id(name)
            xmat = self.sim.data.body_xmat[body_id]
        else:
            raise NotImplementedError

        return xmat.reshape(3, 3)

    def set_color(self, name, color, object_type="geom"):
        """Change the color of an object.

        Parameters
        ----------
        name : string
            The name of the object.
        color : list
            The [r, g, b, a] to set.
        object_type : string
            The type of object.
        """
        if object_type == "geom":
            geom_id = self.sim.model.geom_name2id(name)
            self.sim.model.geom_rgba[geom_id] = color
        else:
            raise NotImplementedError

    def set_mocap_xyz(self, name, xyz):
        """Set the position of a Mocap object in the Mujoco environment.

        Parameters
        ----------
        name : string
            The name of the MoCap object.
        xyz : np.array
            The target [x,y,z] location [meters].
        """
        self.sim.data.set_mocap_pos(name, xyz)

    def set_mocap_orientation(self, name, quat):
        """Sets the orientation of an object in the Mujoco environment

        Sets the orientation of an object using the provided Euler angles.
        Angles must be in a relative xyz frame.

        Parameters
        ----------
        name : string
            The name of the MoCap object.
        quat : np.array
            The target [w x y z] quaternion [radians].
        """
        self.sim.data.set_mocap_quat(name, quat)
