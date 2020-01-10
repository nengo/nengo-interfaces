****************
Nengo Interfaces
****************

The Nengo Interfaces library is a Python package that provides simple API interfaces
for different simulation environments for use in Nengo models.

Installation
============

To install Nengo Interfaces, run::

    pip install -e .

from the root directory.

Usage
=====

Mujoco Interface
----------------

To use the Mujoco interface, you must provide the path to the XML file written in
MJCF that describes the environment you want to simulate. You must also provide a list
of joint names from which you would like feedback.

Once you create the interface, you can create the Node in Nengo using the `make_node`
method::

    interface = nengo_interface.mujoco.Mujoco('rover.xml', ['steering_wheel'])
    with nengo.Network() as net:
        mujoco_node = interface.make_node()

The input to this node is the set of forces to apply to the joints being controlled.
The output returned from this node is a vector of the joint angles and then joint
velocities of each of the joints specified using the `joint_list` parameter.

If the `render_params` parameter has identified cameras in the environment to render
the image from, then the output of `mujoco_node` will also include flattened output
from each of the identified cameras concatenated.

There are also several handy functions provided to call from the interface if you need
information from the environment, including

* ``get_position``
* ``get_orientation``
* ``set_mocap_xy``
* ``set_mocap_orientation``
