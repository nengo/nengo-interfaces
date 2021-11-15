import os
from collections import OrderedDict

import nengo
import numpy as np
import pyfmi


class FMI(nengo.Process):
    """
    Provides an API for FMUs in Nengo models

    Parameters
    ----------
    path: string
        The full path to the FMU file
    dt: float
        The time step (seconds)
    debug: bool, Optional (Default: False)
        When True, more information is printed during simulation
    init_dict: dictionary, Optional (Default: None)
        A set of state variables and values to set at initialization
    input_keys: list of strings, Optional (Default: None)
        List of FMI input variables to be controlled
        NOTE: Assume that each input is 1 dimensional
    feedback_keys: list, Optional (Default: None)
        List of FMU output variables to return as feedback
    """

    def __init__(
        self, path, dt, debug=False, init_dict=None, input_keys=None, feedback_keys=None
    ):

        self.dt = dt
        self.debug = debug
        self.init_dict = init_dict
        self.curr_dir = os.path.dirname(os.path.abspath(__file__))

        # Load the dynamic library and XML data
        self.model = pyfmi.load_fmu(path)
        self.opts = self.model.simulate_options()
        self.opts["ncp"] = 1  # set the number of communication points to 1
        self.data_dict = self.gen_data_dict(debug=self.debug)

        if self.debug:
            print(f"FMU Model: {path}")
            print(f"Model Type: {type(self.model)}")
            print("---Available Parameters---\n ")
            for key in self.data_dict:
                print(f"-{key}: [{self.data_dict[key]}]")

        self.input_keys = [] if input_keys is None else input_keys
        self.feedback_keys = [] if feedback_keys is None else feedback_keys

        self.reset()  # initialize the model and self.t
        super().__init__(0, 0, default_dt=0.001, seed=1)

    def gen_data_dict(self, debug=False):
        """
        Returns a dictionary of available I/O keys in the FMU. The keys have a tuple
        value for (int(ndim_data), [])

        Takes raw keys available in model, and creates a dict with those keys,
        with tuple values. The first entry is the dimensionality of that data type,
        and the second is an empty list to store data. Keys that are arrays
        (ie. contain "[" in their name, such as "val[0]"... "val[N]") are amalgamated
        into a key with the base (ignoring the square brackets and index).

        The dimensionality is stored in the first entry of the tuple. When data is
        returned from the sim, the dimensionality is used to make sure all the data is
        collected.

        Parameters
        ----------
        debug: bool, Optional (Default: False)
            when True more information will print out during sim
        """
        self.params = self.model.get_model_variables()
        self.array_keys = []
        result_dict = {}
        # cycle through all keys
        for ii, key in enumerate(self.params):
            if debug:
                if ii == 0:
                    print("__________\nRaw keys loaded from FMI model")
                print(f"- {key}")

            # if the key contains square open bracket,
            # it is an array entry with multiple values
            if "[" in key:
                subkey = key.split("[")[0]
                # save the array key without its index for further processing
                # if not already saved to the list
                if subkey not in self.array_keys:
                    self.array_keys.append(subkey)
            else:
                # we only have one entry, so add it with a length of 1
                result_dict[key] = (1, [])

        # go through our array keys and count how many indexed entries it hasj
        # NOTE this is assuming the numbering will start at 0 and increments by 1
        for key in self.array_keys:
            n_dims = 0
            for raw_key in self.params:
                if key in raw_key:
                    n_dims += 1

            # set the dimensionality based on the number of time the subkey is found
            result_dict[key] = (n_dims, [])

        return result_dict

    def reset(self):
        """
        Resets the local variable tracking sim time and re-initializes the model
        """
        self.model.initialize()
        self.set_state(self.init_dict)
        self.t = 0.0

    def make_step(self, shape_in, shape_out, dt, rng, state):
        """Create the function for the FMU interfacing Nengo Node"""

        def step(t, x):
            """
            Send input, then step through the FMU sim by dt and return feedback
            """
            self.send_control(x)

            feedback_dict = self.get_feedback(self.feedback_keys)
            feedback = np.hstack(feedback_dict.values())
            return feedback

        return step

    def set_state(self, var_dict):
        """
        Accepts dict with key value pairs that match FMU variables and the value to set
        them to.
        """
        for key in var_dict:
            self.model.set(key, var_dict[key])

    def send_control(self, u):
        """
        Accepts dict with key value pairs that match FMU variables and the value to set
        them to for the next time step of simulation. Then simulates a time step.

        Parameters
        ----------
        u: float, list, np.array
            A float or set of values the same length as self.input_keys
            NOTE: Each dimension of u is assumed to correspond to a separate input_key
        """
        # create dictionary of input states and values to be set to
        u_dict = {}
        for ii, key in enumerate(self.input_keys):
            u_dict[key] = u[ii]
        # set the input variables
        self.set_state(u_dict)

        # step the FMU sim forward one step of length dt
        status = self.model.do_step(current_t=self.t, step_size=self.dt)
        self.t += self.dt

        if status != pyfmi.fmi.FMI_OK:
            raise Exception("ERROR IN do_step")

    def get_feedback(self, feedback_keys=None):
        """
        Accepts list of strings corresponding to FMU variables,
        and returns a dict with the string as a key and the value matching the
        current sim value of that variable

        Parameters
        ----------
        feedback_keys: list, Optional (Default: None)
            list of strings corresponding to the variables in the FMU that
            we want feedback from
        """
        feedback_keys = self.feedback_keys if feedback_keys is None else feedback_keys

        feedback = OrderedDict()
        for key in feedback_keys:
            # check if the value we want to return is an array
            if key in self.array_keys:
                # if it is, create an array with the whole vector
                _tmp = []
                for ii in range(0, self.data_dict[key][0]):
                    _tmp.append(self.model.get(f"{key}[{ii}]"))
                feedback[key] = np.squeeze(np.asarray(_tmp))
            else:
                feedback[key] = self.model.get(key)
        return feedback
