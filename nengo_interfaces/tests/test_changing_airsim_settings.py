# NOTE tests if the airsim interface warns the user if a setting.json parameter changes and requires
# a UE4 sim restart
# make sure that the Capture Settings: Width parameter does not match what you current have in your
# ~/Documents/Airsim/settings.json
import math

import nengo
import json
import numpy as np
from os.path import expanduser

from nengo_interfaces.airsim import AirSim

# read the current capture width so we can:
# - change it to test if we get warned, change it back
# - test that we do not get an exception if it doesn't change
home = expanduser("~")
with open(  # pylint: disable=W1514
    f"{home}/Documents/AirSim/settings.json", "r+"
) as fp:
    data = json.load(fp)
    prev_width = data["CameraDefaults"]["CaptureSettings"][0]["Width"]


# Test begins here
airsim_dt = 0.01
steps = 500

green = '\033[92m'
red = '\033[91m'
endc = '\033[0m'

# change a value in settings.json, this should throw a RuntimeError
try:
    interface = AirSim(
        dt=airsim_dt,
        camera_params={
            "use_physics": True,
            "fps": 1,
            "save_name": "hi",
            "camera_name": 0,
            "capture_settings": {"Width": prev_width+1},
        },
        show_display=True,
    )
except RuntimeError as e:
    print(f"{green}RuntimeError thrown as expected to warn user to restart UE4 sim to settings.json changes to take place, test passed{endc}")
    try:
        # set the value back to the original value
        interface = AirSim(
            dt=airsim_dt,
            camera_params={
                "use_physics": True,
                "fps": 1,
                "save_name": "hi",
                "camera_name": 0,
                "capture_settings": {"Width": prev_width},
            },
            show_display=True,
        )
    except RuntimeError as e:
        print(f"{green}Manually caught RuntimeError to change settings.json back{endc}")
        # set the value back to the original value
        interface = AirSim(
            dt=airsim_dt,
            camera_params={
                "use_physics": True,
                "fps": 1,
                "save_name": "hi",
                "camera_name": 0,
                "capture_settings": {"Width": prev_width},
            },
            show_display=True,
        )
        print(f"{green}No error thrown when settings.json does not change, test passed{endc}")

except Exception as e:
        # If the runtime error was not thrown, throw an error
        raise Exception (f"\n{red}If you are reading this, the Airsim Interface FAILED to throw an Exception to warn the user to restart the UE4 sim for changes in settings.json to take place{endc}")
