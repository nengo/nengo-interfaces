# NOTE tests if the airsim interface warns the user if a setting.json parameter changes and requires
# a UE4 sim restart
# make sure that the Capture Settings: Width parameter does not match what you current have in your
# ~/Documents/Airsim/settings.json
import json
import math
from os.path import expanduser

import nengo
import numpy as np

from nengo_interfaces.airsim import AirSim

green = "\033[92m"
yellow = "\u001b[33m"
red = "\033[91m"
endc = "\033[0m"

# read the current capture width so we can:
# - change it to test if we get warned, change it back
# - test that we do not get an exception if it doesn't change
home = expanduser("~")
with open(  # pylint: disable=W1514
    f"{home}/Documents/AirSim/settings.json", "r+"
) as fp:
    data = json.load(fp)
    try:
        prev_width = data["CameraDefaults"]["CaptureSettings"][0]["Width"]
    except KeyError:
        print(
            f"{yellow}No CameraDefaults found in settings.json, using default airsim values for testing{endc}"
        )
        prev_width = 256


# Test begins here
airsim_dt = 0.01
steps = 500


# change a value in settings.json, this should throw a RuntimeError
try:
    interface = AirSim(
        dt=airsim_dt,
        camera_params={
            "use_physics": True,
            "fps": 1,
            "save_name": None,
            "camera_name": 0,
            "capture_settings": {"Width": prev_width + 1},
        },
        show_display=True,
    )
except RuntimeError:
    print(
        f"{green}RuntimeError thrown as expected to warn user to restart UE4 sim to settings.json changes to take place, test passed{endc}"
    )
    try:
        # set the value back to the original value
        interface = AirSim(
            dt=airsim_dt,
            camera_params={
                "use_physics": True,
                "fps": 1,
                "save_name": None,
                "camera_name": 0,
                "capture_settings": {"Width": prev_width},
            },
            show_display=True,
        )
    except RuntimeError:
        print(f"{green}Manually caught RuntimeError to change settings.json back{endc}")
        # set the value back to the original value
        interface = AirSim(
            dt=airsim_dt,
            camera_params={
                "use_physics": True,
                "fps": 1,
                "save_name": None,
                "camera_name": 0,
                "capture_settings": {"Width": prev_width},
            },
            show_display=True,
        )
        print(
            f"{green}No error thrown when settings.json does not change, test passed{endc}"
        )

except Exception:
    # If the runtime error was not thrown, throw an error
    raise Exception(
        f"\n{red}If you are reading this, the Airsim Interface FAILED to throw an Exception to warn the user to restart the UE4 sim for changes in settings.json to take place{endc}"
    )
