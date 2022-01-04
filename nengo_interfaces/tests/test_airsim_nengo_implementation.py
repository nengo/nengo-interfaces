# Compares the feedback when using the nengo node vs interface directly
import time

import matplotlib.pyplot as plt
import nengo
import numpy as np

from nengo_interfaces.airsim import AirSim

dt = 0.01
sim_time = 3
control_u = np.array([6800, 6800, 6810, 6800])

# RUN THROUGH NENGO NODE
interface = AirSim(dt=dt)
interface.connect()
# let the drone settle to the ground
time.sleep(1)
net = nengo.Network(seed=0)

with net:
    interface_node = nengo.Node(interface)
    u = nengo.Node(lambda x: control_u, size_in=0, size_out=4)
    nengo.Connection(u, interface_node)
    state_probe = nengo.Probe(interface_node)
sim = nengo.Simulator(net, dt=0.01, seed=0)
sim.run(sim_time)

interface.disconnect()

# NENGO NODE CLASS WITHOUT USING NENGO
interface = AirSim(dt=dt)
interface.connect()
# let the drone settle to the ground
time.sleep(1)
state = []

labs = ["x", "y", "z", "dx", "dy", "dz", "a", "b", "g", "da", "db", "dg"]
for ii in range(0, int(sim_time / dt)):
    output = interface.get_feedback()
    output = np.hstack(
        (
            np.hstack((output["position"], output["linear_velocity"])),
            np.hstack((output["taitbryan"], output["angular_velocity"])),
        )
    )

    state.append(output)
    interface.send_pwm_signal(control_u)

state = np.asarray(state).T
interface.disconnect()

# NOTE can add test with old interface if we add it to abr_control?
# OLD INTERFACE WITHOUT NENGO
# from airsim_interface import airsimInterface
# interface = airsimInterface(dt=dt)
# interface.connect()
# state_old = []
#
# labs = ['x', 'y', 'z', 'dx', 'dy', 'dz', 'a', 'b', 'g', 'da', 'db', 'dg']
# for ii in range(0, int(sim_time/dt)):
#     output = interface.get_feedback()
#     output = np.hstack((
#         np.hstack((output['pos'], output['lin_vel'])),
#         np.hstack((output['ori'], output['ang_vel']))
#     ))
#
#     state_old.append(output)
#     interface.send_motor_commands(control_u)
#
# state_old = np.asarray(state_old).T
# interface.disconnect()
#
#
#

# COMPARE
plt.figure()
data = sim.data[state_probe].T
print(data.shape)
for ii in range(0, len(data)):
    plt.subplot(len(data), 1, ii + 1)
    plt.plot(data[ii], label="%s Node" % labs[ii])
    plt.plot(state[ii], label="%s Node No Nengo" % labs[ii])
    # plt.plot(state_old[ii], label='%s Old Airsim' % labs[ii])
    plt.legend()
plt.show()
