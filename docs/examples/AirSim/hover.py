from nengo_interfaces.airsim import AirSim
import nengo

airsim = AirSim()

with nengo.Network() as net:
    input = nengo.Node([1000]*4)
    anode = airsim.make_node()
    nengo.Connection(input, anode)

with nengo.Simulator(net) as sim:
    sim.run(.1)

airsim._disconnect()
