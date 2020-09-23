from nengo_interfaces.airsim import AirSim
import nengo

airsim = AirSim()
airsim.connect()

with nengo.Network() as net:
    input = nengo.Node([1000] * 4)
    anode = nengo.Node(airsim, label="AirSim")
    nengo.Connection(input, anode)

with nengo.Simulator(net) as sim:
    sim.run(0.1)

# Eventually we should be able to do this automatically (but not yet)
airsim.disconnect()
