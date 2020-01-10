import nengo
import nengo_interfaces

with nengo.Network() as net:
    input_node = nengo.Node([10, -10, 10, 10, 10, 10])
    interface = nengo_interfaces.CoppeliaSim("ur5")
    nengo.Connection(input_node, interface)

with nengo.Simulator(net) as sim:
    sim.run(1)
