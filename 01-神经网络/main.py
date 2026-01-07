from nn import Network

if __name__ == "__main__":
    NET_SHAPE = (2, 3, 4, 2)
    network = Network(network_shape=NET_SHAPE)
    print(network.shape)
    print(network.layers)
    network.net_forward()
