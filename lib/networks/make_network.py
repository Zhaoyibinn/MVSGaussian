import os
import imp
from lib.networks.mvsgs.network import Network 

def make_network(cfg) -> Network:
    module = cfg.network_module
    path = cfg.network_path
    network = imp.load_source(module, path).Network()
    return network
