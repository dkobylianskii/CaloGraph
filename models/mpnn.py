import dgl
import dgl.function as fn
import torch
import torch.nn as nn

from models.mlp import MLP


class NodeNetwork(nn.Module):
    def __init__(self, inputsize, outputsize, layers, feature="features"):
        super().__init__()
        self.feature = feature
        self.net = MLP(
            input_size=3 * inputsize, output_size=outputsize, hidden_layers=layers
        )

    def forward(self, x):
        inputs = torch.sum(x.mailbox["message"], dim=1)
        inputs = torch.cat(
            [inputs, x.data[self.feature], x.data["global_features"]], dim=1
        )

        output = self.net(inputs)
        output = output / (torch.norm(output, p="fro", dim=1, keepdim=True) + 1e-8)

        return {self.feature: output}


class MPNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.node_type = config["node_type"]
        self.feature = config["features"]
        self.etype = config["edge_type"]
        self.init_network = MLP(
            input_size=config["input_size"],
            output_size=config["hidden_size"],
            hidden_layers=config["init_layers"],
        )

        self.hidden_size = config["hidden_size"]

        self.n_iter = config["n_iter"]
        self.node_update_networks = nn.ModuleList()
        for _ in range(self.n_iter):
            self.node_update_networks.append(
                NodeNetwork(
                    self.hidden_size,
                    self.hidden_size,
                    config["mpnn_layers"],
                    feature=self.feature,
                )
            )

    def update_global_rep(self, g):
        global_rep = dgl.sum_nodes(g, self.feature, ntype=self.node_type)
        global_rep = global_rep / (
            torch.norm(global_rep, p="fro", dim=1, keepdim=True) + 1e-8
        )

        g.nodes[self.node_type].data["global_features"] = dgl.broadcast_nodes(
            g, global_rep, ntype=self.node_type
        )
        g.nodes["global_node"].data["global_features"] = global_rep

    def forward(self, g, **kwargs):

        g.nodes[self.node_type].data[self.feature] = self.init_network(
            g.nodes[self.node_type].data[self.feature]
        )
        self.update_global_rep(g)

        for iter_i in range(self.n_iter):
            g.update_all(
                # fn.copy_src(self.feature, "message"), # for dgl version 0.9.1
                fn.copy_u(self.feature, "message"),  # for dgl version > 1.0
                self.node_update_networks[iter_i],
                etype=self.etype,
            )
            self.update_global_rep(g)
        return g.nodes[self.node_type].data[self.feature]
