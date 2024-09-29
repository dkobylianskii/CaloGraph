import dgl
import torch
import torch.nn as nn

from models.mlp import MLP
from models.mpnn import MPNN


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, embd_size, max_positions=200, endpoint=False):
        super().__init__()
        self.embd_size = embd_size
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.embd_size // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.embd_size // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.outer(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class DDIMDenoisingModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.dataset = self.config.get("dataset", "cocoa")
        self.use_attention = self.config["use_attention"]
        self.context = self.config.get("context", False)
        self.init_config = self.config["init_model"]
        self.gnn_config = self.config["gnn_model"]
        self.output_config = self.config["output_model"]

        self.particle_encoder = MLP(
            input_size=self.init_config["particle_encoder"]["input_size"],
            output_size=self.init_config["particle_encoder"]["output_size"],
            hidden_layers=self.init_config["particle_encoder"]["layers"],
        )
        self.cell_encoder = MLP(
            input_size=self.init_config["cell_encoder"]["input_size"]
            + self.init_config["layer_embedding"],
            output_size=self.init_config["cell_encoder"]["output_size"],
            hidden_layers=self.init_config["cell_encoder"]["layers"],
        )

        self.layer_embedding = nn.Embedding(45, self.init_config["layer_embedding"])
        # self.cell_encoder = MLP(31, 128, [128, 128, 128])

        self.input_dim = (
            self.init_config["particle_encoder"]["output_size"]  # particle features
            + self.init_config["cell_encoder"]["output_size"]  # corrupted energy
            + self.config["time_embedding_size"]  # global time embedding
        )
        print("input dim", self.input_dim)
        self.ctx_size = (
            self.init_config["particle_encoder"]["output_size"]
            + self.config["time_embedding_size"]
        )
        self.init_network = MLP(
            input_size=self.input_dim,
            output_size=self.init_config["output_size"],
            hidden_layers=self.init_config["layers"],
        )
        self.message_net = MPNN(self.gnn_config)

        self.noise_net = MLP(
            input_size=self.gnn_config["output_size"] + self.input_dim,
            output_size=1,
            hidden_layers=self.output_config["noise_prediction_layers"],
        )

        self.time_embd_model = PositionalEmbedding(
            embd_size=self.config["time_embedding_size"],
            max_positions=self.config.get("max_positions", 200),
        )

    def forward(self, g, t):

        time_embd = self.time_embd_model(t)

        self.init_feat(g)
        global_time_embd = dgl.broadcast_nodes(g, time_embd, ntype="cells")
        condition = self.particle_encoder(g.nodes["global_node"].data["features"])
        global_condition = dgl.broadcast_nodes(
            g,
            condition,
            ntype="cells",
        )

        g.nodes["global_node"].data["context"] = torch.cat(
            [
                time_embd,
                condition,
            ],
            dim=1,
        )

        cell_embedding = self.cell_encoder(
            torch.cat(
                [
                    g.nodes["cells"].data["energy_corrupted"],
                    g.nodes["cells"].data["node_features"],
                ],
                dim=1,
            )
        )

        g.nodes["cells"].data["features"] = self.init_network(
            torch.cat(
                [
                    cell_embedding,
                    global_time_embd,
                    global_condition,
                ],
                dim=1,
            )
        )
        features = self.message_net(g)

        g.nodes["cells"].data["updated_features"] = torch.cat(
            [
                features,  # ndata
                cell_embedding,
                global_time_embd,
                global_condition,
            ],
            dim=1,
        )

        F_x = self.noise_net(g.nodes["cells"].data["updated_features"])
        g.nodes["cells"].data["noise_pred"] = F_x

        return g

    def init_feat(self, g):

        g.nodes["global_node"].data["features_0"] = g.nodes["global_node"].data[
            "energy"
        ]
        g.nodes["cells"].data["node_features"] = torch.cat(
            [
                g.nodes["cells"].data["features_0"],
                self.layer_embedding(g.nodes["cells"].data["layer"]),
            ],
            dim=1,
        )

        g.nodes["global_node"].data["features"] = g.nodes["global_node"].data[
            "features_0"
        ]

        return g
