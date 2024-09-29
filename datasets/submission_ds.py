import dgl
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.XMLHandler import XMLHandler


def collate_graphs(samples):
    if isinstance(samples[0], dgl.DGLGraph):
        batched_g = dgl.batch(samples)
        return batched_g
    batched_g = dgl.batch(samples[0])
    return batched_g


class CaloChallengeDataset(Dataset):
    def __init__(
        self,
        energies,
        config=None,
        reduce_ds=-1,
        entry_start=0,
        ds_type="photon",
    ) -> None:
        super().__init__()
        self.config = config
        self.ds_type = ds_type
        self.n_events = len(energies)
        self.entry_start = entry_start
        if reduce_ds > 1.0:
            self.n_events = reduce_ds

        # graph creation
        self.graph_type = config["graph_type"]
        self.graph_k = config["graph_k"]
        self.graph_radius = config["graph_radius"]
        self.graph_ext = config.get("graph_ext", False)

        # norm configs
        self.max_deposit = float(config.get("max_deposit", 6.5))
        self.alpha = 1e-6
        self.e_min = 0.2560
        self.e_max = 4194.3042
        self.log_e = config.get("log_e", False)

        self.min_max = config.get("min_max", False)
        self.log_coord = config.get("log_coord", False)

        self.energies = (
            torch.tensor(
                energies[self.entry_start : self.entry_start + self.n_events]
            ).float()
            / 1000
        )

        self.init_vars()
        self.init_edges()

        self.pred_type = self.config.get("pred_type", "cell")
        self.log_scale = self.config["log_scale"]
        self.logit = self.config["logit"]
        self.layer_norm = self.config.get("layer_norm", False)

        self.var_transform = self.config["var transform"]

    def init_vars(self):
        self.xml_handler = XMLHandler(
            self.ds_type, f"configs/binning_dataset_1_{self.ds_type}s.xml"
        )
        self.eta = torch.tensor(np.concatenate(self.xml_handler.eta_all_layers)).float()
        self.phi = torch.tensor(np.concatenate(self.xml_handler.phi_all_layers)).float()
        self.n_cells = len(self.eta)
        self.layers = torch.concat(
            [
                torch.ones(len(el)) * i
                for i, el in enumerate(self.xml_handler.eta_all_layers)
                if len(el) > 0
            ]
        )
        self.n_layers = self.config["n_layers"]
        self.mask = self.layers < self.n_layers
        self.r = (self.eta**2 + self.phi**2) ** 0.5
        self.theta = torch.atan2(self.phi, self.eta)
        self.z = self.layers
        self.n_a_bins = np.array(self.xml_handler.a_bins)[
            np.array(self.xml_handler.relevantlayers)
        ]
        self.n_r_bins = np.array(self.xml_handler.r_bins)[
            np.array(self.xml_handler.relevantlayers)
        ]
        self.n_bins = self.n_a_bins * self.n_r_bins
        self.layer_cells = torch.arange(self.n_cells).split(self.n_bins.tolist())

    def get_layer_angular_edges(self, layer):
        angular_edges_start = self.layer_cells[layer]
        angular_edges_end = (
            torch.stack(self.layer_cells[layer].split(self.n_r_bins[layer].tolist()))
            .roll(-1, dims=0)
            .flatten()
        )
        radial_edges_start = self.layer_cells[layer][: -self.n_r_bins[layer]]
        radial_edges_end = (
            torch.stack(self.layer_cells[layer].split(self.n_r_bins[layer].tolist()))[
                :-1
            ]
            .roll(-1, dims=1)
            .flatten()
        )

        one_layer_edges_start = torch.cat([angular_edges_start, radial_edges_start])
        one_layer_edges_end = torch.cat([angular_edges_end, radial_edges_end])
        return one_layer_edges_start, one_layer_edges_end
        # return angular_edges_start, angular_edges_end

    def connect_layers_ext(self):
        edges_start = []
        edges_end = []
        max_layer = len(self.xml_handler.relevantlayers) - 1
        for layer in range(max_layer - 1):
            if self.n_a_bins[layer] != 1:
                tmp_start = self.layer_cells[layer][
                    self.n_r_bins[layer] :: self.n_r_bins[layer]
                ]
                flag = False
                for i in range(layer + 1, max_layer):
                    if self.n_a_bins[i] != 1:
                        tmp_end = self.layer_cells[i][
                            self.n_r_bins[i] :: self.n_r_bins[i]
                        ][: len(tmp_start)]
                        flag = True
                        break
                if flag:
                    edges_start.append(tmp_start)
                    edges_end.append(tmp_end)
                tmp_start = self.layer_cells[layer].reshape(self.n_a_bins[layer], -1)[
                    :, -1
                ]
                flag = False
                for i in range(layer + 1, max_layer):
                    if self.n_a_bins[i] != 1:
                        tmp_end = self.layer_cells[i].reshape(self.n_a_bins[i], -1)[
                            :, -1
                        ][: len(tmp_start)]
                        flag = True
                        break
                if flag:
                    edges_start.append(tmp_start)
                    edges_end.append(tmp_end)

                layer_bins = self.layer_cells[layer].reshape(self.n_a_bins[layer], -1)
                tmp_start = layer_bins[:, -layer_bins.shape[1] // 2]
                flag = False
                for i in range(layer + 1, max_layer):
                    if self.n_a_bins[i] != 1:
                        layer_bins = self.layer_cells[i].reshape(self.n_a_bins[i], -1)
                        tmp_end = layer_bins[:, -layer_bins.shape[1] // 2][: len(tmp_start)]
                        flag = True
                        break
                if flag:
                    edges_start.append(tmp_start)
                    edges_end.append(tmp_end)
            if self.n_a_bins[layer] == 1:
                tmp_start = self.layer_cells[layer][0]
                flag = False
                for i in range(layer + 1, max_layer):
                    if self.n_a_bins[i] != 1:
                        tmp_end = self.layer_cells[i][
                            self.n_r_bins[i] :: self.n_r_bins[i]
                        ]
                        flag = True
                        break
                if flag:
                    tmp_start = torch.repeat_interleave(tmp_start, len(tmp_end))
                    edges_start.append(tmp_start)
                    edges_end.append(tmp_end)
            if self.n_a_bins[layer] == 1:
                tmp_start = self.layer_cells[layer][-1]
                flag = False
                for i in range(layer + 1, max_layer + 1):
                    if self.n_a_bins[i] == 1:
                        tmp_end = self.layer_cells[i][-1]
                        flag = True
                        break
                if flag:
                    edges_start.append(tmp_start.unsqueeze(0))
                    edges_end.append(tmp_end.unsqueeze(0))

        if self.ds_type == 'pion':
            tmp_start = self.layer_cells[max_layer - 1][
                self.n_r_bins[max_layer - 1] :: self.n_r_bins[max_layer - 1]
            ]
            tmp_end = torch.repeat_interleave(
                self.layer_cells[max_layer][0], len(tmp_start)
            )
            edges_start.append(tmp_start)
            edges_end.append(tmp_end)

        if self.ds_type == 'photon':
            tmp_start = torch.stack([
                self.layer_cells[max_layer][-1],
                self.layer_cells[max_layer][0],
            ])
            tmp_end = torch.stack([
                self.layer_cells[max_layer-1][-1],
                self.layer_cells[max_layer-1][0],

            ])
            edges_start.append(tmp_start)
            edges_end.append(tmp_end)
            tmp_start = self.layer_cells[max_layer - 2][
                self.n_r_bins[max_layer - 2] :: self.n_r_bins[max_layer - 2]
            ]
            tmp_end = torch.repeat_interleave(
                self.layer_cells[max_layer-1][0], len(tmp_start)
            )
            edges_start.append(tmp_start)
            edges_end.append(tmp_end)

            tmp_start = self.layer_cells[max_layer - 2][
                self.n_r_bins[max_layer - 2] :: self.n_r_bins[max_layer - 2]
            ]
            tmp_end = torch.repeat_interleave(
                self.layer_cells[max_layer][0], len(tmp_start)
            )
            edges_start.append(tmp_start)
            edges_end.append(tmp_end)
        return edges_start, edges_end

    def connect_layers(self):
        pass

    def init_edges(self):
        if self.graph_type == "manual":
            if self.graph_ext:
                edges_start, edges_end = self.connect_layers_ext()
            else:
                edges_start, edges_end = self.connect_layers()
            for layer in range(len(self.xml_handler.relevantlayers)):
                if self.n_a_bins[layer] != 1:
                    (
                        one_layer_edges_start,
                        one_layer_edges_end,
                    ) = self.get_layer_angular_edges(layer)
                    edges_start.append(one_layer_edges_start)
                    edges_end.append(one_layer_edges_end)
                if self.n_a_bins[layer] == 1:
                    one_layer_edges_start = self.layer_cells[layer][:-1]
                    one_layer_edges_end = self.layer_cells[layer][1:]
                    edges_start.append(one_layer_edges_start)
                    edges_end.append(one_layer_edges_end)
            self.cell_to_cell_start = torch.cat(edges_start)
            self.cell_to_cell_end = torch.cat(edges_end)
            tmp_start = torch.cat(
                [self.cell_to_cell_start, self.cell_to_cell_end],
                dim=0,
            )
            tmp_end = torch.cat(
                [self.cell_to_cell_end, self.cell_to_cell_start],
                dim=0,
            )
            self.cell_to_cell_start, self.cell_to_cell_end = tmp_start, tmp_end
        else:
            raise "Only manual graph creation is supported"

    def __len__(self) -> int:
        return self.n_events

    def __getitem__(self, index: int):

        n_cells = self.mask.sum()
        num_nodes_dict = {"cells": n_cells, "global_node": 1}
        graph_dict = {
            ("cells", "cell_to_cell", "cells"): (
                self.cell_to_cell_start,
                self.cell_to_cell_end,
            ),
            ("cells", "cells_to_global", "global_node"): (
                torch.arange(n_cells, dtype=torch.long),
                torch.zeros(n_cells, dtype=torch.long),
            ),
            ("global_node", "global_to_cells", "cells"): (
                torch.zeros(n_cells, dtype=torch.long),
                torch.arange(n_cells, dtype=torch.long),
            ),
        }
        g = dgl.heterograph(graph_dict, num_nodes_dict)

        if self.min_max:
            g.nodes["cells"].data["eta"] = (self.eta[self.mask] - self.eta.min()) / (
                self.eta.max() - self.eta.min()
            )
            g.nodes["cells"].data["phi"] = (self.phi[self.mask] - self.phi.min()) / (
                self.phi.max() - self.phi.min()
            )
        else:
            g.nodes["cells"].data["eta"] = (
                self.eta[self.mask] - self.eta[self.mask].mean()
            ) / self.eta[self.mask].std()
            g.nodes["cells"].data["phi"] = (
                self.phi[self.mask] - self.phi[self.mask].mean()
            ) / self.phi[self.mask].std()

        g.nodes["cells"].data["features_0"] = torch.cat(
            [
                g.nodes["cells"].data["eta"].unsqueeze(1),
                g.nodes["cells"].data["phi"].unsqueeze(1),
            ],
            dim=1,
        )

        g.nodes["cells"].data["layer"] = self.layers[self.mask].long()
        g.nodes["global_node"].data["energy"] = self.energies[index].unsqueeze(-1)
        g.nodes["cells"].data["energy"] = torch.zeros((n_cells, 1))
        if self.logit:
            e = g.nodes["cells"].data["energy"] / (
                self.max_deposit * g.nodes["global_node"].data["energy"].squeeze()
            )
            e = self.alpha + (1 - 2 * self.alpha) * e
            e = torch.log(e / (1 - e)).nan_to_num(0)
        g.nodes["cells"].data["energy"] = (
            e - self.var_transform["energy_logit_mean"]
        ) / self.var_transform["energy_logit_std"]
        if self.log_e:
            g.nodes["global_node"].data["energy"] = torch.log10(
                g.nodes["global_node"].data["energy"] / self.e_min
            ) / torch.log10(self.e_max / self.e_min)
        else:
            g.nodes["global_node"].data["energy"] = (
                g.nodes["global_node"].data["energy"] - self.e_min
            ) / (self.e_max - self.e_min)
        return g
