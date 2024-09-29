import json
import os
import h5py

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.submission_ds import CaloChallengeDataset, collate_graphs
from models.ddim_diffusion import DDIMDiffusionModel


def parse_args():
    """
    Argument parser for evaluation script
    """
    parser = argparse.ArgumentParser(description="Evaluate CaloGraph model")

    parser.add_argument(
        "-d",
        "--dataset",
        choices=["1-photons", "1-pions"],
        help="Which dataset is evaluated",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default=".", help="Folder for generated files"
    )
    parser.add_argument(
        "-n",
        "--n_events",
        type=int,
        default=-1,
        help="""
        How many samples to generate, bounded by the number of incident energies 
        (-1 to generate all of them)
        """,
    )
    parser.add_argument("-bs", "--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--no_cuda", default=False, type=bool, help="Do not use cuda.")
    parser.add_argument(
        "--which_cuda", default=0, type=int, help="Which cuda device to use"
    )
    return parser.parse_args()


def main():
    l_e_tr = []
    l_cell_e_pred = []

    args = parse_args()
    print(torch.cuda.is_available())
    args.device = torch.device(
        "cuda:" + str(args.which_cuda)
        if torch.cuda.is_available() and not args.no_cuda
        else "cpu"
    )
    print(f"Using {args.device}")

    if args.dataset == "1-pions":
        config_path = "configs/ddim_pions.json"
        ckpt_path = "saved_models/ddim_pions.ckpt"
        n_cells = 533
        num_events = {
            256: 10000,
            512: 10000,
            1024: 10000,
            2048: 10000,
            4096: 10000,
            8192: 10000,
            16384: 10000,
            32768: 10000,
            65536: 10000,
            131072: 10000,
            262144: 9800,
            524288: 5000,
            1048576: 3000,
            2097152: 2000,
            4194304: 1000,
        }

    elif args.dataset == "1-photons":
        config_path = "configs/ddim_photons.json"
        ckpt_path = "saved_models/ddim_photons.ckpt"
        n_cells = 368
        num_events = {
            256: 10000,
            512: 10000,
            1024: 10000,
            2048: 10000,
            4096: 10000,
            8192: 10000,
            16384: 10000,
            32768: 10000,
            65536: 10000,
            131072: 10000,
            262144: 10000,
            524288: 5000,
            1048576: 3000,
            2097152: 2000,
            4194304: 1000,
        }

    incident_energies = np.concatenate(
        [key * np.ones(value) for key, value in num_events.items()]
    )
    shuffled_idx = np.arange(len(incident_energies))
    rng = np.random.default_rng(42)
    rng.shuffle(shuffled_idx)
    incident_energies = incident_energies[shuffled_idx].reshape(-1, 1)
    with open(config_path, "r") as f:
        config = json.load(f)

    model = DDIMDiffusionModel(config=config)
    state_dict = torch.load(ckpt_path)["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[key[4:]] = value
    model.load_state_dict(new_state_dict)
    model.eval()

    ds = CaloChallengeDataset(
        energies=incident_energies,
        config=config,
        reduce_ds=args.n_events,
        entry_start=0,
        ds_type=args.dataset[2:-1],
    )
    loader = DataLoader(
        ds,
        num_workers=4,
        batch_size=args.batch_size,
        collate_fn=collate_graphs,
        pin_memory=False,
    )

    model.eval()
    model.to(args.device)

    for g in tqdm(loader):
        g = g.to(args.device)
        g, _ = model.generate_samples(g, save_seq=False, num_steps=50)
        e_pred = g.nodes["cells"].data["energy_corrupted"].reshape(-1, n_cells)
        e_pred = unnormalize(e_pred.cpu().numpy(), config=config, g=g, ds=ds)
        particle_e = g.nodes["global_node"].data["energy"].cpu().numpy()
        if config.get("log_e", False):
            particle_e = ds.e_min * (ds.e_max / ds.e_min) ** particle_e
        else:
            particle_e = ds.e_min + (ds.e_max - ds.e_min) * particle_e
        l_cell_e_pred.append(e_pred)
        l_e_tr.append(particle_e)

    truth_e = np.concatenate(l_e_tr).astype(np.float32) * 1000
    pred_cell_e = np.concatenate(l_cell_e_pred).astype(np.float32) * 1000

    with h5py.File(
        os.path.join(args.output_dir, f"generated_{args.dataset[2:]}.h5"), "w"
    ) as dataset_file:
        dataset_file.create_dataset(
            "incident_energies",
            data=truth_e.reshape(len(truth_e), -1),
            compression="gzip",
        )
        dataset_file.create_dataset(
            "showers",
            data=pred_cell_e.reshape(len(pred_cell_e), -1),
            compression="gzip",
        )


def unnormalize(x, config, logit=False, g=None, ds=None):
    if g is not None:
        energy = g.nodes["global_node"].data["energy"].cpu().numpy()
        if config.get("log_e", False):
            energy = ds.e_min * (ds.e_max / ds.e_min) ** energy
        else:
            energy = ds.e_min + (ds.e_max - ds.e_min) * energy

    def _unnormalize(x):
        return (
            x * config["var transform"]["energy_logit_std"]
            + config["var transform"]["energy_logit_mean"]
        )

    def _unscale(x):
        exp = np.exp(x)
        y = exp / (1 + exp)
        y = (y - ds.alpha) / (1 - 2 * ds.alpha)
        y[y < 0] = 0
        return y * ds.max_deposit * energy

    return _unscale(_unnormalize(x))


if __name__ == "__main__":
    main()
