import dgl
import math
import torch
import torch.nn as nn
from tqdm import tqdm

from models.ddim_denoising import DDIMDenoisingModel


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.tensor(
        [[i / 10_000 ** (2 * j / d) for j in range(d)] for i in range(n)]
    )
    sin_mask = torch.arange(0, n, 2)

    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])
    time_embed = nn.Embedding(n, d)
    time_embed.weight.data = embedding
    time_embed.requires_grad_(False)
    return time_embed


class VPDiffusionSchedule:
    def __init__(self, max_sr: float = 1, min_sr: float = 1e-2) -> None:
        self.max_sr = max_sr
        self.min_sr = min_sr

    def __call__(self, time: torch.Tensor) -> torch.Tensor:
        return self.cosine_diffusion_shedule(time, self.max_sr, self.min_sr)

    def get_betas(self, time: torch.Tensor) -> torch.Tensor:
        return self.cosine_beta_shedule(time, self.max_sr, self.min_sr)

    def cosine_beta_shedule(
        self, diff_time: torch.Tensor, max_sr: float = 1, min_sr: float = 1e-2
    ) -> torch.Tensor:
        """Returns the beta values for the continuous flows using the above cosine
        scheduler."""
        start_angle = math.acos(max_sr)
        end_angle = math.acos(min_sr)
        diffusion_angles = start_angle + diff_time * (end_angle - start_angle)
        return 2 * (end_angle - start_angle) * torch.tan(diffusion_angles)

    def cosine_diffusion_shedule(
        self, diff_time: torch.Tensor, max_sr: float = 1, min_sr: float = 1e-2
    ):
        """Calculates the signal and noise rate for any point in the diffusion
        processes.

        Using continuous diffusion times between 0 and 1 which make switching between
        different numbers of diffusion steps between training and testing much easier.
        Returns only the values needed for the jump forward diffusion step and the reverse
        DDIM step.
        These are sqrt(alpha_bar) and sqrt(1-alphabar) which are called the signal_rate
        and noise_rate respectively.

        The jump forward diffusion process is simply a weighted sum of:
            input * signal_rate + eps * noise_rate

        Uses a cosine annealing schedule as proposed in
        Proposed in https://arxiv.org/abs/2102.09672

        Args:
            diff_time: The time used to sample the diffusion scheduler
                Output will match the shape
                Must be between 0 and 1
            max_sr: The initial rate at the first step
            min_sr: How much signal is preserved at end of diffusion
                (can't be zero due to log)
        """

        # Use cosine annealing, which requires switching from times -> angles
        start_angle = math.acos(max_sr)
        end_angle = math.acos(min_sr)
        diffusion_angles = start_angle + diff_time * (end_angle - start_angle)
        signal_rates = torch.cos(diffusion_angles)
        noise_rates = torch.sin(diffusion_angles)
        return signal_rates, noise_rates


class DDIMDiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.timesteps = self.config["timesteps"]
        if config["loss_type"] == "mse":
            self.loss = torch.nn.MSELoss(reduce=None)
        else:
            KeyError("Loss type not implemented: {}".format(config["loss_type"]))

        self.num_steps = self.config["timesteps"]
        self.pred_type = self.config["pred_type"]
        print(config["model_type"])
        self.diff_sched = VPDiffusionSchedule()
        self.net = DDIMDenoisingModel(self.config)
        self.self_ctx = self.config.get("self_ctx", False)

    def get_loss(self, g):
        t = torch.rand(g.batch_size, device=g.device)
        signal_rates, noise_rates = self.diff_sched(
            dgl.broadcast_nodes(g, t, ntype="cells").unsqueeze(1)
        )
        n = torch.randn_like(g.nodes["cells"].data["energy"])

        x = signal_rates * g.nodes["cells"].data["energy"] + noise_rates * n
        n_pred = torch.zeros_like(n, device=g.device).detach()
        if self.self_ctx:
            if torch.rand(1) > 0.5:
                g.nodes["cells"].data["energy_corrupted"] = torch.cat([x, n_pred], -1)
                g = self.net(g, t)
                n_pred = g.nodes["cells"].data["noise_pred"].detach()
            g.nodes["cells"].data["energy_corrupted"] = torch.cat([x, n_pred], -1)
        else:
            g.nodes["cells"].data["energy_corrupted"] = x
        g = self.net(g, t)

        loss = (self.loss(g.nodes["cells"].data["noise_pred"], n)).mean()
        return loss

    def generate_samples(
        self, g, save_seq=False, num_steps=None, sampler=None, **kwargs
    ):
        if num_steps is not None:
            self.num_steps = num_steps
        if sampler is not None:
            self.sampler = sampler
        # return self.euler_maruyama_sampler(g, num_steps=self.num_steps, keep_all=save_seq)
        return self.pndm_sampler(
            g, num_steps=self.num_steps, dt=0.5 / self.num_steps, **kwargs
        )

    @torch.no_grad()
    def transfer(self, x, t, t_next, et, dt=0.01):
        at, _ = self.diff_sched(t + dt)
        at = at**2
        at_next, _ = self.diff_sched(t_next + dt)
        at_next = at_next**2
        x_delta = (at_next - at) * (
            (1 / (at.sqrt() * (at.sqrt() + at_next.sqrt()))) * x
            - 1
            / (at.sqrt() * (((1 - at_next) * at).sqrt() + ((1 - at) * at_next).sqrt()))
            * et
        )
        x_next = x + x_delta
        return x_next

    @torch.no_grad()
    def runge_kutta(self, g, t_list, model, ets, dt=0.01, prev_noise=None):
        x = g.nodes["cells"].data["energy_corrupted"]

        if prev_noise is not None:
            g.nodes["cells"].data["energy_corrupted"] = torch.cat([x, prev_noise], -1)
        g = model(g, t_list[0].repeat(g.batch_size))
        e_1 = g.nodes["cells"].data["noise_pred"]
        ets.append(e_1)
        x_2 = self.transfer(x, t_list[0], t_list[1], e_1, dt=dt)
        g.nodes["cells"].data["energy_corrupted"] = x_2
        if prev_noise is not None:
            g.nodes["cells"].data["energy_corrupted"] = torch.cat([x_2, e_1], -1)

        g = model(g, t_list[1].repeat(g.batch_size))
        e_2 = g.nodes["cells"].data["noise_pred"]
        x_3 = self.transfer(x, t_list[0], t_list[1], e_2, dt=dt)
        g.nodes["cells"].data["energy_corrupted"] = x_3
        if prev_noise is not None:
            g.nodes["cells"].data["energy_corrupted"] = torch.cat([x_3, e_2], -1)

        g = model(g, t_list[1].repeat(g.batch_size))
        e_3 = g.nodes["cells"].data["noise_pred"]
        x_4 = self.transfer(x, t_list[0], t_list[2], e_3, dt=dt)
        g.nodes["cells"].data["energy_corrupted"] = x_4
        if prev_noise is not None:
            g.nodes["cells"].data["energy_corrupted"] = torch.cat([x_4, e_3], -1)

        g = model(g, t_list[2].repeat(g.batch_size))
        e_4 = g.nodes["cells"].data["noise_pred"]

        et = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)
        g.nodes["cells"].data["energy_corrupted"] = x

        return et

    @torch.no_grad()
    def gen_order_4(self, g, t, t_next, model, ets, dt=0.01, prev_noise=None):
        t_list = [t, (t + t_next) / 2, t_next]
        x = g.nodes["cells"].data["energy_corrupted"]
        if len(ets) > 2:
            if prev_noise is not None:
                # print(prev_noise.shape, g.nodes["cells"].data["energy_corrupted"].shape)
                g.nodes["cells"].data["energy_corrupted"] = torch.cat(
                    [x, prev_noise], -1
                )
            g = model(g, t.repeat(g.batch_size))
            noise_ = g.nodes["cells"].data["noise_pred"]
            ets.append(noise_)
            noise = (1 / 24) * (
                55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4]
            )
        else:
            noise = self.runge_kutta(
                g, t_list, model, ets, dt=dt, prev_noise=prev_noise
            )
        if prev_noise is not None:
            prev_noise = noise
        x_next = self.transfer(x, t, t_next, noise, dt=dt)
        return x_next, prev_noise

    def pndm_sampler(self, g, num_steps, dt=0.01, shift=0, **kwargs):
        initial_noise = torch.randn((g.num_nodes("cells"), 1), device=g.device)
        ets = []
        g.nodes["cells"].data["energy_corrupted"] = initial_noise
        seq = range(0, num_steps - shift)
        seq_next = [-1] + list(seq[:-1])
        if self.self_ctx:
            prev_noise = torch.zeros_like(initial_noise, device=g.device)
        else:
            prev_noise = None
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), total=num_steps):
            t = torch.tensor([i], device=initial_noise.device) / num_steps
            t_next = torch.tensor([j], device=initial_noise.device) / num_steps

            x_next, prev_noise = self.gen_order_4(
                g, t, t_next, self.net, ets=ets, dt=dt, prev_noise=prev_noise
            )
            # print(prev_noise[0][0])
            g.nodes["cells"].data["energy_corrupted"] = x_next
        return g, []
