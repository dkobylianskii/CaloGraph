import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_layers,
        activation=nn.GELU,
        final_activation=False,
        norm_layer=None,
        norm_final_layer=False,
        dropout=0.0,
        ctx_size=0,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.ctx_size = ctx_size

        nodelist = [input_size + ctx_size, *hidden_layers, output_size]

        num_layers = len(nodelist) - 1
        layers = []

        for i in range(num_layers):
            is_final_layer = i == num_layers - 1

            if norm_layer and (norm_final_layer or not is_final_layer):
                layers.append(norm_layer(nodelist[i], elementwise_affine=False))

            if dropout and (norm_final_layer or not is_final_layer):
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(nodelist[i], nodelist[i + 1]))

            if activation and (final_activation or not is_final_layer):
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x, ctx=None):
        if ctx is not None:
            x = torch.cat([x, ctx], dim=-1)
        return self.net(x)
