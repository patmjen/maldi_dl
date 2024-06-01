import torch
import torch.nn as nn

from model.init import init_weights_xavier, init_weights_truncnormal


class MLP(nn.Sequential):
    def __init__(self, hparams):
        super().__init__()
        assert hparams.model.classname == 'mlp'
        features = [hparams.data.spectrum_len] + hparams.model.hidden_channels

        if hparams.model.activation == 'ReLU':
            Act = nn.ReLU
        elif hparams.model.activation == 'PReLU':
            Act = nn.PReLU
        else:
            raise ValueError(f'Uknown activation: {hparams.model.activation}')

        if hparams.model.norm == 'Batch':
            Norm = nn.BatchNorm1d
        elif hparams.model.norm == 'Layer':
            Norm = nn.LayerNorm
        else:
            raise ValueError(f'Uknown normalization: {hparams.model.norm}')

        # Add layers
        self.append(nn.Flatten())  # Combine channel and spatial dim
        for f_in, f_out in zip(features[:-1], features[1:]):
            self.append(nn.Linear(f_in, f_out))
            self.append(Norm(f_out))
            self.append(Act())
        self.append(nn.Linear(features[-1], len(hparams.data.classes)))

        if hparams.model.weight_init == 'xavier':
            self.apply(init_weights_xavier)
        elif hparams.model.weight_init == 'trunc_normal':
            self.apply(init_weights_truncnormal)