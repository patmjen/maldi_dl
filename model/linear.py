import torch.nn as nn

from model.init import init_weights_xavier, init_weights_truncnormal

class LinearModel(nn.Sequential):
    def __init__(self, hparams):
        super().__init__()
        assert hparams.model.classname == 'linear'
        self.append(nn.Flatten())
        self.append(
            nn.Linear(hparams.data.spectrum_len, len(hparams.data.classes))
        )

        if hparams.model.weight_init == 'xavier':
            self.apply(init_weights_xavier)
        elif hparams.model.weight_init == 'trunc_normal':
            self.apply(init_weights_truncnormal)