from copy import deepcopy

from lightning_fabric.utilities.seed import seed_everything
import torch.nn as nn
from pytorch_lightning.utilities.seed import isolate_rng

import model  # Do not import classname_to_class here to avoid circular import


class Ensemble(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        RESERVED_KEYS = ['classname', 'name', 'combine_method', 'seed']
        assert hparams.model.classname == 'ensemble'
        assert hparams.model.combine_method in ['sum', 'mean']

        self.combine_method = hparams.model.combine_method
        self.models = nn.ModuleDict()
        for name, modelparams in hparams.model.items():
            if name in RESERVED_KEYS:
                continue

            # Need to construct new hparams where the sub-module is the
            # top-level model, so it can inspect all hyperparameters as normal.
            hparams_model = deepcopy(hparams)
            hparams_model.model = modelparams

            with isolate_rng():
                if 'seed' in modelparams and modelparams.seed is not None:
                    seed_everything(modelparams.seed)
                Model = model.classname_to_class(modelparams.classname)
                self.models[name] = Model(hparams_model)


    def forward(self, x):
        logits = 0
        for submodel in self.models.values():
            logits = logits + submodel(x)
        if self.combine_method == 'mean':
            logits = logits / len(self.models)
        return logits