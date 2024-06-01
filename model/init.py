import torch.nn as nn


def init_weights_xavier(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def init_weights_truncnormal(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=1e-6, a=-1.0, b=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    if isinstance(m, nn.Linear):
        torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=1e-6, a=-1.0, b=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)