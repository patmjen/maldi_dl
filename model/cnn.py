import torch
import torch.nn as nn


def conv_output_len(input_len, kernel_size, padding, stride):
    return int((input_len + 2 * padding - kernel_size) / stride + 1)


def pool_output_len(input_len, kernel_size, padding=0, stride=None):
    if stride is None:
        stride = kernel_size
    return int((input_len + 2 * padding - (kernel_size - 1) - 1) / stride + 1)


class Cnn(nn.Sequential):
    def __init__(self, hparams):
        super().__init__()
        assert hparams.model.classname == 'cnn'
        channels = [hparams.model.in_channels] + hparams.model.hidden_channels
        conv_params = zip(
            channels[:-1],
            channels[1:],
            hparams.model.kernel_sizes,
            hparams.model.strides,
        )

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

        # Add convolutional layers
        seq_len = hparams.data.spectrum_len
        for c_in, c_out, k, s in conv_params:
            self.append(nn.Conv1d(c_in, c_out, kernel_size=k, stride=s))
            self.append(Norm(c_out))
            self.append(Act())
            seq_len = conv_output_len(seq_len, k, 0, s)

        # Add final linear layer
        self.append(nn.Flatten())  # Combine channel and spatial dim
        self.append(nn.Linear(seq_len * channels[-1],
                              len(hparams.data.classes)))


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        assert kernel_size % 2 == 1, \
            f'kernel_size must be odd but was: {kernel_size}'
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        if in_channels != out_channels or stride != 1:
            self.project = nn.Conv1d(in_channels, out_channels, 1, stride)
        else:
            self.project = nn.Identity()


    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)

        out = out + self.project(x)

        return out


class ResNet(nn.Sequential):
    def __init__(self, hparams):
        super().__init__()
        assert hparams.model.classname == 'resnet'
        seq_len = hparams.data.spectrum_len
        seq_pad_to = hparams.model.get('assume_padded_to', None)
        if seq_pad_to is not None:
            extra =  ((seq_len % seq_pad_to) != 0) * seq_pad_to
            seq_len = (seq_len // seq_pad_to) * seq_pad_to + extra

        if hparams.model.init_pool_size > 0:
            self.append(nn.MaxPool1d(kernel_size=4))
            seq_len = pool_output_len(seq_len, 4)

        channels = [hparams.model.in_channels] + hparams.model.hidden_channels
        res_block_params = zip(
            channels[:-1],
            channels[1:],
            hparams.model.kernel_sizes,
            hparams.model.strides,
        )

        # Add convolutional layers
        for c_in, c_out, k, s in res_block_params:
            self.append(ResBlock(c_in, c_out, kernel_size=k, stride=s))
            self.append(nn.ReLU())
            seq_len = conv_output_len(seq_len, k, k // 2, s)

        # Add final linear layer
        self.append(nn.Flatten())  # Combine channel and spatial dim
        self.append(nn.Linear(seq_len * channels[-1],
                              len(hparams.data.classes)))
