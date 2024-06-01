import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.stats import truncnorm
import torch
import torch.nn.functional as F


########################
# Spectrum Augmentations
########################


def pad_to_divisible(spectra, labels, probes, d=256, value=0.0):
    l = spectra.shape[-1]
    if l % d != 0:
        p = d - (l % d)
        l += p
        spectra = F.pad(spectra, (0, p), value=value)

    return spectra, labels, probes


class PadToDivisible:
    def __init__(self, d=256, value=0.0):
        self.d = d
        self.value = value


    def __call__(self, spectra, labels, probes):
        return pad_to_divisible(spectra, labels, probes, self.d, self.value)


def scaled_gaussian_noise(spectra, labels, probes, std_factor=0.01):
    spectrum_max = spectra.max(dim=-1, keepdim=True).values
    spectra_n = spectra + torch.randn_like(spectra) * std_factor * spectrum_max
    return spectra_n, labels, probes


class ScaledGaussianNoise:
    def __init__(self, std_factor=0.01):
        self.std_factor = std_factor


    def __call__(self, spectrum, labels, probes):
        return scaled_gaussian_noise(spectrum, labels, probes,
                                     std_factor=self.std_factor)


def random_scaling(spectra, labels, probes, scale=0.1):
    factors = torch.rand(spectra.shape[:-1], dtype=spectra.dtype,
                         device=spectra.device)
    factors = 1.0 + scale * (factors[..., None] * 2.0 - 1.0)
    spectra_s = spectra * factors
    return spectra_s, labels, probes


class RandomScaling:
    def __init__(self, scale=0.1):
        self.scale = scale


    def __call__(self, spectrum, labels, probes):
        return random_scaling(spectrum, labels, probes, scale=self.scale)


def cut_out(spectra, labels, probes, window_range=(0.1, 0.4)):
    _, L = spectra.shape

    width = torch.rand(1, device=spectra.device)
    width *= (window_range[1] - window_range[0])
    width = (width + window_range[0])

    start = (torch.rand(1, device=spectra.device) - width)

    start = (start * (L - 1)).long()
    width = (width * (L - 1)).long()

    end = start + width
    spectra = spectra.clone()
    spectra[:, start:end] = 0

    return spectra, labels, probes


class CutOut:
    def __init__(self, window_range=(0.1, 0.4)):
        self.window_range = window_range


    def __call__(self, spectra, labels, probes):
        return cut_out(spectra, labels, probes, window_range=self.window_range)


def spectra_dropout(spectra, labels, probes, p=0.1):
    mask = torch.rand_like(spectra) < p
    spectra_d = spectra.clone()
    spectra_d[mask] = 0
    return spectra_d, labels, probes


class SpectraDropout:
    def __init__(self, p=0.1):
        self.p = p


    def __call__(self, spectra, labels, probes):
        return spectra_dropout(spectra, labels, probes, p=self.p)


def random_peaks(spectra, labels, probes, num_peaks=(1, 3)):
    max_val = spectra.max(dim=-1).values
    new_spectra = spectra.clone()
    for _ in range(torch.randint(num_peaks[0], num_peaks[1], (1,))):
        idx = torch.randint(spectra.shape[-1], (1,))
        new_spectra[:, idx] = max_val
    return spectra, labels, probes


class RandomPeaks:
    def __init__(self, num_peaks=(1, 3)):
        self.num_peaks = num_peaks


    def __call__(self, spectra, labels, probes):
        return random_peaks(spectra, labels, probes, num_peaks=self.num_peaks)


def quantize(spectra, labels, probes, thresholds=[0.25, 0.50, 0.75, 1.00]):
    C, _ = spectra.shape
    assert C == 1, "quantize is only implemented for single channel input"
    thresholds = torch.as_tensor(thresholds, dtype=spectra.dtype,
                                 device=spectra.device)
    thresholds = torch.atleast_2d(thresholds).T
    spectra = (spectra >= thresholds).to(spectra.dtype)
    return spectra, labels, probes


class Quantize:
    def __init__(self, thresholds=[0.25, 0.50, 0.75, 1.00]):
        self.thresholds = thresholds


    def __call__(self, spectra, labels, probes):
        return quantize(spectra, labels, probes, thresholds=self.thresholds)


def max_normalize(spectra, labels, probes):
    spectra = spectra / spectra.max(dim=-1, keepdims=True).values
    return spectra, labels, probes


class MaxNormalize:
    def __call__(self, spectra, labels, probes):
        return max_normalize(spectra, labels, probes)


def tic_normalize(spectra, labels, probes):
    spectra = spectra / spectra.sum(dim=-1, keepdims=True)
    return spectra, labels, probes


class TicNormalize:
    def __call__(self, spectra, labels, probes):
        return tic_normalize(spectra, labels, probes)


def log_transform(spectra, labels, probes, eps=1e-8):
    spectra_l = torch.log(spectra.abs() + eps)
    return spectra_l, labels, probes


class LogTransform:
    def __init__(self, eps=1e-8):
        self.eps = eps


    def __call__(self, spectra, labels, probes):
        return log_transform(spectra, labels, probes, eps=self.eps)


def rand_mz_offset(spectra, labels, probes, std, std_clip=2):
    offset = truncnorm.rvs(-std_clip, std_clip, scale=std)
    x = np.arange(spectra.shape[-1])
    fill_value = float(0.5 * (spectra.squeeze()[0] + spectra.squeeze()[-1]))

    #interp = interp1d(x, spectra, kind='cubic', fill_value=fill_value,
    #                  bounds_error=False)
    interp = RegularGridInterpolator((x,), spectra.numpy().squeeze(),
                                     method='linear',
                                     fill_value=fill_value, bounds_error=False)
    spectra_out = torch.from_numpy(interp(x + offset)).clamp(min=0)
    spectra_out = spectra_out.to(spectra.dtype).squeeze()[None]
    return spectra_out, labels, probes


class RandMzOffset:
    def __init__(self, std, std_clip=2):
        self.std = std
        self.std_clip = std_clip


    def __call__(self, spectra, labels, probes):
        return rand_mz_offset(spectra, labels, probes, self.std, self.std_clip)


def anscombe(spectra, labels, probes):
    spectra = 2 * np.sqrt(spectra - spectra.min() + 3/8)
    return spectra, labels, probes


class Anscombe:
    def __call__(self, spectra, labels, probes):
        return anscombe(spectra, labels, probes)


# ############################
# Spectrum Batch Augmentations
# ############################

class MixUp:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.rand_beta = torch.distributions.beta.Beta(alpha, alpha)


    def __call__(self, spectrum, labels, probes):
        a = self.rand_beta.sample((len(spectrum), 1, 1)).to(spectrum.device)
        idx = torch.randperm(len(spectrum), device=spectrum.device)
        spectrum = a * spectrum + (1 - a) * spectrum[idx]
        a = a.squeeze(-1)  # Squeeze out sequence dim for labels
        labels = a * labels + (1 - a) * labels[idx]
        return spectrum, labels, probes


def mix_up(spectrum, labels, probes, alpha=0.4):
    m = MixUp(alpha)
    return m(spectrum, labels, probes)


def patch_mixup(spectra, labels, probes, p=256):
    N, C, L = spectra.shape
    assert L % p == 0
    chunked_spectra = spectra.clone().view(N, C, -1, p)
    if labels.ndim > 1 and labels.shape[-1] > 1:
        # One-hot - convert back to index
        labels_index = labels.argmax(dim=-1)
    else:
        labels_index = labels
    u_labels, counts = labels_index.unique(return_counts=True)
    for i in range(chunked_spectra.shape[2]):
        perm = torch.arange(N, device=spectra.device)
        for l, c in zip(u_labels, counts):
            mask = labels_index == l
            perm_l = torch.randperm(c, device=spectra.device)
            perm[mask] = perm[mask][perm_l]
        chunked_spectra[:, :, i] = chunked_spectra[perm, :, i]
    return chunked_spectra.view(N, C, L), labels, probes


class PatchMixup:
    def __init__(self, p=256):
        self.p = p


    def __call__(self, spectra, labels, probes):
        return patch_mixup(spectra, labels, probes, self.p)


################
# Common Utility
################

class NoOp:
    def __init__(self):
        pass


    def __call__(self, *args):
        return args

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms


    def __call__(self, spectrum, labels, probes):
        for t in self.transforms:
            spectrum, labels, probes = t(spectrum, labels, probes)

        return spectrum, labels, probes


def from_config(cfg):
    assert len(cfg.keys()) == 1, 'cfg must correspond to a single augmentation'
    name = list(cfg.keys())[0]
    params = cfg[name]
    if name == 'pad_to_divisible':
        return PadToDivisible(**params)
    elif name == 'scaled_gaussian_noise':
        return ScaledGaussianNoise(params['std_factor'])
    elif name == 'random_scaling':
        return RandomScaling(params['scale'])
    elif name == 'cut_out':
        return CutOut(params['window_range'])
    elif name == 'spectra_dropout':
        return SpectraDropout(params['p'])
    elif name == 'random_peaks':
        return RandomPeaks((params['min_peaks'], params['max_peaks']))
    elif name == 'quantize':
        return Quantize(params['thresholds'])
    elif name == 'log_transform':
        return LogTransform(params['eps'])
    elif name == 'max_normalize':
        return MaxNormalize()
    elif name == 'tic_normalize':
        return TicNormalize()
    elif name == 'rand_mz_offset':
        return RandMzOffset(**params)
    elif name == 'anscombe':
        return Anscombe()
    elif name == 'mix_up':
        return MixUp(params['alpha'])
    elif name == 'patch_mixup':
        return PatchMixup(params['p'])
    else:
        raise ValueError(f'Uknown augmentation: {name}, cfg: {cfg}')


def from_config_list(augmentations):
    if augmentations:
        return Compose([from_config(a) for a in augmentations])
    else:
        return None
