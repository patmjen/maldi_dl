import h5py
import numpy as np
import torch
import torch.nn.functional as F

def decode_h5(fin: h5py.File) -> dict:
    """
    Decode Frederic's HDF5 files to standard format

    Args:
        fin: Open HDF5 file object.

    Returns:
        Dictionary with
            classes: class names as strings
            samples: (N, S) numpy array with spectrum data
            groundtruth_labels: (N,) numpy array with labels
            sample_coords: (N, 2) xy-coordinates for spectra locations
            cluster: (N,) if found, the tissue cluster of the spectrum

    """
    classes = [None] * len(fin['class_names'])
    for s in fin['class_names']:
        cid, name = s.decode().split(':')
        classes[int(cid)] = name

    out = {'classes': classes,
           'samples': fin['spots'][()],
           'groundtruth_labels': fin['spot_labels_groundtruth'][()],
           'sample_coords': fin['spot_coords'][()]}

    try:
        out['cluster'] = fin['cluster'][()]
    except KeyError:
        # no cluster defined/ no probe mode, only spectra prediction
        pass

    return out


def psnr(spectra: np.array, cutoff: float = 1.0) -> np.array:
    """
    (Modified) Peak Signal to Noise Ratio

    Can specify a cutoff fraction so the signal variance is only computed from
    the (cutoff * len(spectra))'th smallest values.

    Args:
        spectra: (N, S) numpy array with N spectra of length S
        curoff: float with fraction of small spectra to keep

    Returns:
        (N,) array with (modified) peak signal to noise ratio
    """
    if cutoff < 1.0:
        end = int(spectra.shape[1] * cutoff)
        s_spectra = np.partition(spectra, end, axis=1)[:, :end]
    else:
        s_spectra = spectra
    return 10.0 * np.log10(np.max(spectra, axis=1) / np.std(s_spectra, axis=1))


class SpectrumDataset(torch.utils.data.Dataset):
    def __init__(self, h5, transforms=None, one_hot=False, filter_spectra=True):
        super().__init__()
        data = decode_h5(h5)

        self.spectra = torch.as_tensor(data['samples']).float()
        self.labels = torch.as_tensor(data['groundtruth_labels'])
        self.probes = torch.as_tensor(data['cluster'])
        self.transforms = transforms

        if one_hot:
            self.labels = F.one_hot(self.labels, len(data['classes'])).float()

        if filter_spectra:
            mpsnr = psnr(self.spectra.numpy(), cutoff=0.6)
            keep_idx = mpsnr >= 15
            self.spectra = self.spectra[keep_idx]
            self.labels = self.labels[keep_idx]
            self.probes = self.probes[keep_idx]

        invalid_mask = ~torch.isfinite(self.spectra).all(dim=1) \
                     | (self.spectra.abs() > 1e8).any(dim=1)
        if invalid_mask.any():
            import warnings
            warnings.warn(
                f'Found {invalid_mask.sum()} spectra with invalid values - removing'
            )
            self.spectra = self.spectra[~invalid_mask]
            self.labels = self.labels[~invalid_mask]
            self.probes = self.probes[~invalid_mask]


    def __getitem__(self, index):
        spectra = self.spectra[index][None]
        labels = self.labels[index]
        probes = self.probes[index]
        if self.transforms is not None:
            spectra, labels, probes = self.transforms(
                spectra, labels, probes
            )

        return spectra, labels, probes


    def __len__(self):
        return len(self.spectra)
