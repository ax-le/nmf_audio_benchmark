"""
This module contains a template dataloader, which can be used to add a new dataloader.
It follows the standard in Python for dataloaders in PyTorch.

It loads the data, computes the spectrogram, format folders if needed, and returns a spectrogram and the annotatioons.
If large files are computed, they can be saved in a cache folder to avoid recomputing them.

It uses 3 functions:
- __init__ : Constructor of the BaseDataloader class.
- __getitem__ : Return the data of the index-th track.
- __len__ : Return the number of tracks in the dataset.
"""

import mirdata # Can be used to load a MIR dataset

import librosa
import shutil
import numpy as np
import warnings

import base_audio.signal_to_spectrogram as signal_to_spectrogram

eps = 1e-10

class TemplateDataloader():
    def __init__(self, feature, cache_path = None, sr=44100, n_fft = 2048, hop_length = 512, verbose = False, multichannel = False):
        """
        Constructor of the BaseDataloader class.

        Parameters
        ----------
        feature : string
            The feature to compute the spectrogram. Must be a valid feature name.
        cache_path : string
            The path where to save the computed barwise TF matrices and bars. If None, the cache is not used.
            The default is None.
        sr : int
            The sampling rate of the audio files.
            The default is None, meaning that it will keep the original sampling rate of the audio file.
        n_fft : int
            The number of samples in each STFT window.
        hop_length : int
            The hop length of the spectrogram.
            The default is 512.
        verbose : bool
            If True, print some information about the cache.
            The default is False
        multichannel : bool
            If True, the dataloader will return the multichannel audio.
            The default is False.
        """
        self.cache_path = cache_path
        self.verbose = verbose

        self.feature_object = signal_to_spectrogram.FeatureObject(sr, feature, hop_length=hop_length, n_fft = n_fft)

    def __getitem__(self, index):
        """
        Return the data of the index-th track.
        """
        raise NotImplementedError("This method should be implemented.") from None
        return track_id, spectrogram, annotations

    def __len__(self):
        """
        Return the number of tracks in the dataset.
        """
        raise NotImplementedError("This method should be implemented.") from None