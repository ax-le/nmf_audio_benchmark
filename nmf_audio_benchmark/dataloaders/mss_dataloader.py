"""
This module contains the dataloaders for the main dataset in MSS: MusDB18 dataset. Can be extended to other datasets.

It loads the spectrogram and the stems annotations for each song in the dataset.
"""

import mirdata
import musdb
import pathlib
import shutil
import numpy as np
import warnings

import base_audio.signal_to_spectrogram as signal_to_spectrogram

eps = 1e-10

class BaseDataloader():
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

        self.multichannel = multichannel

    def __getitem__(self, index):
        """
        Return the data of the index-th track.
        """
        raise NotImplementedError("This method should be implemented in the child class") from None

    def __len__(self):
        """
        Return the number of tracks in the dataset.
        """
        raise NotImplementedError("This method should be implemented in the child class") from None

    def get_spectrogram(self, signal): # The spectrogram is not saved in the cache because it is too large in general
        """
        Returns the spectrogram, from the signal of a song.
        """
        return self.feature_object.get_spectrogram(signal)

class MusDBDataloader(BaseDataloader):
    
    name = "musdb"

    def __init__(self, datapath, feature, cache_path = None, sr=44100, hop_length = 512, verbose = False, multichannel = False, chunk_duration = 30):
        """
        Constructor of the MusDBDataloader class.

        Parameters
        ----------
        datapath : string
            The path to the MusDB dataset.
        feature : string
            The feature to compute the spectrogram. Must be a valid feature name.
        cache_path : string
            The path where to save the computed barwise TF matrices and bars. If None, the cache is not used.
            The default is None.
        sr : int
            The sampling rate of the audio files.
            The default is None, meaning that it will keep the original sampling rate of the audio file.
        hop_length : int
            The hop length of the spectrogram.
            The default is 512.
        verbose : bool
            If True, print some information about the cache.
            The default is False
        """
        super().__init__(feature = feature, cache_path = cache_path, sr=sr, hop_length=hop_length, verbose=verbose, multichannel=multichannel)
        self.mus = musdb.DB(root=datapath, subsets="test", download=False)
        self.indexes = range(len(self.mus))
        self.chunk_duration = chunk_duration

    def __getitem__(self, index):
        """
        Return the data of the index-th track.
        """
        track = self.mus.tracks[index]

        # track.chunk_start=0
        track.chunk_duration=self.chunk_duration
        signal = track.audio.T
        if not self.multichannel:
            signal = np.mean(signal, axis=0) # Average the channels
            stems = np.array([np.mean(stem, axis=1) for stem in track.stems[1:]])

        else:
            raise NotImplementedError("Multichannel is not implemented yet") from None

        spectrogram = self.get_spectrogram(signal)
        return track.name, spectrogram, stems
    
    def __len__(self):
        """
        Return the number of tracks in the dataset.
        """
        return len(self.mus)