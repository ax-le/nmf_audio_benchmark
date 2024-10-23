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

# import base_audio.signal_to_spectrogram as signal_to_spectrogram
from nmf_audio_benchmark.dataloaders.base_dataloader import *

eps = 1e-10

class MSSBaseDataloader(BaseDataloader):
    def __init__(self, feature, cache_path = None, sr=44100, n_fft = 2048, hop_length = 512, verbose = False, multichannel = False):
        """
        Constructor of the MSSBaseDataloader class. Inherits from the BaseDataloader class.

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
        super().__init__(feature=feature, cache_path=cache_path, sr=sr, n_fft=n_fft, hop_length = hop_length, verbose = verbose, multichannel = multichannel)

        if self.multichannel:
            raise NotImplementedError("Multichannel is not implemented yet") from None

class MusDBDataloader(MSSBaseDataloader):

    # You actually need to install the musdb package, to handle musdb files.
    
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
        self.all_stems = ["drums", "bass", "other", "vocals"] # Adding the stems labels for the whole dataset. Easy for MusDB, may be tedious for other datasets.

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

        # In MusDB, all annotations are ordered as [drums, bass, other, vocals]
        stems_labels = self.all_stems # Labels are used to store the results according to each source.

        spectrogram = self.get_spectrogram(signal)
        return track.name, spectrogram, stems, stems_labels
    
if __name__ == "__main__":
    musdb_18 = MusDBDataloader('/home/a23marmo/datasets/musdb18', feature = "mel", cache_path = None)
    print(len(musdb_18))