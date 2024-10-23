"""
Created on June 2024

@author: a23marmo

Defines a generic dataloader, for audio files which are standalones.
"""

# Imports
import librosa
import os

from nmf_audio_benchmark.dataloaders.base_dataloader import *

class GenericDataloader(BaseDataloader):
    # This class is a generic Dataloder class, which can be used to import standard datafiles.
    def __init__(self, audio_path, feature, cache_path = None, sr=44100, n_fft = 2048, hop_length = 512, verbose = False):
        """
        Initializes a generic dataset.

        Parameters
        ----------
        audio_path : str
            Path to the audio files.
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
        super().__init__(feature=feature, cache_path = cache_path, sr=sr, n_fft = n_fft, hop_length = hop_length, verbose = verbose, multichannel = False) # multichannel is False by default.

        self.audio_path = audio_path
        self.indexes = list(os.listdir(self.audio_path))
    
    def __getitem__(self, idx):
        """
        Returns the spectrogram and the annotations of the idx-th signal in the dataset.
        
        Parameters
        ----------
        idx : int
            Index of the signal in the dataset.

        Returns
        -------
        spec : numpy array
            Spectrogram of the signal.
        annot_this_file : not defined
            Annotations of the signal, the type of which will depend on the dataset.
        """
        signal, orig_sr = librosa.load(os.path.join(self.audio_path, self.indexes[idx]), sr = self.feature_object.sr)
        if self.feature_object.sr != orig_sr:
            signal = librosa.resample(signal, orig_sr, self.feature_object.sr)
        spec = self.feature_object.get_spectrogram(signal)
        annot_this_file = self.get_annotations(idx)

        return spec, annot_this_file
    
if __name__ == "__main__":
    four_whales = GenericDataloader('/home/a23marmo/datasets/bioac/whales', feature = "mel", cache_path = None)
    print(len(four_whales))
