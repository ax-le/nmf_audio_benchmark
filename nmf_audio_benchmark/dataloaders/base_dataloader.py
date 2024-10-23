"""
This module contains a base dataloader, which is used a a super-class for all other dataloaders.

It allows to load the signals present in the dataset, and define how to compute spectrograms.
"""

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

        self.indexes = NotImplementedError("Needs to be defined in child classes") # Needs to be redefined in child classes.

    def __getitem__(self, index):
        """
        Return the data of the index-th track.
        """
        raise NotImplementedError("This method should be implemented in the child class") from None

    def __len__(self):
        """
        Return the number of tracks in the dataset.

        By default, returns the number of elements in the indexes.
        """
        return len(self.indexes)

    def get_spectrogram(self, signal): # The spectrogram is not saved in the cache because it is too large in general
        """
        Returns the spectrogram, from the signal of a song.
        """
        return self.feature_object.get_spectrogram(signal)
    
    def get_item_of_id(self, audio_id):
        """
        Returns the item of the given id. Requires self.indexes to be set.
        
        Parameters
        ----------
        audio_id : str
            Id of the signal in the dataset.

        Returns
        -------
        Whatever is returned in the getter of the current class.
        """
        # index = self.indexes.index(audio_id)
        try:
            index = self.indexes.index(audio_id)
        except ValueError:
            try:
                index = self.indexes.index(str(audio_id))
            except ValueError:
                raise ValueError(f"Audio {audio_id} not found in the dataset") from None

        return self.__getitem__(index)
    
if __name__ == "__main__":
    base = BaseDataloader(feature = "ahah")
    len(base)