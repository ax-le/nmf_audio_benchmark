"""
This module contains the dataloaders for the MAPS dataset. Can be extended to other datasets (notably MAESTRO).

It loads the spectrogram and the annotations for each song in the dataset.
"""

import mirdata # Might be useful
import librosa
import pathlib
import shutil
import os
import glob
import warnings

# import base_audio.signal_to_spectrogram as signal_to_spectrogram
from nmf_audio_benchmark.dataloaders.base_dataloader import *

eps = 1e-10

class TranscriptionBaseDataloader(BaseDataloader):
    def __init__(self, feature, cache_path = None, sr=44100, n_fft = 2048, hop_length = 512, verbose = False, multichannel = False):
        """
        Constructor of TranscriptionBaseDataloader class. Inherits from the BaseDataloader class.

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
        assert not multichannel # Multichannel is not handled yet.


class MAPSDataloader(TranscriptionBaseDataloader):
    
    name = "MAPS"

    def __init__(self, datapath, feature, subfolder, cache_path = None, sr=44100, n_fft = 2048, hop_length = 512, verbose = False, multichannel = False, chunk_duration = 30):
        """
        Constructor of the MAPSDataloader class.

        Parameters
        ----------
        TODO
        """
        super().__init__(feature = feature, cache_path = cache_path, sr=sr, n_fft=n_fft, hop_length=hop_length, verbose=verbose, multichannel=multichannel)
        self.datapath = datapath
        self.subset_path = f"{datapath}/{subfolder}/MUS" # Listing all files from the "MUS" subsubfolder, i.e. all musics
        self.song_path = glob.glob(rf"{self.subset_path}/*.wav")
        if self.song_path == []:
            raise FileNotFoundError(f"No .wav files found in {self.subset_path}. The path is probably incorect.")
        self.indexes = [song_path.split("/")[-1].split("\\")[-1].split(".")[0] for song_path in self.song_path]
        self.annotations = glob.glob(rf"{self.subset_path}/*.txt")
        self.chunk_duration = chunk_duration

        if self.multichannel:
            raise NotImplementedError("Multichannel is not implemented yet") from None

    def __getitem__(self, index):
        """
        Return the data of the index-th track.
        """
        song_idx = self.indexes[index]
        song_path = f"{self.subset_path}/{song_idx}.wav"
        annotation_path = f"{self.subset_path}/{song_idx}.txt"
        annotations = load_reference_annotations(annotation_path, time_limit=self.chunk_duration)

        if self.multichannel:
            raise NotImplementedError("Multichannel is not implemented yet") from None

        mono = not self.multichannel
        signal = librosa.load(song_path, sr=self.feature_object.sr, mono=mono, duration=self.chunk_duration)[0]

        spectrogram = self.get_spectrogram(signal)
        return song_idx, spectrogram, annotations
    
    def format_dataset(self, delete_original=False):
        """
        Automatically format the dataset in a consistent way, because folder ENSTDkAm1 and ENSTDkAm2 are not in the same format than the others.
        """
        os.makedirs(f"{self.datapath}/ENSTDkAm", exist_ok=True)

        try:
            shutil.copytree(f"{self.datapath}/ENSTDkAm1/ENSTDkAm", f"{self.datapath}/ENSTDkAm", dirs_exist_ok=True)
            shutil.copytree(f"{self.datapath}/ENSTDkAm2/ENSTDkAm", f"{self.datapath}/ENSTDkAm", dirs_exist_ok=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Source file not found: {self.datapath}/ENSTDkAm1/ENSTDkAm") from None
        except PermissionError:
            print(f"Permission denied: {self.datapath}/ENSTDkAm")
        except Exception as e:
            print(f"Error occurred: {e}")

        if delete_original:
            shutil.rmtree(f"{self.datapath}/ENSTDkAm1")
            shutil.rmtree(f"{self.datapath}/ENSTDkAm2")
    
def load_reference_annotations(ref_path, time_limit = None):
    """
    Load the ground truth transcription_evaluation in an array, for comparing it to the found transcription_evaluation in 'compute_statistical_rates_on_array()'
    The reference needs to be a txt, and format as in MAPS (which is the dataset for which this function has been developed)

    Parameters
    ----------
    ref_path: String
        The path to the reference file (in txt)
    time_limit: None or integer
        The time limit index, to crop the reference when only an excerpt is transcribed
        Default: None

    Returns
    -------
    truth_array: list of lists
        List of all notes of the reference, format in lists containing Onsets, Offsets and Pitches, at respective indexes 0, 1, 2
    """
    truth_array = []

    with open(ref_path) as f:
        truth_lines = f.readlines()[1:] # To discard the title/legend in ground truth

    for lines_index in range(len(truth_lines)):
        # Creates a list with the line of the reference, splitted on tabulations
        if truth_lines[lines_index] != '\n': # To avoid empty lines
            line_to_array = (truth_lines[lines_index].replace("\n", "")).split("\t")
            if (time_limit != None) and (float(line_to_array[0]) > time_limit): # if onset > time_limit (note outside of the cropped excerpt)
                truth_lines = truth_lines[:lines_index]
                break
            else:
                truth_array.append(line_to_array)

    return truth_array

if __name__ == "__main__":
    musdb_18 = MAPSDataloader('/home/a23marmo/datasets/MAPS', feature = "mel", subfolder = "AkPnBcht", cache_path = None)
    print(len(musdb_18))