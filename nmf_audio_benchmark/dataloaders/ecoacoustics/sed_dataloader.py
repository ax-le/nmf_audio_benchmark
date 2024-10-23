"""
Created on June 2024

@author: a23marmo

Defines the dataloaders class for the Sound Event Detection task (SED), restricted to whale calls.
Contains the NARW (North Atlantic Right Whale) and Bouffaut2020 dataset.
For now, corresponds to calls from Whales at very low frequencies.
"""

import pandas as pd
import os

from nmf_audio_benchmark.dataloaders.ecoacoustics.generic_dataloader import *

class NARWDataset(GenericDataloader): # Using the generic dataloader as mother class

    def __init__(self, audio_path, annotations_file, feature, cache_path = None, sr=44100, n_fft = 2048, hop_length = 512, verbose = False):

        """
        Initializes the dataset object.

        Parameters
        ----------
        audio_path : str
            Path to the audio files.
        annotations_file : str
            Path to the annotations file.
        feature_object : FeatureObject
            Feature object, defining the important parameters to compute spectrograms.
        """
        super().__init__(audio_path=audio_path, feature=feature, cache_path = cache_path, sr=sr, n_fft = n_fft, hop_length = hop_length, verbose = verbose)

        self.annotations = pd.read_csv(annotations_file)
    
    def get_annotations(self, idx):
        """
        Returns the annotations of the idx-th signal in the dataset.
        Annotations consist here of the time and types of the calls.

        Parameters
        ----------
        idx : int
            Index of the signal in the dataset.

        Returns
        -------
        annot_this_file : pandas Series
            Time and type of the calls.
        """ 
        annot_this_file = self.annotations[self.annotations["filename"] == self.indexes[idx]]
        annotated_detections = list(annot_this_file["timestamp"])
        return annotated_detections

class Bouffaut(GenericDataloader):
    def __init__(self, audio_path, annotations_file, feature, cache_path = None, sr=44100, n_fft = 2048, hop_length = 512, verbose = False):
        """
        Initializes the dataset object.
        
        Parameters
        ----------
        audio_path : str
            Path to the audio files.
        annotations_file : str
            Path to the annotations file.
        feature_object : FeatureObject
            Feature object, defining the important parameters to compute spectrograms.
        """
        super().__init__(audio_path=audio_path, feature=feature, cache_path = cache_path, sr=sr, n_fft = n_fft, hop_length = hop_length, verbose = verbose)
        self.annotations = pd.read_csv(annotations_file, delimiter='\t')

    def get_annotations(self, idx):
        """
        Returns the annotations of the idx-th signal in the dataset.
        Annotations consist here of the type of call, the beginning and the end of the call.

        Parameters
        ----------
        idx : int
            Index of the signal in the dataset.
        
        Returns
        -------
        type : pandas Series
            Type of the calls in the signal.
        begin : pandas Series
            Beginning of the calls in the signal.
        end : pandas Series
            End of the calls in the signal.
        """
        annot_this_file = self.annotations[self.annotations["File"] == self.indexes[idx].split(".")[0]]
        type = annot_this_file['Type']
        begin = annot_this_file['Begin Time (s)']
        end = annot_this_file['End Time (s) ']
        return type, begin, end
    
    def crop_annotations(self, annotations, time_limit_s):
        """
        Crops the annotations to the time limit.
        Useful to focus experiments on a part of the audio signal.

        Parameters
        ----------
        annotations : tuple
            Tuple of pandas Series, containing the type of the calls, the beginning and the end of the calls.
        time_limit_s : float
            Time limit in seconds.

        Returns
        -------
        type : pandas Series
            Type of the calls in the signal.
        begin : pandas Series
            Beginning of the calls in the signal.
        end : pandas Series
            End of the calls in the signal.
        """
        type, begin, end = annotations
        indices_ok = begin[begin < time_limit_s].keys()
        type = type[indices_ok]
        begin = begin[indices_ok]
        end = end[indices_ok]
        return type, begin, end
    
# Mainly developped for the Bouffaut Dataset. Allow to create a unique annotation file instead of 2.
def create_one_annotation_file(dataset_path, annotations_1="Annotation.txt", annotations_2="Annotation_RR48_2013_D151.txt"):
    """
    Merges two annotation files into one.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset.
    annotations_1 : str
        Name of the first annotation file.
    annotations_2 : str
        Name of the second annotation file.
    """
    annot_1 = pd.read_csv(f"{dataset_path}/{annotations_1}", delimiter='\t')
    annot_2 = pd.read_csv(f"{dataset_path}/{annotations_2}", delimiter='\t')
    pd.concat([annot_1, annot_2]).to_csv(f"{dataset_path}/merged_annotations.txt", sep='\t', index=False)

if __name__ == "__main__":
    bouffaut2020 = Bouffaut('/home/a23marmo/datasets/bioac/bouffaut2020/audio', annotations_file = "/home/a23marmo/datasets/bioac/bouffaut2020/merged_annotations.txt", feature = "mel", cache_path = None)
    print(len(bouffaut2020))
