"""
Created on October 2024

@author: a23marmo

Defines the dataset class for the Source Count task, in particular for the AnuraSet dataset. Inherits from the generic dataset class.
Annotations are simplified to the number of species present in the signal.
Species are counted as individual species, hence the number of species may be different from the number of calls.
"""

import pandas as pd

from nmf_audio_benchmark.dataloaders.ecoacoustics.generic_dataloader import *

class AnuraSet(GenericDataloader):
    # A class to handle and process data in AnuraSet, so as to estimate the number of species present in the audio.
    def __init__(self, audio_path, subfolder, annotations_file, feature, cache_path = None, sr=44100, n_fft = 2048, hop_length = 512, verbose = False):
        self.subset_path = f'{audio_path}/{subfolder}'
        super().__init__(audio_path=self.subset_path, feature=feature, cache_path = cache_path, sr=sr, n_fft = n_fft, hop_length = hop_length, verbose = verbose)
        self.annotations = pd.read_csv(annotations_file)
        if 'nonzero_species_count' not in self.annotations.columns:
            self.annotations['nonzero_species_count'] = self.annotations.filter(regex='^SPECIES').apply(lambda row: (row != 0).sum(), axis=1)

    def get_annotations(self, idx):
        """
        Returns the annotations of the idx-th signal in the dataset.
        Annotations consist here of the number of species present in the signal.
        This is a simplification of the original dataset, where the annotations are more complex.
        Species are counted as individual species, hence the number of species may be different from the number of calls.

        Parameters
        ----------
        idx : int
            Index of the signal in the dataset.

        Returns
        -------
        annot_this_file : int
            Number of species present in the signal.
        """ 
        number_of_species = self.annotations['nonzero_species_count'][self.annotations["AUDIO_FILE_ID"] == self.indexes[idx].split(".")[0]].values[0]
        return number_of_species
    
if __name__ == "__main__":
    anuraset = AnuraSet('/home/a23marmo/datasets/anuraset/raw_data', subfolder = 'INCT4', annotations_file = "/home/a23marmo/datasets/anuraset/metadata.csv", feature = "mel", cache_path = None)
    print(len(anuraset))
