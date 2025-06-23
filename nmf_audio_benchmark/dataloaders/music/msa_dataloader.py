"""
This module contains the dataloaders for the main datasets in MSA: SALAMI [1], RWCPOP [2] and Beatles [3] datasets.

Using NMF for the MSA task is not very standard. Here, it is used at the barscale, following the work presented in [4,5].
We advise to read papers [4,5] for more details about the method and the specifics treatments used in these dataloaders (Barwise TF matrix) and this task.

It loads the data, but also computes the barwise TF matrix and the bars from the audio files.
When the barwise TF matrix and bars are computed, they are saved in a cache folder (if provided) to avoid recomputing them.

References
----------
[1] Smith, J. B. L., Burgoyne, J. A., Fujinaga, I., De Roure, D., & Downie, J. S. (2011). Design and creation of a large-scale database of structural annotations. In Proceedings of the International Society for Music Information Retrieval Conference (ISMIR) (pp. 555-560). Miami, FL.

|2] Goto, M., Hashiguchi, H., Nishimura, T., & Oka, R. (2002). RWC Music Database: Popular, classical and jazz music databases. In Proceedings of the International Society for Music Information Retrieval Conference (ISMIR) (pp. 287-288).

[3] Harte, C. (2010). Towards automatic extraction of harmony information from music signals (Doctoral dissertation). Queen Mary University of London.
http://isophonics.net/content/reference-annotations-beatles

[4] Marmoret, A., Cohen, J. E., & Bimbot, F. (2023). Barwise Music Structure Analysis with the Correlation Block-Matching Segmentation Algorithm. Transactions of the International Society for Music Information Retrieval (TISMIR), 6(1), 167-185.

[5] Marmoret, A., Cohen, J. E., & Bimbot, F. (2022, June). Barwise compression schemes for audio-based music structure analysis. Proceedings of the Sound and Music Computing Conference (SMC 2022), Saint-Étienne, France.
"""
import librosa
import mirdata
import mirdata.download_utils
import pathlib
import shutil
import numpy as np
import os
import warnings

from nmf_audio_benchmark.dataloaders.base_dataloader import *

import as_seg

eps = 1e-10

class MSABaseDataloader(BaseDataloader):
    def __init__(self, feature, cache_path = None, sr=44100, n_fft = 2048, hop_length = 512, subdivision = 96, verbose = False):
        """
        Constructor of the MSABaseDataloader class. Inherits from the BaseDataloader class.

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
            The default is 32.
        subdivision : int
            The number of subdivisions of a bar.
            The default is 96.
        verbose : bool
            If True, print some information about the cache.
            The default is False
        """
        super().__init__(feature=feature, cache_path=cache_path, sr=sr, n_fft=n_fft, hop_length = hop_length, verbose = verbose, multichannel = False) # Multichannel is not handled for now in MSA.

        # For barwise or beatwise processing
        self.subdivision = subdivision
    
    def get_bars(self, audio_path, index = None):
        """
        Return the bars of the song.
        They are computed from the audio file.
        If the cache is used, the bars are saved in the cache.
        An identifier of the song should be provided to save the bars in the cache.
        """
        def _compute_bars(): # Define the function to compute the bars
            return as_seg.data_manipulation.get_bars_from_audio(audio_path)

        # If a cache is set
        if self.cache_path is not None:

            # No identifier is provided for this song, hence it cannot be saved in the cache
            if index is None:
                warnings.warn("No index provided for the cache, the cache will be ignored")
            
            # An identifier is provided
            else:
                dir_save_bars_path = f"{self.cache_path}/bars"
                
                # Tries to load the bars from the cache
                try:
                    bars = np.load(f"{dir_save_bars_path}/{index}.npy", allow_pickle=True)
                    if self.verbose:
                        print("Using cached bars.")
                
                # If the file is not found, the bars are computed and saved in the cache
                except FileNotFoundError:
                    bars = _compute_bars() # Compute the bars

                    # Save the bars in the cache
                    pathlib.Path(dir_save_bars_path).mkdir(parents=True, exist_ok=True)
                    np.save(f"{dir_save_bars_path}/{index}.npy", bars)
                
                # Return the bars
                return bars
        # No cache is set, the bars are computed and returned
        return _compute_bars()

    def get_barwise_tf_matrix(self, track_path, bars, index = None):
        """
        Return the barwise TF matrix of the song.
        It is computed from the signal of the song and the bars.
        If the cache is used, the barwise TF matrix is saved in the cache.
        An identifier of the song should be provided to save the barwise TF matrix in the cache.
        """
        def _compute_barwise_tf_matrix(): # Define the function to compute the barwise TF matrix
            # Load the signal of the song
            sig, _ = librosa.load(track_path, sr=self.feature_object.sr, mono=True) #torchaudio.load(track.audio_path)
            # Compute the spectrogram
            spectrogram = self.feature_object.get_spectrogram(sig)
            return as_seg.barwise_input.barwise_TF_matrix(spectrogram, bars, self.feature_object.hop_length/self.feature_object.sr, self.subdivision) + eps
        
        # If a cache is set 
        if self.cache_path is not None:
            # No identifier is provided for this song, hence it cannot be saved in the cache
            if index is None:
                warnings.warn("No index provided for the cache, the cache will be ignored")
            
            # An identifier is provided
            else:
                cache_file_name = f"{index}_{self.feature_object.feature}_subdiv{self.subdivision}"
                dir_save_barwise_tf_path = f"{self.cache_path}/barwise_tf_matrix"
                
                # Tries to load the barwise TF matrix from the cache
                try:
                    barwise_tf_matrix = np.load(f"{dir_save_barwise_tf_path}/{cache_file_name}.npy", allow_pickle=True)
                    if self.verbose:
                        print("Using cached Barwise TF matrix.")
                
                # If the file is not found, the barwise TF matrix is computed and saved in the cache
                except FileNotFoundError:
                    barwise_tf_matrix = _compute_barwise_tf_matrix() # Compute the barwise TF matrix

                    # Save the barwise TF matrix in the cache
                    pathlib.Path(dir_save_barwise_tf_path).mkdir(parents=True, exist_ok=True)
                    np.save(f"{dir_save_barwise_tf_path}/{cache_file_name}.npy", barwise_tf_matrix)
                
                # Return the barwise TF matrix
                return barwise_tf_matrix
            
        # No cache is set, the barwise TF matrix is computed and returned
        return _compute_barwise_tf_matrix()

    def save_segments(self, segments, name):
        """
        Save the segments of a song in the original folder.
        Important for reproducibility.
        """
        # mirdata_segments = mirdata.annotations.SectionData(intervals=segments, interval_unit="s")
        # jams_segments = mirdata.jams_utils.sections_to_jams(mirdata_segments)
        dir_save_path = f"{self.data_path}/estimations/segments/{self.name.lower()}"
        pathlib.Path(dir_save_path).mkdir(parents=True, exist_ok=True)
        np.save(f"{dir_save_path}/{name}.npy", segments)

    def score_flat_segmentation(self, segments, annotations):
        """
        Compute the score of a flat segmentation.
        """
        close_tolerance = as_seg.data_manipulation.compute_score_of_segmentation(annotations, segments, window_length=0.5)
        large_tolerance = as_seg.data_manipulation.compute_score_of_segmentation(annotations, segments, window_length=3)
        return close_tolerance, large_tolerance
    
    def segments_from_bar_to_seconds(self, segments, bars):
        """
        Convert the segments from bars to seconds. Wrapper for the function in data_manipulation.
        """
        # May be useful, if ever.
        return as_seg.data_manipulation.segments_from_bar_to_time(segments, bars)

class RWCPopDataloader(MSABaseDataloader):
    """
    Dataloader for the RWC Pop dataset.
    """

    name = "rwcpop"

    def __init__(self, datapath, feature, cache_path = None, download=False, sr=44100, hop_length = 32, subdivision = 96):
        """
        Constructor of the RWCPopDataloader class.

        Parameters
        ----------
        Same then for BaseDataloader, with the addition of:

        datapath : string
            The path to the dataset.
        download : bool
            If True, download the dataset using mirdata.
            The default is False.
        """
        super().__init__(feature=feature, cache_path=cache_path, sr=sr, hop_length=hop_length, subdivision=subdivision)
        self.datapath = datapath
        rwcpop = mirdata.initialize('rwc_popular', data_home = datapath)
        if download:
            # Adding the MIREX10_SECTIONS annotations
            mirdata.datasets.rwc_popular.REMOTES["MIREX10_SECTIONS"]=mirdata.download_utils.RemoteFileMetadata(
                    filename="MIREX10_SECTIONS.zip",
                    url="https://github.com/ax-le/mirex10_sections/archive/main.zip",
                    checksum="85f71a8cf3dda4438366b55364d29c59",
                    destination_dir="annotations")

            rwcpop.download()
            
        self.all_tracks = rwcpop.load_tracks()

        # Modifying the audio paths: now, they are named with the indexes of the songs
        # Instead of having several folders indexing from 1 to 16 containing the files.
        # This goes against the mirdata standards, but it is more convenient for me.
        for a_track in self.all_tracks.values():
            a_track.audio_path = f"{datapath}/audio/{a_track.track_id}.mp3"

        self.indexes = rwcpop.track_ids

        # self.dataset_name = "rwcpop"

    def __getitem__(self, index):
        """
        Return the data of the index-th track.
        """
        track_id = self.indexes[index]
        track = self.all_tracks[track_id]

        # Compute the bars
        bars = self.get_bars(track.audio_path, index=track_id)

        # Compute the barwise TF matrix
        barwise_tf_matrix = self.get_barwise_tf_matrix(track.audio_path, bars, index=track_id)

        # Get the annotationks
        # Using annotations from MIRDATA, which uses the AIST annotations. They are said to be worst than those of MIREX10, hence we use these ones instead
        # annotations_intervals = track.sections.intervals

        # Using the MIREX10 annotations
        annot_path_mirex = f"{self.datapath}/annotations/mirex10_sections-main/{track_id}.BLOCKS.lab"
        annotations_intervals = np.array(as_seg.data_manipulation.get_segmentation_from_txt(annot_path_mirex, "MIREX10"))[:,0:2]

        # Return the the bars, the barwise TF matrix and the annotations
        return track_id, bars, barwise_tf_matrix, annotations_intervals
    
    def format_dataset_from_mirdata_standards(self, file_extension = "mp3"):
        """      
        I found very confusing the way mirdata handles the paths, because songs are not named with their indexes.
        So I changed them to a unique list of files, ranging from 1 to 100.
        You can follow mirdata standards if you want, but you will have to modify the dataloader accordingly (in particular the datapaths in the tracks).
        Also, the cache may not work, please be careful.
        If you want to follow my version but have the data as mirdata, you can use the following function to copy the audio files to the right location.
        CAREFUL: It is not extensively tested though, so don't delete the original files.

        Parameters
        ----------
        file_extension : string
            The extension of the audio files.
            Default is "mp3".
        """
        def _filename_as_RWC(val):
            if type(val) == int:
                val = str(val)
            return "RM-P" + val.zfill(3)
        
        offset_previous_folders = 0
        for folder in sorted(os.listdir(f"{self.datapath}/audio")):
            try:
                count_files = 0
                for file in sorted(os.listdir(f"{self.datapath}/audio/{folder}")):

                    if file_extension not in file:
                        continue
                    new_number_file = int(file[:2]) + offset_previous_folders
                    src = f"{self.datapath}/audio/{folder}/{file}"
                    dest = f"{self.datapath}/audio/{_filename_as_RWC(new_number_file)}.{file_extension}"

                    print(f"Copying {src} to {dest}")
                    shutil.copy(src, dest)

                    count_files += 1

                offset_previous_folders += count_files
            
            except NotADirectoryError:
                pass

class SALAMIDataloader(MSABaseDataloader):
    """
    Dataloader for the SALAMI dataset.
    """

    name = "salami"

    def __init__(self, datapath, feature, cache_path = None, download=False, subset = None, sr=44100, hop_length = 32, subdivision = 96):
        """
        Constructor of the SALAMIDataloader class.

        Parameters
        ----------  
        Same then for BaseDataloader, with the addition of:

        datapath : string
            The path to the dataset.
        download : bool
            If True, download the dataset using mirdata.
            The default is False.
        subset : string
            The subset of the dataset to use. Can be "train", "test" or "debug".
        """
        super().__init__(feature=feature, cache_path=cache_path, sr=sr, hop_length=hop_length, subdivision=subdivision)
        
        # self.dataset_name = "salami"

        self.datapath = datapath
        salami = mirdata.initialize('salami', data_home = datapath)
        if download:
            salami.download()            
        self.all_tracks = salami.load_tracks()
        self.indexes = salami.track_ids

        self.subset = subset
        if subset is not None:
            train_indexes, test_indexes = self.split_training_test()
            if subset == "train":
                self.indexes = train_indexes
            elif subset == "test":
                self.indexes = test_indexes
            elif subset == "debug":
                self.indexes = test_indexes[:4]
            else:
                raise ValueError("Subset should be either 'train' or 'test'")

    def __getitem__(self, index):
        """
        Return the data of the index-th track.
        """
        # Parsing through files ordered with self.indexes
        track_id = self.indexes[index]
        track = self.all_tracks[track_id]

        try:           
            # Compute the bars
            bars = self.get_bars(track.audio_path, index=track_id)

            # Compute the barwise TF matrix
            barwise_tf_matrix = self.get_barwise_tf_matrix(track.audio_path, bars, index=track_id)

            # Get the annotations
            dict_annotations = self.get_annotations(track)

            # Return the the bars, the barwise TF matrix and the annotations
            return track_id, bars, barwise_tf_matrix, dict_annotations
    
        except FileNotFoundError:
            print(f'{track_id} not found.')
            return track_id, None, None, None
            # raise FileNotFoundError(f"Song {track_id} not found, normal ?") from None
            
    def get_annotations(self, track):
        """
        Return the annotations of the track, in the form of a dict.
        It returns the annotations of the first annotator, and if available, the annotations of the second annotator.
        It returns both levels of annotations (upper and lower) for each annotator.
        """
        dict_annotations = {}
        try: 
            # Trying to get the first annotator
            dict_annotations["upper_level_annotations"] = track.sections_annotator_1_uppercase.intervals
            dict_annotations["lower_level_annotations"] = track.sections_annotator_1_lowercase.intervals
            try: # Trying to load the second annotator
                dict_annotations["upper_level_annotations_2"] = track.sections_annotator_2_uppercase.intervals
                dict_annotations["lower_level_annotations_2"] = track.sections_annotator_2_lowercase.intervals
                dict_annotations["annot_number"]  = 2
            except AttributeError: # Only the first annotator was loaded
                dict_annotations["annot_number"]  = 1
        except AttributeError:
            try:
                # Trying to get the second annotator (no first one)
                dict_annotations["upper_level_annotations"] = track.sections_annotator_2_uppercase.intervals
                dict_annotations["lower_level_annotations"] = track.sections_annotator_2_lowercase.intervals
                dict_annotations["annot_number"]  = 1
            except AttributeError:
                raise AttributeError(f"No annotations found for {track.track_id}")
        
        return dict_annotations
    
    def get_this_set_annotations(self, dict_annotations, annotation_level = "upper", annotator = 1):
        """
        Return a particular set of annotations from all the annotations.
        """
        if annotator == 1:
            if annotation_level == "upper":
                annotations = dict_annotations["upper_level_annotations"]
            elif annotation_level == "lower":
                annotations = dict_annotations["lower_level_annotations"]
            else:
                raise ValueError("Invalid annotation level")
        elif annotator == 2:
            assert dict_annotations["annot_number"] == 2, "No second annotator found."
            if annotation_level == "upper":
                annotations = dict_annotations["upper_level_annotations"]
            elif annotation_level == "lower":
                annotations = dict_annotations["lower_level_annotations"]
            else:
                raise ValueError("Invalid annotation level")
        # elif annotator == "both":
        #     assert dict_annotations["annot_number"] == 2, "No second annotator found."
        #     annotations = dict_annotations["upper_level_annotations"] + dict_annotations["upper_level_annotations_2"]
        else:
            raise ValueError("Invalid annotator number")
        return annotations

    def split_training_test(self):
        """
        Split the dataset in training and test set.
        """
        indexes_train = []
        indexes_test = []
        for track_id in self.indexes:
            track = self.all_tracks[track_id]
            try:
                track.sections_annotator_1_uppercase.intervals
                track.sections_annotator_2_uppercase.intervals
                indexes_test.append(track_id)
            except AttributeError:
                indexes_train.append(track_id)
        return indexes_train, indexes_test
    
    def score_flat_segmentation(self, segments, dict_annotations, annotation_level = "upper", annotator = 1):
        """
        Score a flat segmentation.
        """
        if annotator == "both":
            assert dict_annotations["annot_number"] == 2, "No second annotator found."
            score_annot_1 = self.score_flat_segmentation(segments, dict_annotations, annotation_level = annotation_level, annotator = 1)
            score_annot_2 = self.score_flat_segmentation(segments, dict_annotations, annotation_level = annotation_level, annotator = 2)
            return score_annot_1, score_annot_2
        
        annotations = self.get_this_set_annotations(dict_annotations, annotation_level = annotation_level, annotator = annotator)
        return super().score_flat_segmentation(segments, annotations)
        
    def score_flat_segmentation_twolevel(self, segments_upper_level, segments_lower_level, dict_annotations, annotator = 1):
        """
        Score a flat segmentation at both levels of annotations.
        """
        score_upper_level = self.score_flat_segmentation(segments_upper_level, dict_annotations, annotation_level = "upper", annotator = annotator)
        score_lower_level = self.score_flat_segmentation(segments_lower_level, dict_annotations, annotation_level = "lower", annotator = annotator)
        return score_upper_level, score_lower_level
    
    def score_flat_segmentation_twolevel_best_of_several(self, list_segments_upper_level, list_segments_lower_level, dict_annotations, annotator = 1):
        """
        Score a flat segmentation at both levels of annotations, and return the best score from the different annotators.
        """
        assert annotator != "both", "Not implemented yet"
        stack_upper_scores = -np.inf * np.ones((len(list_segments_upper_level),2,3))
        for idx, segments in enumerate(list_segments_upper_level):
            stack_upper_scores[idx] = self.score_flat_segmentation(segments, dict_annotations, annotation_level = "upper", annotator = annotator)
        idx_close = np.argmax(stack_upper_scores[:,0,2]) # Selecting best f measure at 0.5s
        idx_large = np.argmax(stack_upper_scores[:,1,2]) # Selecting best f measure at 3s
        score_upper_level = (stack_upper_scores[idx_close,0,:], stack_upper_scores[idx_large,1,:])

        stack_lower_scores = -np.inf * np.ones((len(list_segments_lower_level),2,3))
        for idx, segments in enumerate(list_segments_lower_level):
            stack_lower_scores[idx] = self.score_flat_segmentation(segments, dict_annotations, annotation_level = "lower", annotator = annotator)
        idx_close = np.argmax(stack_lower_scores[:,0,2]) # Selecting best f measure at 0.5s
        idx_large = np.argmax(stack_lower_scores[:,1,2]) # Selecting best f measure at 3s
        score_lower_level = (stack_lower_scores[idx_close,0,:], stack_lower_scores[idx_large,1,:])

        return score_upper_level, score_lower_level

    def get_sizes_of_annotated_segments(self, annotation_level = "upper", annotator = 1, plot = False):
        """
        Return the lengths of the annotated segments.
        """
        lengths = []
        for track_id in self.indexes:
            track = self.all_tracks[track_id]

            try:           
                # Compute the bars
                bars = self.get_bars(track.audio_path, index=track_id)

                # Get the annotations
                dict_annotations = self.get_annotations(track)

                annotations = self.get_this_set_annotations(dict_annotations, annotation_level = annotation_level, annotator = annotator)

                barwise_annot = as_seg.data_manipulation.frontiers_from_time_to_bar(np.array(annotations)[:,1], bars) # Convert the annotations from time to bar
                for i in range(len(barwise_annot) - 1):
                    lengths.append(barwise_annot[i+1] - barwise_annot[i]) # Store the length of the annotated segment
        
            except FileNotFoundError:
                print(f'{track_id} not found.')
                # raise FileNotFoundError(f"Song {track_id} not found, normal ?") from None

        if plot:
            as_seg.model.common_plot.plot_lenghts_hist(lengths)
        return lengths
        
    # def format_dataset(self, path_audio_files): # TODO
        # # Copy audio files to the right location.
        # # Suppose that the audio files are all in the same folder
        # for track_num in range(len(self.all_tracks)):
        #     track_idx = self.indexes[track_num]
        #     song_file_name = self.all_tracks[track_idx].audio_path.split('/')[-1]
        #     src = f"{path_audio_files}/{song_file_name}" # May change depending on your file structure
        #     dest = self.all_tracks[track_idx].audio_path
        #     pathlib.Path(dest).parent.absolute().mkdir(parents=True, exist_ok=True)
        #     shutil.copy(src, dest)
    
class BeatlesDataloader(MSABaseDataloader):
    """
    Dataloader for the Beatles dataset.
    """
    name = "beatles"
    def __init__(self, datapath, feature, cache_path = None, download=False, sr=44100, hop_length = 32, subdivision = 96):
        """
        Constructor of the BeatlesDataloader class.

        Parameters
        ----------
        Same then for BaseDataloader, with the addition of:

        datapath : string
            The path to the dataset.
        download : bool
            If True, download the dataset using mirdata.
            The default is False
        """
        super().__init__(feature, cache_path, sr, hop_length, subdivision)
        self.datapath = datapath
        beatles = mirdata.initialize('beatles', data_home = datapath)
        if download:
            beatles.download()            
        self.all_tracks = beatles.load_tracks()
        self.indexes = beatles.track_ids

    def __getitem__(self, index):
        """
        Return the data of the index-th track.
        """
        track_id = self.indexes[index]
        track = self.all_tracks[track_id]

        # Compute the bars
        bars = self.get_bars(track.audio_path, index=track_id)

        # Compute the barwise TF matrix
        barwise_tf_matrix = self.get_barwise_tf_matrix(track.audio_path, bars, index=track_id)

        # Get the annotationks
        annotations_intervals = track.sections.intervals

        # Return the the bars, the barwise TF matrix and the annotations
        return track_id, bars, barwise_tf_matrix, annotations_intervals
    
    def __len__(self):
        """
        Return the number of tracks in the dataset.
        """
        return len(self.all_tracks) # Why not just len(indexes, an in the base class?)

# if __name__ == "__main__":
#     rwcpop = RWCPopDataloader('/home/a23marmo/datasets/rwcpop', feature = "mel", cache_path = "/home/a23marmo/datasets/rwcpop/cache")
#     # rwcpop.format_dataset('/home/a23marmo/Bureau/Audio samples/rwcpop/Entire RWC')
    
#     print(len(rwcpop))
#     for track_id, bars, barwise_tf_matrix, annotations in rwcpop:
#         print(track_id)

#     # salami = SALAMIDataloader('/home/a23marmo/datasets/salami', feature = "mel", cache_path = '/home/a23marmo/datasets/salami/cache', subset = "train")

#     # for track_id, bars, barwise_tf_matrix, dict_annotations in salami:
#     #     try:
#     #         print(track_id)
#     #     except FileNotFoundError:
#     #         print(f'{track_id} not found.')
