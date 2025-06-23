"""
Code defining the Music Structure Analysis task, using the CBM algorithm [1], [2, Chap 3].
Future algorithms could come from the MSAF toolbox [3].

Functions "predict" and "score" are inspired from scikit-learn, and are used as standards here to compute tasks.

Metrics are computed using the "mir_eval" toolbox [4], and are based on the F-measure, Precision and Recall, with tolerances being 0.5s and 3s, as in MIREX standards [2].

References:
[1] Marmoret, A., Cohen, J. E., & Bimbot, F. (2023). Barwise Music Structure Analysis with the Correlation Block-Matching Segmentation Algorithm. Transactions of the International Society for Music Information Retrieval (TISMIR), 6(1), 167-185.

[2] Marmoret, A. (2022). Unsupervised Machine Learning Paradigms for the Representation of Music Similarity and Structure (Doctoral dissertation, Université Rennes 1).

[3] Nieto, O., & Bello, J. P. (2015). Msaf: Music structure analytis framework. In Proceedings of 16th International Society for Music Information Retrieval Conference (ISMIR 2015).

[4] Raffel, C., McFee, B., Humphrey, E. J., Salamon, J., Nieto, O., Liang, D., & Ellis, D. P. W. (2014). mir_eval: A transparent implementation of common MIR metrics. In Proceedings of the 15th International Society for Music Information Retrieval Conference (ISMIR).
"""

from nmf_audio_benchmark.tasks.base_task import *
import tqdm

# The routine to compute the segmentation. It will be called to run the benchmark.
def compute_segmentation(dataset, nmf, segmentation_algorithm):
    """
    Actually compute the segmentation of the dataset using the NMF and the segmentation algorithm.
    """
    # Empty lists for the scores
    all_scores_05 = []
    all_scores_3 = []

    # Iterate over all the songs in the dataset
    for idx_song in tqdm.tqdm(range(len(dataset))):
        # One song
        track_id, bars, barwise_tf_matrix, annotations_intervals = dataset[idx_song]

        # Compute NMF
        W, H = nmf.run(data=barwise_tf_matrix)

        # Predict the boundaries and convert them in seconds
        segments_estimated = segmentation_algorithm.predict_in_seconds(W, bars)
        
        # file_name = f"song-{track_id}_method{nmf_object.name}_segmentation_alg-{segmentation_algorithm.algorithm_name}.txt".replace(" ", "_")
        # dataset.save_segments(segments_estimated, file_name) # TODO

        # Compute the scores and store them
        tol_05, tol_3 = segmentation_algorithm.score(segments_estimated, annotations_intervals)
        # all_scores_05.append(tol_05)
        # all_scores_3.append(tol_3)
        all_scores_05.append(tol_05[2]) # Keep only the F measure
        all_scores_3.append(tol_3[2]) # Keep only the F measure

    to_return = {
        "scores_05": all_scores_05,
        "scores_3": all_scores_3
    }

    return to_return

# %% One particular algorithm for the MSA task, the CBM algorithm.

import as_seg.CBM_algorithm as CBM_algorithm
import as_seg.autosimilarity_computation as as_comp
import as_seg.data_manipulation as dm

class MSA_CBMEstimator(BaseTask):
    """
    Class for the CBM algorithm [1]. Inspired from the scikit-learn API: https://scikit-learn.org/stable/auto_examples/developing_estimators/sklearn_is_fitted.html, Author: Kushan <kushansharma1@gmail.com>, License: BSD 3 clause
    """
    def __init__(self, similarity_function="cosine", max_size=32, penalty_weight=1, penalty_func="modulo8", bands_number=7):
        """
        Constructor of the CBM estimator.

        Parameters
        ----------
        similarity_function : string, optional
            The similarity function to use for computing the autosimilarity.
            The default is "cosine".
        max_size : integer, optional
            The maximal size of segments.
            The default is 32.
        penalty_weight : float, optional
            The ponderation parameter for the penalty function.
            The default is 1.
        penalty_func : string, optional
            The type of penalty function to use.
            The default is "modulo8".
        bands_number : positive integer or None, optional
            The number of bands in the kernel.
            For the full kernel, bands_number must be set to None
            (or higher than the maximal size, but cumbersome)
            See [1] for details.
            The default is 7.
        """
        self.similarity_function = similarity_function
        self.max_size = max_size
        self.penalty_weight = penalty_weight
        self.penalty_func = penalty_func
        self.bands_number = bands_number
        self.algorithm_name = "CBM"

    def predict(self, barwise_features):
        """
        Perform Predictions
        """
        ssm_matrix = as_comp.switch_autosimilarity(barwise_features, similarity_type=self.similarity_function)
        segments = CBM_algorithm.compute_cbm(ssm_matrix, max_size=self.max_size, penalty_weight=self.penalty_weight, 
                               penalty_func=self.penalty_func, bands_number = self.bands_number)[0]
        return segments
    
    def predict_in_seconds(self, barwise_features, bars):
        """
        Perform Predictions, and convert the segments from bars to seconds.
        """
        segments = self.predict(barwise_features)
        return dm.segments_from_bar_to_time(segments, bars)
    
    def predict_in_seconds_this_autosimilarity(self, ssm_matrix, bars):
        """
        Perform Predictions on a given autosimilarity matrix, and convert the segments from bars to seconds.
        """
        segments = CBM_algorithm.compute_cbm(ssm_matrix, max_size=self.max_size, penalty_weight=self.penalty_weight, 
                               penalty_func=self.penalty_func, bands_number = self.bands_number)[0]
        return dm.segments_from_bar_to_time(segments, bars)

    def score(self, predictions, annotations, trim=False):
        """
        Compute the score of the predictions.
        """
        close_tolerance = dm.compute_score_of_segmentation(annotations, predictions, window_length=0.5, trim=trim)
        large_tolerance = dm.compute_score_of_segmentation(annotations, predictions, window_length=3, trim=trim)
        return close_tolerance, large_tolerance

    def compute_task_on_dataset(self, dataset, nmf):
        """
        Compute the task on the dataset, using the NMF and the task algorithm.
        """
        return compute_segmentation(dataset, nmf, self)