"""
Code defining the Music Structure Analysis task, using the CBM algorithm [1], [2, Chap 3].
Future algorithms could come from the MSAF toolbox [3].

Functions "predict" and "score" are inspired from scikit-learn, and are used as standards here to compute tasks.

Metrics are computed using the "mir_eval" library, and are based on the F-measure, Precision and Recall, with tolerances being 0.5s and 3s, as in MIREX standards [2].

References:
[1] Marmoret, A., Cohen, J. E., & Bimbot, F. (2023). Barwise Music Structure Analysis with the Correlation Block-Matching Segmentation Algorithm. Transactions of the International Society for Music Information Retrieval (TISMIR), 6(1), 167-185.
[2] Marmoret, A. (2022). Unsupervised Machine Learning Paradigms for the Representation of Music Similarity and Structure (Doctoral dissertation, Université Rennes 1).
[3] Nieto, O., & Bello, J. P. (2015). Msaf: Music structure analytis framework. In Proceedings of 16th International Society for Music Information Retrieval Conference (ISMIR 2015).
"""
import as_seg.CBM_algorithm as CBM_algorithm
import as_seg.autosimilarity_computation as as_comp
import as_seg.data_manipulation as dm

class CBMEstimator():
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