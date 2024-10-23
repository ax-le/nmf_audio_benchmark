"""
Created on June 2024

@author: a23marmo

Base code for the task of counting the number of species in a signal, using NMF. 
"""
from nmf_audio_benchmark.tasks.base_task import *

import numpy as np
import sklearn.feature_selection
from sklearn.cluster import DBSCAN # HDBSCAN or OPTICS also possible, to test
from sklearn.metrics import accuracy_score

class SourceCountEstimator(BaseTask):
    def __init__(self, *, var_divide=10, eps=0.8, metric='correlation'):
        """
        Initializes the estimator object.
        
        Parameters
        ----------
        var_divide : float
            Variance divide parameter for considering some components as noise.
            If the variance of a component is lower than the mean variance divided by var_divide, it is considered noise.
        eps : float
            Epsilon parameter for the DBSCAN clustering algorithm.
        metric : str
            Metric for the DBSCAN clustering algorithm.
        """

        self.var_divide = var_divide
        self.eps = eps
        self.metric = metric

    def predict(self, W, H):
        """
        Estimates the number of species in the signal, based on the H matrix of NMF.
        This function is used to return the number of species in the current audio file
        
        Parameters
        ----------
        W, H : np.array
            The W and H matrices of the NMF.

        Important object parameters
        ---------------------------
        var_divide : float
            Variance divide parameter for considering some components as noise.
            If the variance of a component is lower than the mean variance divided by var_divide, it is considered noise.
        eps : float
            Epsilon parameter for the DBSCAN clustering algorithm.
        metric : str
            Metric for the DBSCAN clustering algorithm.

        Returns
        ------- 
        number_species : int
            Estimated number of species in the signal.
        """

        # First, remove all features with a variance lower than the mean variance divided by 10
        H_cropped = threshold_H(H, var_divide=self.var_divide)

        # Then, we count the number of species
        estimated_species_number = DBSCAN_count(H_cropped, eps=self.eps, metric=self.metric)

        return estimated_species_number
    
    def predict_all_H(self, all_H):
        """
        Predicts the number of species in all H matrices.

        Parameters
        ----------
        all_H : list of np.array
            List of the H matrices of the NMF.

        Returns
        -------
        all_estimations : list of int
            List of the estimations of the number of species in the signals.
        """
        # Loop on all the H matrices.
        all_estimations = []
        for H in all_H:
            estimation = self.predict(None, H)
            all_estimations.append(estimation)

        return all_estimations

    def score(self, estimations, annotations):
        """
        Computes the accuracy of the estimator.

        Parameters
        ----------
        estimations : list of int
            List of the estimations of the number of species in the signals.    
        annotations : list of int
            List of the annotations of the number of species in the signals.

        Returns
        -------
        accuracy : float
            Accuracy of the estimator.
        """
        return accuracy_score(annotations, estimations)

def threshold_H(H, var_divide=10):
    """
    Thresholds the H matrix of NMF, removing the components with a variance lower than the mean variance divided by var_divide.
    """
    var = np.var(H, axis=1)
    threshold = np.mean(var)/var_divide

    # Maybe tune the threshold not to be too low, it can lead to massive data loss
    thresh = sklearn.feature_selection.VarianceThreshold(threshold=threshold)
    H_cropped = thresh.fit_transform(H.T).T

    return H_cropped

def DBSCAN_count(H, eps = 0.7, metric = 'correlation'):
    """
    Counts the number of clusters in the H matrix of NMF, using DBSCAN.
    """
    db = DBSCAN(eps=eps, min_samples=1, metric=metric)
    db.fit(H)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    return n_clusters_

def compute_difference(estimation, annotation):
    """
    Computes the difference between the estimation and the annotation.
    """
    return np.abs(estimation - annotation)
