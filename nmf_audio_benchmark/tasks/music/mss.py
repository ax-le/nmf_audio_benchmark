"""
Code used to compute the Music Source Separation.
Details for Music Source Separation with NMF can be found in [1, Chap. 8.1].

The W matrix is clustered in sources using the MFCC representation of the frequency templates, following [2].

Functions "predict" and "score" are inspired from scikit-learn, and are used as standards here to compute tasks.

Metrics used here are SI-SDR and SNR [3].
Code was found at code obtained from https://github.com/sigsep/bsseval/issues/3

One coud also use the BSS eval metrics, as in [4].

References:
[1] Vincent, E., Virtanen, T., & Gannot, S. (Eds.). (2018). Audio source separation and speech enhancement. John Wiley & Sons.
[2] Barker, T., & Virtanen, T. (2013, August). Non-negative tensor factorisation of modulation spectrograms for monaural sound source separation. In INTERSPEECH (Vol. 2813, pp. 827-831).
[3] Le Roux, J., Wisdom, S., Erdogan, H., & Hershey, J. R.(2019, May). SDR-half-baked or well done?. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 626-630). IEEE.
[4] Vincent, E., Gribonval, R., & Févotte, C. (2006). Performance measurement in blind audio source separation. IEEE transactions on audio, speech, and language processing, 14(4), 1462-1469.
"""
from nmf_audio_benchmark.tasks.base_task import *

import numpy as np
import mir_eval
import librosa
from sklearn.cluster import KMeans
import base_audio.spectrogram_to_signal as spectrogram_to_signal

import math
import warnings

class MusicSourceSeparation(BaseTask):
    """
    Class to compute source separation, based on NMF.
    Inspired from the scikit-learn API: https://scikit-learn.org/stable/auto_examples/developing_estimators/sklearn_is_fitted.html, Author: Kushan <kushansharma1@gmail.com>, License: BSD 3 clause
    """
    def __init__(self, feature_object, nb_sources=4, phase_retrieval="original_phase"):
        self.feature_object = feature_object
        self.nb_sources = nb_sources
        self.phase_retrieval = phase_retrieval

    def predict(self, W, H, phase=None):
        if self.phase_retrieval == "original_phase":
            assert phase is not None, "Phase must be provided if phase_retrieval is set to original_phase"
        return estimate_sources(W, H, feature_object=self.feature_object, nb_sources=self.nb_sources, phase_retrieval=self.phase_retrieval, phase=phase)

    def score(self, estimations, annotations):
        si_sdr, idx_argmax = evaluate_si_sdr(annotations, estimations, scaling=True)
        snr, _ = evaluate_si_sdr(annotations, estimations, scaling=False)
        return si_sdr, snr, idx_argmax

# %% Compute sources from NMF
def estimate_sources(W, H, feature_object, nb_sources=4, phase_retrieval = "original_phase", phase=None):
    """
    Function which estimate all sources from W and H matrices.

    Phase retrieval can be done in two ways:
    - original_phase: The original phase is used to reconstruct the signal.
    - griffin_lim: The Griffin-Lim algorithm is used to estimate the phase information, initialized at random.

    """
    # Estimate the spectrograms of each source
    all_specs = estimate_spectrograms(W, H, nb_sources=nb_sources)
    all_signals = []

    # Reconstruct the signals from the spectrograms
    for spec in all_specs:
        all_signals.append(spectrogram_to_signal.spectrogram_to_audio_signal(spec, feature_object, phase_retrieval = phase_retrieval, original_phase=phase))
    return np.array(all_signals)

def estimate_spectrograms(W, H, nb_sources=4):
    """
    Compute the spectrograms of the sources from the W and H matrices.
    """
    # Obtain the source label of each frequency template
    source_labels = cluster_sources(W, n_clusters=nb_sources)
    all_specs = np.zeros((len(np.unique(source_labels)), W.shape[0], H.shape[1]))

    # Assign the frequency templates to the sources
    for idx_factor, current_source_idx in enumerate(source_labels):
        all_specs[current_source_idx] += np.outer(W[:,idx_factor], H[idx_factor])

    return all_specs

def estimate_spectrograms_no_clustering(W, H):
    """
    Compute the spectrograms of the sources from the W and H matrices, considering that each column of W is a different source.
    """
    all_specs = np.array([np.outer(W[:,idx_factor], H[idx_factor]) for idx_factor in range(W.shape[1])])
    return all_specs

def cluster_sources(W, n_clusters=4):
    """
    Estimate which frequency templates are associated to the same sources.
    The sources estimation, from frequency templates, is obtained by clustering the MFCC representation of each template, as in [2].

    The MFCC feature is commonly used in audio processing to represent the timbre of the signal, which seems practical here to cluster the sources according to the instruments.
    
    Different strategies could be used, for instance clustering based on the time activations, as in [2].
    """
    # Compue the MFCC
    log_mel_W = librosa.amplitude_to_db(librosa.feature.melspectrogram(S=W))
    mfcc = librosa.feature.mfcc(S=log_mel_W)

    # Cluster the sources using scikit learn kmeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(mfcc.T)

    if len(np.unique(kmeans.labels_)) < n_clusters: # May happen with failed convergence
        warnings.warn(f"Failed convergence in KMeans, the algorithm found only {len(np.unique(kmeans.labels_))} clusters instead of {n_clusters}")
    return kmeans.labels_

# %% Evaluation functions
def average_scores_sourcewise(scores, estimated_source_idxs, stems_labels):
    """
    Compute the average scores source-wise.
    Suppose that you have already computed the SI-SDR (see below) for each estimated source.

    Parameters
    ----------
    scores : list
        The list of scores for each estimated source.
    estimated_source_idxs : list
        The index indicating which source is estimated.
    stems_labels : list
        The labels of the sources.

    Returns
    -------
    dict_si_sdr : dict
        The dictionary containing the average scores for each source.
        Keys of the dictionary are the source labels.
    """
    # Initialize the final dictionary
    dict_si_sdr = {}
    
    # Parse all estimated sources indexes
    for i in range(len(estimated_source_idxs)):
        # Get the label of the source
        this_source_label = stems_labels[estimated_source_idxs[i]]

        if this_source_label not in dict_si_sdr: # Initialize the list if the source was not previously found
            dict_si_sdr[this_source_label] = []
        dict_si_sdr[this_source_label].append(scores[i]) # Store the results
    
    # Average the results source-wise
    for key in dict_si_sdr:
        dict_si_sdr[key] = np.mean(dict_si_sdr[key], axis=0)
    
    return dict_si_sdr

def evaluate_si_sdr(reference_signals, estimated_signals, scaling=True):
    """
    Evaluate the SI-SDR of the estimated signals compared to the reference signals.
    SI-SDR is defined in [3].
    """
    if reference_signals.shape[1] > estimated_signals.shape[1]:
        reference_signals = reference_signals[:, :estimated_signals.shape[1]]
        
    all_si_sdr = []
    all_idx_max_sdr = []
    for i in range(len(estimated_signals)):
        si_sdr, argmax_si_sdr  = evalute_si_sdr_one_estimation(reference_signals, estimated_signals[i], scaling)
        all_si_sdr.append(si_sdr)
        all_idx_max_sdr.append(argmax_si_sdr)
    return all_si_sdr, all_idx_max_sdr

def evalute_si_sdr_one_estimation(reference_signals, estimated_signal, scaling=True):
    """
    Evaluate the SI-SDR of one estimated signal compared to the reference signals.
    # Credits to J. Le Roux (@Jonathan-LeRoux) et al., code adapted from https://github.com/sigsep/bsseval/issues/3
    """
    if (estimated_signal == 0).all():
        return -np.inf, None

    Rss= np.dot(reference_signals, reference_signals.transpose())

    all_si_sdr = []
    for j in range(reference_signals.shape[0]):
        this_source= reference_signals[j]
        if not (this_source == 0).all():
            if scaling:
                # get the scaling factor for clean sources
                scaling_factor= np.dot( this_source, estimated_signal) / Rss[j,j]
            else:
                scaling_factor= 1
            
            all_si_sdr.append(si_sdr(this_source, estimated_signal, scaling_factor))
    
    idx_max_sdr= np.argmax(all_si_sdr)
    return all_si_sdr[idx_max_sdr], idx_max_sdr

def si_sdr(ref_source, estimated_signal, scaling_factor):
    """
    Compute the SI-SDR between the reference source and the estimated signal.
    # Credits to J. Le Roux (@Jonathan-LeRoux) et al., code obtained from https://github.com/sigsep/bsseval/issues/3
    # Ref: Le Roux, J., Wisdom, S., Erdogan, H., & Hershey, J. R.(2019, May). SDR-half-baked or well done?. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 626-630). IEEE.
    """

    e_true= scaling_factor * ref_source
    e_res= estimated_signal - e_true

    Sss= (e_true**2).sum()
    Snn= (e_res**2).sum()

    SDR= 10 * math.log10(Sss/Snn)
    return SDR
    
def compute_bss_eval(audio_ref, audio_estimate):
    """
    Wrapper for the BSS eval metrics.
    # Ref: Vincent, E., Gribonval, R., & Févotte, C. (2006). Performance measurement in blind audio source separation. IEEE transactions on audio, speech, and language processing, 14(4), 1462-1469.
    """
    if (audio_estimate == 0).all():
        return -np.inf, -np.inf, -np.inf
    if audio_ref.shape[1] > audio_estimate.shape[1]:
        audio_ref = audio_ref[:, :audio_estimate.shape[1]]
    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(audio_ref, audio_estimate)
    return sdr, sir, sar