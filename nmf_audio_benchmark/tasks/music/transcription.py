"""
Code defining the Transcription task.
Music transcription using NMF is an old task, which originates from [1].
The task is to transcribe the music, i.e. to find the notes and their onset and offset times.

The task is divided into two parts:
- W to notes: Estimate the fundamental frequency of each note in the codebook.
- H to activations: Estimate the activations of the notes in the transcription.

Both parts are based on the NMF decomposition of the spectrogram of the music.
Both parts are implemented in a relatively naïve way, inspired from [2], and could/should be improved in the future.

TODO: better algorithms for W to notes and H to activations, e.g. using a more sophisticated pitch detection algorithm or a more sophisticated activation detection algorithm.

Functions "predict" and "score" are inspired from scikit-learn, and are used as standards here to compute tasks.

Metrics are computed using the "mir_eval" library, and are based on the F-measure, Precision and Recall, with the tolerance on the onset being 50ms.

References:
[1] Smaragdis, P., & Brown, J. C. (2003, October). Non-negative matrix factorization for polyphonic music transcription. In 2003 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (IEEE Cat. No. 03TH8684) (pp. 177-180). IEEE.

[2] Marmoret, A., Bertin, N., & Cohen, J. (2019). Multi-Channel Automatic Music Transcription Using Tensor Algebra. arXiv preprint arXiv:2107.11250.
"""
from nmf_audio_benchmark.tasks.base_task import *


import numpy as np
import librosa
from collections import defaultdict
import mir_eval
import tqdm
import math

import nmf_audio_benchmark.utils.errors as err
import nmf_audio_benchmark.utils.data_manipulation as dm
import nmf_audio_benchmark.tasks.generic.sound_event_detection as sed

import base_audio.signal_to_spectrogram as signal_to_spectrogram

# %% Scripts to compute the transcription
def compute_transcription(dataset, nmf, transcription_algorithm, time_tolerance=0.1):
    """
    Compute the transcription for all the songs in the dataset, with this particular set of parameters.
    """
    # Empty dicts for the scores
    all_accuracies = {}
    all_f1 = {}

    # Iterate over all the songs in the dataset
    for idx_song in tqdm.tqdm(range(len(dataset))):
        # One song
        track_id, spectrogram, annotations = dataset[idx_song]

        # Compute NMF
        W, H = nmf.run(data=spectrogram, feature_object=dataset.feature_object)

        # Compute the transcription
        estimated_transcription = transcription_algorithm.predict(W, H)

        # Compute the metrics
        f_mes, accuracy = transcription_algorithm.score(estimated_transcription, annotations, time_tolerance=time_tolerance)

        # Compute the metrics
        all_f1[track_id] = f_mes
        all_accuracies[track_id] = accuracy

    to_return = {
        "f1": all_f1,
        "accuracy": all_accuracies
    }

    return to_return

# TODO: use Optuna instead
# def compute_transcription_with_cross_validation(dataset, nmf, default_transcription_algorithm, param_grid, cv=4, time_tolerance=0.1):
#     """
#     Cross validation to find the best parameters for the transcription algorithm.
#     A second option could be hyperparameter optimization, using hyperopt for example, but it may be considered as "cheating".

#     It would have been easier to use the GridSearchCV from scikit-learn, but it is cumbersome to adapt the current code to it.
#     In particular, it requires to have a single fit for the whole dataset, which is not the mentality of the current code.
#     """

#     # Empty lists for the scores
#     all_results = []

#     param_combinations = list(hyperparams_helper.generate_param_grid(param_grid))

#     def evaluate_one_set_params(W, H, annotations, params):
#         # Clone the default transcription algorithm and update the parameters
#         transcription_algorithm = clone(default_transcription_algorithm)
#         transcription_algorithm.update_params(params)

#         # Compute the transcription with the new parameters
#         estimated_transcription = transcription_algorithm.predict(W, H)
#         f_mes, accuracy = transcription_algorithm.score(estimated_transcription, annotations, time_tolerance=time_tolerance)
#         return (f_mes, accuracy)
    
#     # Iterate over all the songs in the dataset
#     for idx_song in tqdm.tqdm(range(len(dataset))):
#         # One song
#         track_id, spectrogram, annotations = dataset[idx_song]

#         # Compute NMF
#         W, H = nmf.run(data=spectrogram, feature_object=dataset.feature_object)

#         # Compute the transcription for all params
#         all_results.append([evaluate_one_set_params(W, H, annotations, params) for params in param_combinations])

#     # Find the best results in the cross validation scheme
#     final_results, final_best_params = hyperparams_helper.find_best_results_in_cv_scheme(all_results, cv, param_combinations, nb_metrics=2, test_metric_idx=1)

#     # Return the final results, obtained via cross-validation
#     return final_results, final_best_params

class Transcription(BaseTask):
    """
    Class for the Transcription algorithm. Inspired from the scikit-learn API: https://scikit-learn.org/stable/auto_examples/developing_estimators/sklearn_is_fitted.html, Author: Kushan <kushansharma1@gmail.com>, License: BSD 3 clause
    """
    def __init__(self, feature_object, salience_shift_autocorrelation = 0.3,
                 H_process = "thresholding", threshold = 0.01, smoothing_window = 5, H_normalization = None,
                 ruptures_detector_model = 'normal', pelt_penalty = 0, pelt_min_size = 1,
                 energy_filtering_criterion = None, energy_filter_stat = "mean_energy", percentile_detection = 90, energy_filter_threshold = None,
                 time_tol = None, min_duration = None, max_duration = None,
                 verbose = False):
        """
        Constructor of the Transcription estimator.

        Temporal detection of the note activations (H matrix post-processing) is delegated to
        nmf_audio_benchmark.tasks.ecoacoustics.sound_event_detection.detect, so the same three
        detectors ("thresholding", "pelt" and "window") and the same post-processing options
        (energy filtering, gap merging, duration filtering) are available here.

        Parameters
        ----------
        feature_object : object
            The object containing the feature parameters of the audio signal.
        salience_shift_autocorrelation : float, optional
            The threshold for the autocorrelation of the waveform to detect the fundamental frequency.
            Below this threshold, the pitch is considered as invalid.
            The default is 0.3.
        H_process : {"thresholding", "pelt", "window"}, optional
            The detector used to find the note segments in each row of H. The default is "thresholding".
        threshold : float, optional
            The threshold to detect the presence of a note in the activations (used by "thresholding" and, as
            the ruptures ``epsilon``, by "window"). The default is 0.01.
        smoothing_window : integer, optional
            The number of frames to average the activation value in order to detect the presence of a note
            (used by "thresholding"), or the window width (used by "window"). The default is 5.
        H_normalization : {None, "max", "row_max", "mean", "row_mean", "l2", "row_l2"}, optional
            How H is normalized before detection. The default is None (no normalization).
        ruptures_detector_model : str, optional
            The ruptures cost model, used by "pelt" and "window". The default is 'normal'.
        pelt_penalty : float, optional
            The PELT penalty, used by "pelt". The default is 0.
        pelt_min_size : integer, optional
            The PELT minimal segment size, used by "pelt". The default is 1.
        energy_filtering_criterion : {None, "percentile", "threshold"}, optional
            If set, PELT segments are additionally selected by their energy. The default is None.
        energy_filter_stat : {"mean_energy", "cumulative_energy"}, optional
            The statistic used for the energy filtering. The default is "mean_energy".
        percentile_detection : float, optional
            The percentile used when energy_filtering_criterion is "percentile". The default is 90.
        energy_filter_threshold : float, optional
            The threshold used when energy_filtering_criterion is "threshold". Defaults to ``threshold`` when None.
        time_tol : float, optional
            If set, segments closer than time_tol (in seconds) are merged. The default is None.
        min_duration : float, optional
            Minimal duration (in seconds) of a note, used together with max_duration. The default is None.
        max_duration : float, optional
            Maximal duration (in seconds) of a note, used together with min_duration. The default is None.
        verbose : boolean, optional
            verbose mode. The default is False.
        """
        self.feature_object = feature_object
        self.salience_shift_autocorrelation = salience_shift_autocorrelation
        self.H_process = H_process
        self.threshold = threshold
        self.smoothing_window = smoothing_window
        self.H_normalization = H_normalization
        self.ruptures_detector_model = ruptures_detector_model
        self.pelt_penalty = pelt_penalty
        self.pelt_min_size = pelt_min_size
        self.energy_filtering_criterion = energy_filtering_criterion
        self.energy_filter_stat = energy_filter_stat
        self.percentile_detection = percentile_detection
        self.energy_filter_threshold = energy_filter_threshold
        self.time_tol = time_tol
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.verbose = verbose

    def predict(self, W, H):
        """
        Compute the transcription from the NMF decomposition of the spectrogram.
        """
        W_notes = W_to_notes(W=W, feature_object=self.feature_object, salience_shift_autocorrelation=self.salience_shift_autocorrelation, verbose = self.verbose)
        activations = H_to_activations(W_notes=W_notes, H=H, feature_object=self.feature_object,
                                       H_process=self.H_process, threshold=self.threshold, smoothing_window=self.smoothing_window, H_normalization=self.H_normalization,
                                       ruptures_detector_model=self.ruptures_detector_model, pelt_penalty=self.pelt_penalty, pelt_min_size=self.pelt_min_size,
                                       energy_filtering_criterion=self.energy_filtering_criterion, energy_filter_stat=self.energy_filter_stat,
                                       percentile_detection=self.percentile_detection, energy_filter_threshold=self.energy_filter_threshold,
                                       time_tol=self.time_tol, min_duration=self.min_duration, max_duration=self.max_duration,
                                       verbose = self.verbose)
        return activations
    
    def score(self, predictions, annotations, time_tolerance=0.1):
        """
        Compute the score of the predictions.
        """
        f_mes, accuracy = compute_scores(predictions, annotations, time_tolerance=time_tolerance)
        return f_mes, accuracy

    def compute_task_on_dataset(self, dataset, nmf, time_tolerance=0.1, verbose=False):
        return compute_transcription(dataset=dataset, nmf=nmf, transcription_algorithm=self, time_tolerance=time_tolerance)
    
    def update_params(self, param_grid):
        """
        Update the parameters of the model (e.g. threshold and smoothing window)
        """
        for key, value in param_grid.items():
            setattr(self, key, value)


# %% W to notes
def W_to_notes(W, feature_object, pitch_min = 50, pitch_max = 5000, salience_shift_autocorrelation = 0.3, verbose = True,
               pitch_method = "harmonic_sum", piano = True):
    """
    Estimate the MIDI value of each note in the codebook.

    Parameters
    ----------
    W : numpy array
        The W matrix of the NMF decomposition of the spectrogram.
    feature_object : object
        The object containing the feature parameters of the audio signal.
    pitch_min : integer, optional
        The minimal pitch value. The default is 50.
    pitch_max : integer, optional   
        The maximal pitch value. The default is 5000.
    salience_shift_autocorrelation : float, optional
        The threshold for the autocorrelation of the waveform to detect the fundamental frequency. 
        Below this threshold, the pitch is considered as invalid. Only used by pitch_method="legacy" on STFT.
        The default is 0.3.
    verbose : boolean, optional
        verbose mode. The default is True.
    pitch_method : {"harmonic_sum", "legacy"}, optional
        "harmonic_sum" : harmonic summation (see harmonic_sum_pitch), for STFT and CQT/VQT.
        "legacy" : the previous estimators, i.e. autocorrelation of the waveform for STFT and argmax for CQT/VQT.
        The default is "harmonic_sum".
    piano : boolean, optional
        For pitch_method="harmonic_sum" : whether the partials follow the inharmonic model of the piano strings
        (see harmonic_frequencies) or are exact multiples of f0. The default is True.
    """
    f0s = []
    for idx_col in range(0,W.shape[1]): # Fundamental frequency estimation of each atom of the codebook
        try:
            f0s.append(W_column_to_note(W[:,idx_col], feature_object, pitch_min = pitch_min, pitch_max = pitch_max, salience_shift_autocorrelation = salience_shift_autocorrelation,
                                        pitch_method = pitch_method, piano = piano))
        except ValueError as err:
            f0s.append(None)
            if verbose:
                print("Error in the " + str(idx_col) + "-th note-atom of the codebook: " + err.args[0])
    return f0s

def W_column_to_note(W_col, feature_object, pitch_min = 27, pitch_max = 4500, salience_shift_autocorrelation = 0.3, pitch_method = "harmonic_sum", piano = True):
    """
    Estimate the fundamental frequency of a note from the column of the codebook.
    27 and 4500 Hz correspond broadly to the range of the piano keyboard.
    
    Returns the note in the MIDI scale.

    Parameters
    ----------
    W_col : numpy array
        One column of the W matrix of the NMF decomposition of the spectrogram. 
    feature_object : object
        The object containing the feature parameters of the audio signal.
    pitch_min : integer, optional
        The minimal pitch value. The default is 27.
    pitch_max : integer, optional
        The maximal pitch value. The default is 4500.
    salience_shift_autocorrelation : float, optional
        The threshold for the autocorrelation of the waveform to detect the fundamental frequency. 
        Below this threshold, the pitch is considered as invalid. Only used by pitch_method="legacy" on STFT.
        The default is 0.3.
    pitch_method : {"harmonic_sum", "legacy"}, optional
        See W_to_notes. The default is "harmonic_sum".
    piano : boolean, optional
        See W_to_notes. The default is True.
    """
    if pitch_method not in ["harmonic_sum", "legacy"]:
        raise ValueError(f"Unknown pitch_method: {pitch_method}")

    if feature_object.feature in ["stft", "cqt", "vqt"] and pitch_method == "harmonic_sum":
        found_pitch = harmonic_sum_pitch(W_col, feature_object, piano=piano)
        if found_pitch < pitch_min: # A lower bound for the frequency range
            raise ValueError('The pitch is anormally low')
        elif found_pitch > pitch_max:
            raise ValueError('The pitch is anormally high')
        else:
            return freq_to_midi(found_pitch)

    elif feature_object.feature == "stft":

        # Trying to find the maximum of autocorrelation of the waveform, which corresponds to the fundamental frequency in harmonic signals.
        found_pitch = autocorrelate_signal(W_col, feature_object, salience_shift_autocorrelation)
        if found_pitch is None: # It means that the autocorrelation was not strong enough to be considered valid.
            ## Trying to find the maximal autocorrelation on the frequency decomposition directly.
            # found_pitch_idx = autocorrelate_freq(W_col, salience_shift_autocorrelation)
            # if found_pitch_idx is None: # It means that the autocorrelation was not strong enough to be considered valid.
            raise ValueError('This spectrogram is irrelevant')
            
            # If it is valid, we can compute the frequency from the index of the maximum of the autocorrelation
            freqs = librosa.fft_frequencies(sr=feature_object.sr, n_fft=feature_object.n_fft)
            found_pitch = freqs[found_pitch_idx]

        if found_pitch < pitch_min: # A lower bound for the frequency range, must be calculated from the size of the window
            raise ValueError('The pitch is anormally low')

        elif found_pitch > pitch_max:
            raise ValueError('The pitch is anormally high')

        else:
            return dm.freq_to_midi(found_pitch)

    elif feature_object.feature == "mel":

        feat_obj = signal_to_spectrogram.FeatureObject(feature_object.sr, "stft", hop_length=(feature_object.n_fft//4), n_fft=feature_object.n_fft)
        W_col_matrix = np.zeros((len(W_col),1))
        W_col_matrix[:,0] = W_col
        Column = librosa.feature.inverse.mel_to_stft(W_col_matrix, sr=feat_obj.sr, n_fft=feat_obj.n_fft)
        return W_column_to_note(Column[:,0], feat_obj, pitch_min = pitch_min, pitch_max = pitch_max, salience_shift_autocorrelation = salience_shift_autocorrelation,
                                pitch_method = pitch_method, piano = piano)

    elif feature_object.feature == "cqt" or feature_object.feature == "vqt":
        found_pitch = thresholding_column(W_col, feature_object.feature, feature_object)
        #found_pitch = autocorrelation_cqt(W_col, feature_object, salience_shift_autocorrelation)
        if found_pitch < pitch_min: # A lower bound for the frequency range, must be calculated from the size of the window
            raise ValueError('The pitch is anormally low')

        elif found_pitch > pitch_max:
            raise ValueError('The pitch is anormally high')

        else:
            return freq_to_midi(found_pitch)

    elif feature_object.feature == "pcp":
        return np.argmax(W_col)

    else:
        raise NotImplementedError("TODO") from None

# Default parameters of the harmonic summation, per representation. Weighted sum (mean=False) : it is a bit less accurate
# than the weighted mean on isolated notes, but better on NMF codebooks, whose atoms are often a single partial
# (see harmonic_sum_pitch).
HARMONIC_SUM_DEFAULTS = {"stft": {"n_harmonics": 24, "decay": 0.8, "mean": False},
                         "cqt": {"n_harmonics": 24, "decay": 0.9, "mean": False},
                         "vqt": {"n_harmonics": 24, "decay": 0.9, "mean": False}}

def piano_inharmonicity(midi, bass_log=-8.0, bass_slope=-0.09, treble_log=-4.0, treble_slope=0.09):
    """
    Inharmonicity coefficient B of the piano strings as a function of the MIDI note, modelled as the sum of two
    log-linear asymptotes (as in Rigaud et al., "A parametric model and estimation techniques for the inharmonicity
    and tuning of the piano", JASA 2013) : B decreases in the bass (wound strings) and increases strongly in the treble.

        B(m) = exp(bass_log + bass_slope * (m - 21)) + exp(treble_log + treble_slope * (m - 108))

    The default values give B ~ 3.4e-4 for A0, ~ 2.5e-4 for C4 and ~ 1.8e-2 for C8.
    """
    midi = np.asarray(midi, dtype=float)
    return np.exp(bass_log + bass_slope * (midi - 21)) + np.exp(treble_log + treble_slope * (midi - 108))

def harmonic_frequencies(f0, n_harmonics, piano=True, midi=None):
    """
    Frequencies of the first n_harmonics partials of notes of fundamental frequencies f0.

    Parameters
    ----------
    f0 : float or numpy array
        Fundamental frequencies, in Hz.
    n_harmonics : integer
        Number of partials, including the fundamental.
    piano : boolean, optional
        If True, the partials follow the stiff string model of the piano, f_h = h * f0 * sqrt(1 + B * h^2), with B given
        by piano_inharmonicity. If False, they are exact multiples of f0 (f_h = h * f0), which suits most harmonic
        instruments. Other instruments can be added here. The default is True.
    midi : float or numpy array, optional
        MIDI notes of f0, used for the inharmonicity coefficient. Computed from f0 if None.

    Returns
    -------
    numpy array of shape (len(f0), n_harmonics)
    """
    f0 = np.atleast_1d(np.asarray(f0, dtype=float))
    h = np.arange(1, n_harmonics + 1)[None, :]
    if not piano:
        return f0[:, None] * h
    if midi is None:
        midi = 69 + 12 * np.log2(f0 / 440)
    B = piano_inharmonicity(np.atleast_1d(midi))[:, None]
    return f0[:, None] * h * np.sqrt(1 + B * h ** 2)

def frequency_to_bin(frequencies, feature_object):
    """
    Fractional bin index of frequencies (Hz) in the frequency axis of the representation (STFT or CQT/VQT).
    For a pooled CQT, feature_object.bins_per_octave must be the number of bins per octave after pooling.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    match feature_object.feature:
        case "stft":
            return frequencies * feature_object.n_fft / feature_object.sr
        case "cqt" | "vqt":
            return feature_object.bins_per_octave * np.log2(frequencies / feature_object.fmin)
        case _:
            raise NotImplementedError(f"No frequency axis defined for the feature {feature_object.feature}.")

def harmonic_sum_pitch(W_col, feature_object, n_harmonics=None, decay=None, piano=True, midi_min=21, midi_max=108, mean=None):
    """
    Estimates the fundamental frequency of a column of the codebook (STFT or CQT/VQT) by harmonic summation.

    For each candidate note between midi_min and midi_max, the salience is the weighted sum of the energy of the column
    at the frequencies of its partials (harmonic_frequencies, read by linear interpolation between bins), with weights
    decay**(h-1), over the partials falling inside the column. With mean=True, the sum is divided by the sum of these
    weights, so that high candidates (with few partials in range) are not penalized. The estimated note is the candidate
    with the highest salience.

    Why not the argmax / the autocorrelation : on the isolated notes of MAPS (8 pianos), the fundamental is not the
    strongest bin for every note below MIDI 36 (it is 30 to 40 dB below the strongest partial), nor for about half of
    the notes between MIDI 36 and 59. Correct notes on these isolated notes (8 pianos) :
        - CQT36 pooled : argmax 63%, weighted mean 94% (n_harmonics=24, decay=0.95), weighted sum 80% (24, 0.9) ;
        - STFT (200 bins) : autocorrelation 61% (17% of the columns rejected), weighted mean 82% (16, 0.8), weighted sum 78% (24, 0.8).
    In the STFT, the notes below MIDI 36 stay badly estimated (~25%) : with n_fft = 2048, their partials are not
    resolved (the main lobe of the window is wider than their spacing).
    On NMF codebooks (HALS, rank 88, 3 pieces of MAPS AkPnBcht), many atoms are a single partial : the mean then often
    assigns them to a subharmonic, while the sum keeps them at their frequency. Mean F-measure of the transcription :
        - CQT36 pooled : argmax 0.516, weighted mean 0.506, weighted sum 0.534 ;
        - STFT : autocorrelation 0.516, weighted mean 0.452, weighted sum 0.524.
    Hence the weighted sum is the default (HARMONIC_SUM_DEFAULTS).

    Parameters
    ----------
    W_col : numpy array
        One column of the W matrix.
    feature_object : object
        The object containing the feature parameters (feature, sr, n_fft, fmin, bins_per_octave).
    n_harmonics, decay : integer and float, optional
        Number of partials summed (including the fundamental) and geometric decay of their weights.
        If None, the defaults of HARMONIC_SUM_DEFAULTS for the feature are used.
    piano : boolean, optional
        Partials of the piano (inharmonic) if True, exact harmonics otherwise. See harmonic_frequencies. The default is True.
    midi_min, midi_max : integers, optional
        Range of the candidate notes. The default is the piano keyboard (21 to 108).
    mean : boolean, optional
        If True, the salience is the weighted mean over the partials inside the column (normalized by their weights).
        If False, it is the weighted sum : a column made of a single peak at f is then assigned to f rather than to a
        subharmonic of f. If None, the default of HARMONIC_SUM_DEFAULTS for the feature is used.

    Returns
    -------
    f0 : float
        The estimated fundamental frequency, in Hz (equal-tempered frequency of the estimated note).
    """
    if not np.any(W_col > 0):
        raise ValueError('The note-atom has no energy')
    defaults = HARMONIC_SUM_DEFAULTS.get(feature_object.feature, {"n_harmonics": 8, "decay": 0.9})
    n_harmonics = defaults["n_harmonics"] if n_harmonics is None else n_harmonics
    decay = defaults["decay"] if decay is None else decay
    mean = defaults.get("mean", True) if mean is None else mean

    midis = np.arange(midi_min, midi_max + 1)
    f0s = 440 * 2 ** ((midis - 69) / 12)
    positions = frequency_to_bin(harmonic_frequencies(f0s, n_harmonics, piano=piano, midi=midis), feature_object)
    n = len(W_col)
    in_range = (positions >= 0) & (positions <= n - 1)
    values = np.interp(positions, np.arange(n), W_col)
    weights = decay ** np.arange(n_harmonics)[None, :]
    salience = np.where(in_range, values * weights, 0).sum(axis=1)
    if mean:
        salience = salience / np.maximum(np.where(in_range, weights, 0).sum(axis=1), 1e-300)
    salience[~in_range[:, 0]] = -np.inf # The fundamental must be inside the column
    if np.all(np.isinf(salience)):
        raise ValueError('No candidate note inside the frequency range of the representation')
    return float(f0s[np.argmax(salience)])

def thresholding_column(W_col, feat, feature_object):
    """
    Méthode très artificielle pour extraire le pitch d'une bande d'un spectrogramme. Trouve la bin de plus forte énergie
    puis renvoie la fréquence associée à ce bin.
    Ne dépend pas de l'échelle de la colonne (contrairement à un seuil absolu), donc fonctionne quelle que soit la normalisation de W.
    """
    if not np.any(W_col > 0):
        raise ValueError('The note-atom has no energy')
    bin_ind = np.argmax(W_col)
    sr = feature_object.sr
    N = feature_object.n_fft
    match feat:
        case "cqt" | "vqt":
            f_bin = feature_object.fmin * 2**((bin_ind)/feature_object.bins_per_octave)
        case "stft":
            f_bin = bin_ind * (sr/N)
        case "mel":
            assert False, "TODO : thresholding pitch estimate for mel spectrograms"
    return f_bin

def autocorrelate_signal(W_col, feature_object, salience_shift_autocorrelation = 0.3):
    """
    Compute the autocorrelation of the waveform of the note, and estimate the fundamental frequency from the maximum of the autocorrelation.
    """
    # Compute the waveform from the inverse Fourier transform of the note spectrogram
    # Zero-padding up to n_fft//2+1 bins: if the spectrogram was truncated (e.g. only the first frequency bins are kept),
    # the waveform would otherwise be too short and the estimated f0 = sr/lag would be overestimated by n_fft/(2*(len(W_col)-1)).
    wave_signal_W_col = np.fft.irfft(W_col, n=feature_object.n_fft)

    # Auto-correlation of the waveform
    autocorrelation_wave_signal = np.correlate(wave_signal_W_col, wave_signal_W_col, mode='full')

    # Auto-correlation is symmetric, we only keep the second half
    autocorrelation_wave_signal = autocorrelation_wave_signal[len(autocorrelation_wave_signal)//2:]

    # Normalization (for the threshold)
    autocorrelation_wave_signal /= np.amax(autocorrelation_wave_signal)

    # Offset on the potential values for autocorrelation in order not to take the maximum, occuring when the signal is correlated at time 0.
    # This offset has to be large enough to eliminate enough first values which are correlate to the case of 0 delay in autocorrelation.
    # In that context, we have chosen to take the first negative value of the autocorrelation as the offset, because it eliminates all the values that are correlated to the case of 0 delay.
    # (can/should be discussed)
    negative_indices = np.where(autocorrelation_wave_signal < 0)[0]
    if len(negative_indices) > 0:
        offset = negative_indices[0]
    else: # If no negative value is found, it means that the autocorrelation is always positive, which is not possible.
        return None
        # raise err.ToDebugException("No negative value found in the autocorrelation of the inverse Fourier transform of the note spectrogram. This should never happen.")
    
    ## A second offset idea based on a first guess of the frequency, corresponding to the maximum of the column of the codebook (i.e. the stronget frequency) 
    # first_guess = np.argmax(W_col)
    # offset = max(first_guess//2, np.argmin(autocorrelation_wave_signal))

    # If the maximum of the autocorrelation is above a certain threshold, we consider it as a valid pitch
    if np.amax(autocorrelation_wave_signal[offset:]) > salience_shift_autocorrelation:
        return feature_object.sr/(offset + np.argmax(autocorrelation_wave_signal[offset:]))
    else: # Otherwise we consider that the pitch is not valid
        return None

def autocorrelate_freq(W_col, salience_shift_autocorrelation = 0.3):
    """
    Compute the autocorrelation of the frequency decomposition of the note, and estimate the fundamental frequency from the maximum of the autocorrelation.
    Doesn't work so much.
    """
    # Compute the cross-correlation
    cross_correlation = np.correlate(W_col, W_col, mode='full')

    # Normalization
    cross_correlation /= np.amax(cross_correlation)

    # Keeping the positive values (symmetric)
    autocorrelation_idx = len(cross_correlation)//2
    subset_middle_vals = cross_correlation[autocorrelation_idx:]

    # A first guess of the frequency, corresponding to the maximum of the column of the codebook (i.e. the strongest frequency)
    first_guess = np.argmax(W_col)
    offset = max(4, first_guess - 10) # Really arbitrary, should be discussed
    # If the maximum of the autocorrelation is above a certain threshold, we consider it as a valid pitch
    if np.amax(subset_middle_vals[offset:]) > salience_shift_autocorrelation:
        return np.argmax(subset_middle_vals[offset:]) + offset
    else: # Otherwise we consider that the pitch is not valid
        return None


# %% H to onsets
def _detect_note_segments_frames(H, H_process, threshold, smoothing_window, H_normalization,
                                  ruptures_detector_model, pelt_penalty, pelt_min_size, verbose):
    """
    Dispatch the frame-domain segment detection to nmf_audio_benchmark.tasks.ecoacoustics.sound_event_detection.

    Returns a dict mapping note (row of H) index to a list of (start_frame, end_frame) segments.
    """
    match H_process:
        case "thresholding":
            return sed.threshold_H(H, threshold, smoothing_window=smoothing_window, H_normalization=H_normalization, verbose=verbose)

        case "pelt":
            return sed.run_pelt(H, model=ruptures_detector_model, penalty=pelt_penalty, min_size=pelt_min_size, H_normalization=H_normalization, verbose=verbose)

        case "window":
            return sed.run_window_ruptures(H, smoothing_window=smoothing_window, model=ruptures_detector_model, threshold=threshold, H_normalization=H_normalization)

        case _:
            raise ValueError(f"Unsupported H_process method: {H_process}")


def H_to_activations(W_notes, H, feature_object, H_process = "thresholding",
                      threshold = 0.01, smoothing_window = 5, H_normalization = None,
                      ruptures_detector_model = 'normal', pelt_penalty = 0, pelt_min_size = 1,
                      energy_filtering_criterion = None, energy_filter_stat = "mean_energy", percentile_detection = 90, energy_filter_threshold = None,
                      time_tol = None, min_duration = None, max_duration = None,
                      verbose = True):
    """
    Estimate the activations (onset, offset, pitch) of the notes in the transcription.

    Frame-domain segment detection (thresholding, PELT or window change-point detection, via
    H_process) and generic post-processing (energy filtering, gap merging, duration filtering) are
    delegated to nmf_audio_benchmark.tasks.ecoacoustics.sound_event_detection. On top of that, this
    function adds the transcription-specific steps: mapping each detected segment to its W-atom
    pitch, refining onsets with the piano-hammer heuristic (find_onset), and merging activations
    across atoms that share the same pitch.

    Parameters
    ----------
    W_notes : list
        The MIDI value of each note in the codebook.
    H : numpy array
        The H matrix of the NMF decomposition of the spectrogram, corresponding to the activations of the notes.
    feature_object : object
        The object containing the feature parameters of the audio signal.
    H_process : {"thresholding", "pelt", "window"}, optional
        The detector used to find the note segments in each row of H. The default is "thresholding".
    threshold : float, optional
        The threshold to detect the presence of a note in the activations (used by "thresholding" and,
        as the ruptures ``epsilon``, by "window"). The default is 0.01.
    smoothing_window : integer, optional
        The number of frames to average the activation value in order to detect the presence of a note
        (used by "thresholding"), or the window width (used by "window"). The default is 5.
    H_normalization : {None, "max", "row_max", "mean", "row_mean", "l2", "row_l2"}, optional
        How H is normalized before detection. The default is None (no normalization).
    ruptures_detector_model : str, optional
        The ruptures cost model, used by "pelt" and "window". The default is 'normal'.
    pelt_penalty : float, optional
        The PELT penalty, used by "pelt". The default is 0.
    pelt_min_size : integer, optional
        The PELT minimal segment size, used by "pelt". The default is 1.
    energy_filtering_criterion : {None, "percentile", "threshold"}, optional
        If set, segments are additionally selected by their energy. The default is None.
    energy_filter_stat : {"mean_energy", "cumulative_energy"}, optional
        The statistic used for the energy filtering. The default is "mean_energy".
    percentile_detection : float, optional
        The percentile used when energy_filtering_criterion is "percentile". The default is 90.
    energy_filter_threshold : float, optional
        The threshold used when energy_filtering_criterion is "threshold". Defaults to ``threshold`` when None.
    time_tol : float, optional
        If set, segments closer than time_tol (in seconds) are merged. The default is None.
    min_duration : float, optional
        Minimal duration (in seconds) of a note, used together with max_duration. The default is None.
    max_duration : float, optional
        Maximal duration (in seconds) of a note, used together with min_duration. The default is None.
    verbose : boolean, optional
        verbose mode. The default is True.
    """
    segments_frames_and_note = _detect_note_segments_frames(
        H, H_process=H_process, threshold=threshold, smoothing_window=smoothing_window, H_normalization=H_normalization,
        ruptures_detector_model=ruptures_detector_model, pelt_penalty=pelt_penalty, pelt_min_size=pelt_min_size, verbose=verbose,
    )

    normalized_H = dm.normalize_H(H, H_normalization=H_normalization)

    if energy_filter_threshold is None:
        energy_filter_threshold = threshold

    note_tab = []

    for note_index, segments_frames in segments_frames_and_note.items():
        pitch = W_notes[note_index] # Storing the pitch of the actual note
        if pitch is None: # The note is incorrect
            if verbose:
                print(f"The {note_index}-th note in the codebook is incorrect. Skipping it.")
            continue

        if segments_frames:
            ## A test to try to find the exact moment of the onset, which appears to be a bit tricky. This is an heuristic, see find_onset for more details.
            segments_frames = [(find_onset(normalized_H[note_index], start, threshold), end) for start, end in segments_frames]

        # Post process filters the segments, according to different possibilites:
        # - Energy filtering: if the energy of the segment is too low, it is removed. This is useful to remove segments that are not really notes, but just noise.
        # - Gap merging: if two segments are too close, they are merged. This is useful to merge segments that are actually the same note, but that have been split by the detection algorithm
        # - Duration filtering: if a segment is too short or too long, it is removed
        # It also converts the segments from frames to seconds, using the feature_object parameters (sr and hop_length)
        note_segments = sed.post_process_segments(
            H_col=normalized_H[note_index],
            segments_frames=segments_frames,
            feature_object=feature_object,
            energy_filter_criterion=energy_filtering_criterion,
            energy_filter_percentile=percentile_detection,
            energy_filter_threshold=energy_filter_threshold,
            energy_filter_stat=energy_filter_stat,
            time_tol=time_tol,
            min_duration=min_duration,
            max_duration=max_duration,
            verbose=verbose,
            index_number=note_index,
        )

        if not note_segments:
            continue

        for onset, offset in note_segments:
            if offset <= onset:
                raise err.ToDebugException("The offset of the note is before the onset. This should never happen.")
            note_tab.append([onset, offset, pitch]) # Format for the .txt

    ## Notes should be merged, because a same note can be represented with several atoms in W
    note_tab = merge_overlapping_activations(note_tab)

    ## Remove the notes that are too short
    # note_tab = remove_small_notes(note_tab, minimal_length_note=0.2)

    ## Check that there is no overlap between the activations of notes with the same pitch
    test_no_overlap(note_tab)

    return note_tab

def find_onset(H_row, time_index, threshold):
    """
    Find a good candidate as onset.
    This is an heuristic, may not be the best way to find the onset.
    The idea is that the peak of the activation actually corresponds to the note onset, but the annotated onset is generally before the peak.
    This is due to mechanical pianos, where the annotation correspond to the moment where the hammer is launched, and the peak of the activation corresponds to the moment where the hammer hits the string.
    This gap between the annotation and the peak of the note is actually often higher than the tolerance, which is why we try to find the onset by looking at the 3 frames before the annotated onset.
    This is exhibited in [2, Chap 4.1].

    Ref:
    [2] Marmoret, A., Bertin, N., & Cohen, J. (2019). Multi-Channel Automatic Music Transcription Using Tensor Algebra. arXiv preprint arXiv:2107.11250.
    """
    # Finding the actual onset time, starting from the 3 frames before.
    start_idx = max(0, time_index-2)
    end_idx = min(H_row.shape[0], time_index+1)
    for possible_onset in range(start_idx, end_idx+1):
        if H_row[possible_onset] > 0.1 * threshold: # This onset is above 0.1*threshold, to try to find the exact onset, because in general the peak of activations comes after the annotated onset (because it is a mechanical piano).
            return possible_onset
    return time_index

def merge_overlapping_activations(activations):
    """
    Merge overlapping activations of notes with the same pitch.
    """
    # Group activations by pitch
    activations_by_pitch = defaultdict(list)
    for activation in activations:
        pitch = activation[2]
        activations_by_pitch[pitch].append(activation)

    merged_activations = []

    # Merge overlaps within each pitch group
    for pitch, pitch_activations in activations_by_pitch.items():
        # Sort activations by start time
        pitch_activations.sort(key=lambda x: x[0])
        
        current_activation = pitch_activations[0]
        
        for next_activation in pitch_activations[1:]:
            
            # if next_activation[0] - current_activation[0] <= minimal_length_note: # can be implemented to merge notes that are not only overlapping, but too close
            if current_activation[1] >= next_activation[0]:
                # Merge the activations
                current_activation[1] = max(current_activation[1], next_activation[1])
            else:
                # Add the current activation to the merged list and move to the next
                merged_activations.append([current_activation[0], current_activation[1], pitch])
                current_activation = next_activation
        
        # Add the last activation for the current pitch
        merged_activations.append([current_activation[0], current_activation[1], pitch])
    
    return merged_activations

def remove_small_notes(activations, minimal_length_note=0.2):
    """
    Remove notes that are too short.
    """
    return [activation for activation in activations if activation[1] - activation[0] >= minimal_length_note]

def test_no_overlap(activations):
    """
    Test that there is no overlap between the activations of notes with the same pitch.
    """
    from collections import defaultdict

    # Group activations by pitch
    activations_by_pitch = defaultdict(list)
    for activation in activations:
        pitch = activation[2]
        activations_by_pitch[pitch].append(activation)

    # Check for overlaps within each pitch group
    for pitch, pitch_activations in activations_by_pitch.items():
        # Sort activations by start time
        pitch_activations.sort(key=lambda x: x[0])
        
        for i in range(len(pitch_activations) - 1):
            current_end = pitch_activations[i][1]
            next_start = pitch_activations[i + 1][0]
            assert current_end <= next_start, f"Overlap detected between {pitch_activations[i]} and {pitch_activations[i + 1]} for pitch {pitch}"

# %% Utils

def freq_to_midi(frequency):
    """
    Returns the frequency (Hz) in the MIDI scale
    Parameters
    ----------
    frequency: float
        Frequency in Hertz
    Returns
    -------
    midi_f0: integer
        Frequency in MIDI scale
    """
    return int(round(69+ 12 * math.log(frequency/440,2)))

def midi_to_freq(midi_freq):
    """
    Returns the MIDI frequency in Hertz
    Parameters
    ----------
    midi_freq: integer
        Frequency in MIDI scale
    Returns
    -------
    frequency: float
        Frequency in Hertz
    """
    return 440 * 2**((midi_freq - 69)/12)

# %% Metrics
def compute_scores(estimations, annotations, time_tolerance=0.05):
    """
    Compute the F-measure and the accuracy of the transcription_evaluation.
    """
    if estimations == []:
        return 0, 0
    ref_np = np.array(annotations, float)
    ref_times = np.array(ref_np[:,0:2], float)
    ref_pitches = np.array(ref_np[:,2], int)

    est_np = np.array(estimations, float)
    est_times = np.array(est_np[:,0:2], float)
    est_pitches = np.array(est_np[:,2], int)

    prec, rec, f_mes, _ = mir_eval.transcription.precision_recall_f1_overlap(ref_times, ref_pitches, est_times, est_pitches, offset_ratio = None, onset_tolerance = time_tolerance, pitch_tolerance = 0.1)

    accuracy = accuracy_from_recall(rec, len(ref_times), len(est_times))

    return f_mes, accuracy

def accuracy_from_recall(recall, N_gt, N_est):
    """
    Compute the accuracy from recall, number of samples in ground truth (N_gt), and number of samples in estimation (N_est).

    Parameters
    ----------
    recall: float
        Recall value.
    N_gt: int
        Number of samples in ground truth.
    N_est: int
        Number of samples in estimation.

    Returns
    -------
    accuracy: float
        The Accuracy
    """
    TP = int(recall * N_gt)
    FN = int(N_gt - TP)
    FP = int(N_est - TP)
    return accuracy(TP, FP, FN)

def accuracy(TP, FP, FN):
    """
    Computes the accuracy of the transcription_evaluation:

        Accuracy = True Positives / (True Positives + False Positives + False Negatives)

    Parameters
    ----------
    TP: integer
        Number of true positives (Correctly detected notes: pitch and onset)
    FP: integer
        Incorrectly transcribed notes (wrong pitch, wrong onset, or doesn't exit)
    FN: integer
        Untranscribed notes (note in the ground truth, but not found in transcription_evaluation with the correct pitch and the correct onset)

    Returns
    -------
    accuracy: float
        The Accuracy
    """
    try:
        return TP/(TP + FP + FN)
    except ZeroDivisionError:
        return 0

