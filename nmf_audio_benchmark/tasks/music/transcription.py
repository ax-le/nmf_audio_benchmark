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

import math
import numpy as np
import librosa
from collections import defaultdict
import mir_eval
import tqdm

import nmf_audio_benchmark.utils.errors as err
import nmf_audio_benchmark.utils.find_hyperparameters as hyperparams_helper
from sklearn.base import clone


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
    def __init__(self, feature_object, salience_shift_autocorrelation = 0.3, threshold = 0.01, smoothing_window = 5, H_normalization = True, adaptative_threshold = False, averaging_window_adaptative = 10, verbose = False):
        """
        Constructor of the Transcription estimator.

        Parameters
        ----------
        feature_object : object
            The object containing the feature parameters of the audio signal. 
        salience_shift_autocorrelation : float, optional
            The threshold for the autocorrelation of the waveform to detect the fundamental frequency. 
            Below this threshold, the pitch is considered as invalid.
            The default is 0.3.
        threshold : float, optional
            The threshold to detect the presence of a note in the activations.
            The default is 0.01.
        smoothing_window : integer, optional
            The number of frames to average the activation value in order to detect the presence of a note, in the default way (non adaptative).
            The default is 5.
        H_normalization : boolean, optional
            If True, the activations are normalized.
            The default is True.
        adaptative_threshold : boolean, optional
            If True, the threshold is adaptative, i.e. the activation value should be above the threshold and the average activation value over several frames.
            The default is False.
        averaging_window_adaptative : integer, optional
            The number of frames to average the activation value in order to detect the presence of a note, in the adaptative way.
            This number counts for negative and positive frames, i.e. computing the average over 2*averaging_window_adaptative + 1 frames.
            The default is 10.
        verbose : boolean, optional
            verbose mode. The default is False.
        """
        self.feature_object = feature_object
        self.salience_shift_autocorrelation = salience_shift_autocorrelation
        self.threshold = threshold
        self.smoothing_window = smoothing_window
        self.H_normalization = H_normalization
        self.adaptative_threshold = adaptative_threshold
        self.averaging_window_adaptative = averaging_window_adaptative
        self.verbose = verbose

    def predict(self, W, H):
        """
        Compute the transcription from the NMF decomposition of the spectrogram.        
        """
        W_notes = W_to_notes(W=W, feature_object=self.feature_object, salience_shift_autocorrelation=self.salience_shift_autocorrelation, verbose = self.verbose)
        activations = H_to_activations(W_notes=W_notes, H=H, feature_object=self.feature_object, threshold=self.threshold, 
                                       smoothing_window=self.smoothing_window, H_normalization=self.H_normalization, 
                                       adaptative_threshold=self.adaptative_threshold, averaging_window_adaptative=self.averaging_window_adaptative, verbose = self.verbose)
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
def W_to_notes(W, feature_object, pitch_min = 50, pitch_max = 5000, salience_shift_autocorrelation = 0.3, verbose = True):
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
        Below this threshold, the pitch is considered as invalid.
        The default is 0.3.
    verbose : boolean, optional
        verbose mode. The default is True.
    """
    f0s = []
    for idx_col in range(0,W.shape[1]): # Fundamental frequency estimation of each atom of the codebook
        try:
            f0s.append(W_column_to_note(W[:,idx_col], feature_object, pitch_min = pitch_min, pitch_max = pitch_max, salience_shift_autocorrelation = salience_shift_autocorrelation))
        except ValueError as err:
            f0s.append(None)
            if verbose:
                print("Error in the " + str(idx_col) + "-th note-atom of the codebook: " + err.args[0])
    return f0s

def W_column_to_note(W_col, feature_object, pitch_min = 27, pitch_max = 4500, salience_shift_autocorrelation = 0.3):
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
        Below this threshold, the pitch is considered as invalid.
        The default is 0.3.
    """
    if feature_object.feature == "stft":

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
            return freq_to_midi(found_pitch)

    else:
        raise NotImplementedError("TODO") from None

def autocorrelate_signal(W_col, feature_object, salience_shift_autocorrelation = 0.3):
    """
    Compute the autocorrelation of the waveform of the note, and estimate the fundamental frequency from the maximum of the autocorrelation.
    """
    # Compute the waveform from the inverse Fourier transform of the note spectrogram
    wave_signal_W_col = np.fft.irfft(W_col)

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
def H_to_activations(W_notes, H, feature_object, threshold, smoothing_window = 5, H_normalization = True, adaptative_threshold = False, averaging_window_adaptative = 10, verbose = True):
    """
    Estimate the activations of the notes in the transcription.
    Notes are detected when the activation level is above a certain threshold.
    The default way is to consider that a note is detected if the activation level is above the threshold and the average activation level over several frames is above the threshold, to avoid spurious peaks.
    The adaptative way is to consider that a note is detected if the activation level is above a fixed value for the thrshold + the averaged value of activations over <averaging_window_adaptative> frames in the past and the future.

    Parameters
    ----------
    W_notes : list
        The MIDI value of each note in the codebook.
    H : numpy array
        The H matrix of the NMF decomposition of the spectrogram, corresponding to the activations of the notes.
    feature_object : object
        The object containing the feature parameters of the audio signal.
    threshold : float
        The fixed threshold value to detect the presence of a note in the activations.
    smoothing_window : integer, optional
        The number of frames to average the activation value in order to detect the presence of a note, in the default way (non adaptative).
        The default is 5.
    H_normalization : boolean, optional
        If True, the activations are normalized.
        The default is True.
    adaptative_threshold : boolean, optional
        If True, the threshold is adaptative, i.e. the activation value should be above the threshold and the average activation value over several frames.
        The default is False.
    averaging_window_adaptative : integer, optional
        The number of frames to average the activation value in order to detect the presence of a note, in the adaptative way.
        This number counts for negative and positive frames, i.e. computing the average over 2*averaging_window_adaptative + 1 frames.
        The default is 10.
    verbose : boolean, optional
        verbose mode. The default is True.
    """
    if H_normalization:
        H_max = np.linalg.norm(H, 'fro')
    else:
        H_max = 1

    note_tab = []

    presence_of_a_note = False
    current_pitch = 0
    current_onset = 0
    current_offset = 0

    for note_index in range(H.shape[0]): # Looping over the notes
        if presence_of_a_note: # Avoiding an uncontrolled situation (boolean to True before looking at this note, should never happen in theory)
            presence_of_a_note = False

        current_pitch = W_notes[note_index] # Storing the pitch of the actual note
        if current_pitch is None: # The note is incorrect
            # An error occured, the note is incorrect
            if verbose:
                print(f"The {note_index}-th note in the codebook is incorrect. Skipping it.")
            continue

        for time_index in range(H.shape[1]): # Taking each time bin
            # Detecting a note
            if adaptative_threshold: # Using an adaptative threshold, i.e. considering that a note should be detected if it is larger than a threshold value + the activation average over several frames
                note_detected = detect_a_note_with_adaptative_threshold(H=H, note_index=note_index, time_index=time_index, threshold=threshold, H_max=H_max, averaging_window_adaptative=averaging_window_adaptative)
            else: # Using a fixed threshold
                note_detected = detect_a_note(H=H, note_index=note_index, time_index=time_index, threshold=threshold, H_max=H_max, smoothing_window=smoothing_window)

            if note_detected: # A note is detected
                if not presence_of_a_note: # The note was not detected before
                    current_pitch = W_notes[note_index] # Storing the pitch of the actual note

                    ## A test to try to find the exact moment of the onset, which appears to be a bit tricky. This is an heuristic, see the function for more details
                    onset_time_index = find_onset(H, note_index, time_index, threshold, H_max)
                    # onset_time_index = time_index

                    current_onset = librosa.frames_to_time(onset_time_index, sr=feature_object.sr, hop_length=feature_object.hop_length, n_fft=feature_object.n_fft)
                    presence_of_a_note = True # Note detected (for the future frames)

                # Else, the note was already detected, hence we continue until the activation level is below the threshold

            else: # Note level is too low
                if presence_of_a_note: # If a note was detected before, it means that the note is over
                    current_offset = librosa.frames_to_time(time_index, sr=feature_object.sr, hop_length=feature_object.hop_length, n_fft=feature_object.n_fft)
                    if current_offset <= current_onset:
                        raise err.ToDebugException("The offset of the note is before the onset. This should never happen.")
                    
                    note_tab.append([current_onset, current_offset, current_pitch]) # Format for the .txt

                    presence_of_a_note = False # Reinitializing the detector of a note

    ## Notes should be merged, because a same note can be represented with several atoms in W
    note_tab = merge_overlapping_activations(note_tab)

    ## Remove the notes that are too short
    # note_tab = remove_small_notes(note_tab, minimal_length_note=0.2)

    ## Check that there is no overlap between the activations of notes with the same pitch
    test_no_overlap(note_tab)

    return note_tab

def detect_a_note(H, note_index, time_index, threshold, H_max, smoothing_window):
    """
    Detect the presence of a note in the activations, in the default way.
    """
    # The activation value of the note at the current time
    current_val = H[note_index, time_index]

    # The average activation value of the note over the <smoothing_window> next frames
    end_time = min(H.shape[1], time_index + smoothing_window)
    average_value_smoothing_window = np.mean(H[note_index, time_index:end_time]) # Average the activation value on several consecutive frames to eliminate spurious peaks

    normalized_threshold = threshold * H_max
    # The activation should be above than the threshold and the averaged value for several consecutive frames to be considered detected.
    return (is_above_threshold(value=current_val, threshold=normalized_threshold) and is_above_threshold(value=average_value_smoothing_window, threshold=normalized_threshold))

def detect_a_note_with_adaptative_threshold(H, note_index, time_index, threshold, H_max, averaging_window_adaptative):
    """
    Adaptative threshold to detect the presence of a note in the activations.
    The threshold is adaptative, i.e. the threshold value is the sum of a fixed value and the averaged value of the activation over several frames.
    """
    start_time = max(0, time_index-averaging_window_adaptative)
    end_time = min(H.shape[1], time_index+averaging_window_adaptative)
    adaptative_average_activation = np.mean(H[note_index, start_time:end_time]) # Possible to compute it for the whole matrix, maybe cheaper.
    return is_above_threshold(value=H[note_index, time_index], threshold=threshold * H_max + adaptative_average_activation) # The activation should be above than the threshold + an averageed value for several consectuvie frames.

def find_onset(H, note_index, time_index, threshold, H_max):
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
    # Finding the actual onset time, starting from the 3 frames before.
    start_idx = max(0, time_index-2)
    end_idx = min(H.shape[1], time_index+1)
    for possible_onset in range(start_idx, end_idx+1):
        if is_above_threshold(value=H[note_index, possible_onset], threshold=0.1 * threshold * H_max): # This onset is above 0.1*threshold, to try to find the exact onset, because in general the peak of activations comes after the annotated onset (because it is a mechanical piano).
            return possible_onset
    return time_index

def is_above_threshold(value, threshold):
    """
    Wrapper to test if a value is above a threshold.
    """
    return value > threshold

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

def l2_normalise(an_array, eps=1e-10):
    """
    Normalise an array along the second axis using the L2 norm.
    """
    norm = np.linalg.norm(an_array, axis = 1)
    an_array_T = np.transpose(an_array)
    out = np.inf * np.ones_like(an_array_T)
    np.divide(an_array_T, norm, out = out, where=norm!=0)
    an_array_T = np.where(np.isinf(out), eps, out)
    return np.transpose(an_array_T)

