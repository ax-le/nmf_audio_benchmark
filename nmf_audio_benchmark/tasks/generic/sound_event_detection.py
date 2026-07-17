#!/usr/bin/env python3
"""NMF Detection functions for temporal events (i.e. post-processing the H matrix)."""

import numpy as np
import ruptures as rpt

from nmf_audio_benchmark.tasks.base_task import *
import nmf_audio_benchmark.utils.data_manipulation as dm

# %% Temporal detection functions for NMF activations (H matrix)
# =============================================================================
# MAIN DETECT FUNCTION
# =============================================================================

def detect(H, H_process, feature_object,
           threshold=0.01, smoothing_window=5, H_normalization=None,
           detector_model='normal', detector_penalty=0, detector_min_size=1, 
           energy_filtering_criterion=None, energy_filter_stat="mean_energy",percentile_detection=90, energy_filter_threshold=None,
           time_tol=None, min_duration=None, max_duration=None,
           verbose=True):  # time_tol=0.05, min_duration=0.1, max_duration=300
    """
    Detect temporal events from the activation matrix H.

    Runs one of three detectors (``H_process``: "thresholding", "pelt" or
    "window") to find per-event segments in frames, then converts them to
    seconds and post-processes them (merging nearby segments, filtering by
    duration).

    Returns:
        Dict mapping event index to a list of (start_time, end_time) segments, in seconds.
    """
    match H_process:
        case "thresholding":
            if verbose:
                print(f"Running thresholding with threshold={threshold}, smoothing_window={smoothing_window}, H_normalization={H_normalization}")
            segments_frames_and_event = threshold_H(H, threshold, smoothing_window=smoothing_window, H_normalization=H_normalization, verbose=verbose)

        case "pelt":
            if verbose:
                print(f"Running PELT change point detection with model='{detector_model}', penalty={detector_penalty}, min_size={detector_min_size}, H_normalization={H_normalization}")
            segments_frames_and_event = run_pelt(
                H=H, model=detector_model, penalty=detector_penalty, min_size=detector_min_size,
                H_normalization=H_normalization, verbose=verbose,
            )

        case "window":
            if verbose:
                print(f"Running window-based change point detection with smoothing_window={smoothing_window}, model='{detector_model}', threshold={threshold}, H_normalization={H_normalization}")
            segments_frames_and_event = run_window_ruptures(
                H=H, smoothing_window=smoothing_window, model=detector_model, threshold=threshold, H_normalization=H_normalization,
            )

        case _:
            raise ValueError(f"Unsupported H_process method: {H_process}")

    normalized_H = dm.normalize_H(H, H_normalization=H_normalization)

    segment_times_and_event = {}

    if energy_filter_threshold is None:
        energy_filter_threshold = threshold

    for event_index, segments_frames in segments_frames_and_event.items():
        segment_times_and_event[event_index] = post_process_segments(
            H_col=normalized_H[event_index],
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
            index_number=event_index,
        )

    return segment_times_and_event


# =============================================================================
# THRESHOLDING DETECTION
# =============================================================================

def threshold_H(H, threshold, smoothing_window=5, H_normalization=None, verbose=True):
    """
    Detect contiguous above-threshold segments independently for each row (event) of H.

    Returns:
        Dict mapping event index to a list of (start_frame, end_frame) segments.
    """
    segments = {}
    normalized_H = dm.normalize_H(H, H_normalization=H_normalization)

    for event_index in range(H.shape[0]):
        bucket = segments.setdefault(event_index, [])
        presence_of_an_event = False
        current_onset = None

        for time_index in range(H.shape[1]):
            event_detected = detect_an_event(
                H_col=normalized_H[event_index], time_index=time_index,
                threshold=threshold, smoothing_window=smoothing_window,
            )

            if event_detected:
                if not presence_of_an_event:
                    current_onset = time_index
                    presence_of_an_event = True
                # Else the event was already detected; keep going until it drops below threshold.

            elif presence_of_an_event:
                # Activation just dropped below threshold: the event just ended.
                bucket.append((int(current_onset), int(time_index)))
                presence_of_an_event = False

        if presence_of_an_event:
            # Activation is still above threshold at the last frame: close the event there.
            bucket.append((int(current_onset), int(H.shape[1])))

        if verbose:
            print(f"Component {event_index}: kept {len(bucket)} thresholded segments")

    return segments


def detect_an_event(H_col, time_index, threshold, smoothing_window):
    """
    Detect the presence of aan event in the activations, in the default way.

    The activation must be above the threshold both at the current frame and,
    on average, over the following ``smoothing_window`` frames, to eliminate
    spurious peaks.
    """
    current_val = H_col[time_index]

    end_time = min(H_col.shape[0], time_index + smoothing_window)
    average_value_smoothing_window = np.mean(H_col[time_index:end_time])

    return (current_val >= threshold) and (average_value_smoothing_window >= threshold) # TODO: Are both necessary? Isn't the average enough? 


# =============================================================================
# CHANGE-POINT DETECTION (ruptures)
# =============================================================================

def _detect_breakpoints_per_row(normalized_H, make_algo, predict_kwargs, verbose_fn=None):
    """
    Shared driver for ruptures-based detectors: fit/predict one algorithm per
    row (event) of H and convert the resulting breakpoints into segments.
    """
    all_detected_segments = {}

    for i, row in enumerate(normalized_H):
        algo = make_algo(row)
        breakpoints = algo.predict(**predict_kwargs)

        if verbose_fn is not None:
            verbose_fn(i, breakpoints)

        segments = dm.boundaries_to_segments(breakpoints, n_samples=len(row))
        all_detected_segments[i] = dm._normalize_segments(segments)

    return all_detected_segments


def run_pelt(H, model, penalty, min_size, H_normalization=None, verbose=False):
    normalized_H = dm.normalize_H(H, H_normalization=H_normalization)

    verbose_fn = None
    if verbose:
        def verbose_fn(i, breakpoints):
            print(f"Component {i}: detected {len(breakpoints) - 1} breakpoints with PELT "
                  f"(model={model}, penalty={penalty}, min_size={min_size})")

    return _detect_breakpoints_per_row(
        normalized_H,
        make_algo=lambda row: rpt.Pelt(model=model, min_size=min_size, jump=1).fit(row),
        predict_kwargs={"pen": penalty},
        verbose_fn=verbose_fn,
    )


def run_window_ruptures(H, smoothing_window=40, model="l2", threshold=0.1, H_normalization=None):  # model: "l2", "l1", "rbf", "linear", "normal", "ar"
    normalized_H = dm.normalize_H(H, H_normalization=H_normalization)

    return _detect_breakpoints_per_row(
        normalized_H,
        make_algo=lambda row: rpt.Window(width=smoothing_window, model=model).fit(row),
        predict_kwargs={"epsilon": threshold},
    )


# =============================================================================
# SEGMENT POST-PROCESSING
# =============================================================================

def compute_segments_energies(H_col, segments):

    if segments is None or len(segments) == 0:
        return None, None

    starts, ends = dm.get_starts_and_ends_from_segments(segments)

    cumulative_energy = np.array([np.sum(H_col[starts[i]:ends[i]]) for i in range(len(starts))], dtype=float)

    durations = (np.asarray(ends) - np.asarray(starts)).astype(float)
    assert len(durations) == len(starts) == len(ends)
    assert all(d > 0 for d in durations), f"All segments should have positive duration, got: {durations}"

    mean_energy = cumulative_energy / durations

    return cumulative_energy, mean_energy


def filter_segments_by_energy(H_col, segments, filtering_criterion="percentile", percentile=None, threshold=None, stat='mean_energy'):
    cumulative_energy, mean_energy = compute_segments_energies(H_col, segments)

    if cumulative_energy is None:
        return None

    if stat == 'mean_energy':
        stats_array = mean_energy
    elif stat == 'cumulative_energy':
        stats_array = cumulative_energy
    else:
        raise ValueError(f"Invalid stat. Expected 'mean_energy' or 'cumulative_energy', got {stat}")

    match filtering_criterion:
        case "percentile":
            if percentile is None:
                raise ValueError("percentile must be provided when filtering_criterion is 'percentile'")
            assert isinstance(percentile, (float, int)), f"Expected float, got {type(percentile)}"
            assert 0 <= percentile <= 100, f"Expected percentile between 0 and 100, got {percentile}"
            threshold = np.percentile(stats_array, percentile)

        case "threshold":
            if threshold is None:
                raise ValueError("threshold must be provided when filtering_criterion is 'threshold'")

        case _:
            raise ValueError(f"Invalid filtering_criterion. Expected 'percentile' or 'threshold', got {filtering_criterion}")

    idx = np.where(stats_array > threshold)[0]
    return [segments[i] for i in idx]


def filter_segments_by_length(detected_segments, min_duration, max_duration):
    """
    Filters detected segments by minimum and maximum duration.
    """
    if min_duration is not None and min_duration < 0:
        raise ValueError("min_duration must be >= 0 or None")

    if max_duration is not None and max_duration < 0:
        raise ValueError("max_duration must be >= 0 or None")

    if min_duration is not None and max_duration is not None and min_duration > max_duration:
        raise ValueError("min_duration cannot be greater than max_duration")

    if min_duration is None and max_duration is None:
        return detected_segments

    filtered = []
    for start, end in detected_segments:
        duration = end - start
        if min_duration is not None and duration < min_duration:
            continue
        if max_duration is not None and duration > max_duration:
            continue
        filtered.append((start, end))
    return filtered


def merge_segments(segments, time_tol, verbose=False):
    """Merge overlapping/nearby segments in seconds.

    Segments are merged when the gap between consecutive segments is
    less than or equal to ``time_tol``.
    """
    if segments is None:
        return None

    if time_tol is None:
        raise ValueError("time_tol must be provided")

    if not np.isfinite(time_tol):
        raise ValueError("time_tol must be a finite number")

    if time_tol < 0:
        raise ValueError("time_tol must be >= 0")

    if len(segments) == 0:
        return None

    normalized = dm._normalize_segments(segments, integer_frames=False)

    merged_segments = [normalized[0]]

    for start, end in normalized[1:]:
        previous_start, previous_end = merged_segments[-1]
        gap = start - previous_end
        if gap <= time_tol:
            merged_segments[-1] = (previous_start, max(previous_end, end))
        else:
            merged_segments.append((start, end))

    if verbose:
        print(f"\nMerged segments count: {len(merged_segments)} after applying time tolerance of {time_tol} seconds")

    return merged_segments


def post_process_segments(H_col, segments_frames, feature_object,
                           energy_filter_criterion=None, energy_filter_percentile=None,
                           energy_filter_threshold=None, energy_filter_stat="mean_energy",
                           time_tol=None, min_duration=None, max_duration=None, verbose=False, index_number=None):
    """
    Normalize, optionally filter by energy, convert to seconds, merge nearby
    segments and filter by duration.

    ``energy_filter_x`` is the frame-domain activation values (e.g. the
    normalized H row) for this event; when provided, segments are kept only
    if their energy (see ``energy_filter_criterion``/``energy_filter_stat``) passes
    the threshold/percentile before being converted to seconds.
    """
    if segments_frames is None:
        return None

    len_before = len(segments_frames)
    if verbose:
        print(f"Event {index_number}: detected {len_before} segments in frames before post-processing")

    segments_frames = dm._normalize_segments(segments_frames)

    if verbose and len_before != 0 and len(segments_frames) != len_before:
        print(f"Event {index_number}: {len(segments_frames)} segments after normalization (checks); normalization removed {len_before - len(segments_frames)} segments")

    if energy_filter_criterion is not None:
        if verbose:
            print(f"\nFiltering segments for event {index_number} by energy: criterion='{energy_filter_criterion}', stat='{energy_filter_stat}', percentile={energy_filter_percentile}, threshold={energy_filter_threshold}")
        segments_frames = filter_segments_by_energy(
            H_col, segments_frames,
            filtering_criterion=energy_filter_criterion, percentile=energy_filter_percentile,
            threshold=energy_filter_threshold, stat=energy_filter_stat,
        )

    if verbose:
        print(f"\nConverting detected segments from frames to seconds using feature='{feature_object.name}'")

    this_event_time_segments = dm.segments_in_frames_to_seconds(
        segments_frames, feature_object,
    )

    if time_tol is not None and time_tol > 0:
        if verbose:
            print(f"\nMerging segments for event {index_number} with time tolerance of {time_tol} seconds")
        this_event_time_merged_segments = merge_segments(this_event_time_segments, time_tol, verbose=False)
    else:
        this_event_time_merged_segments = this_event_time_segments

    if min_duration is not None and max_duration is not None:
        if verbose:
            print(f"\nFiltering segments for event {index_number} by duration: min={min_duration}s, max={max_duration}s")
        filtered_segments_this_event = filter_segments_by_length(this_event_time_merged_segments, min_duration, max_duration)
    else:
        filtered_segments_this_event = this_event_time_merged_segments

    return filtered_segments_this_event



# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    H = np.random.rand(3, 100)
    H += np.random.uniform(0, 10, H.shape)

    segments = detect(
        H,
        detector_model='normal',
        detector_penalty=10,
        detector_min_size=5,
        energy_filtering_criterion='percentile',
        percentile_detection=95,
        time_tol=1.0,
        feature='stft',
        sr=500,
        hop_length=20,
        verbose=True,
    )
    print("Detected segments (in seconds):", segments)
