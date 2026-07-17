import numpy as np
from base_audio.data_manipulation import *

# =========================================================================
# Time conversions (frames and time)
# =========================================================================

def segments_in_frames_to_seconds(segments, feature_object):

    return [
        (frame_to_second(start, feature_object), 
        frame_to_second(end, feature_object))
        for (start, end) in segments
    ]

def segments_in_seconds_to_frames(segments, feature_object):

    return [
        (second_to_frame(start, feature_object), 
        second_to_frame(end, feature_object))
        for (start, end) in segments
    ]


# =========================================================================
# Boundaries and segments
# =========================================================================

def boundaries_to_segments(boundaries, n_samples=None):
    """Convert sorted boundary indices to contiguous ``(start, end)`` segments.

    Example:
        boundaries ``[5, 9]`` -> segments ``[(0, 5), (5, 9)]``.
    """
    if boundaries is None or boundaries == []:
        return None

    boundaries = list(boundaries)
    if len(boundaries) == 0:
        return None

    ends = [int(b) for b in boundaries]
    if any(b < 0 for b in ends):
        raise ValueError("boundaries must be non-negative")

    for i in range(1, len(ends)):
        if ends[i] <= ends[i - 1]:
            raise ValueError("boundaries must be strictly increasing")

    if n_samples is not None:
        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0 when provided")
        if ends[-1] > n_samples:
            raise ValueError("last boundary cannot be greater than n_samples")
        if ends[-1] != n_samples:
            ends.append(n_samples)

    starts = [0] + ends[:-1]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def segments_to_boundaries(segments, n_samples=None):
    """Convert contiguous ``(start, end)`` segments to boundary indices.

    The segments must represent a partition that starts at 0 and has no gaps.
    """
    if segments is None or segments == []:
        return None

    segments = list(segments)
    if len(segments) == 0:
        return None

    normalized = _normalize_segments(segments, integer_frames=True)

    _assert_contiguous_segments(normalized)

    boundaries = [end for _, end in normalized]

    if n_samples is not None:
        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0 when provided")
        if boundaries[-1] > n_samples:
            raise ValueError("last segment end cannot be greater than n_samples")
        if boundaries[-1] != n_samples:
            boundaries.append(n_samples)

    return boundaries


def get_starts_and_ends_from_segments(segments):
    normalized = _normalize_segments(segments, integer_frames=True)
    _assert_contiguous_segments(normalized)

    starts = [start for start, _ in normalized]
    ends = [end for _, end in normalized]
    return starts, ends



# def frequency_bins_to_hz(feature_object):
#     """Convert frequency bin indices to hertz based on the feature configuration."""
#     match feature_object.feature:
#         case "stft" | "stft_complex" | "pcen" | "ltsa" | "ltsa_pcen":
#             return np.fft.rfftfreq(feature_object.n_fft, d=1.0 / feature_object.sr)

#         # case "mel" | "log_mel" | "nn_log_mel" | "padded_log_mel" | "minmax_log_mel":
#         #     if feature_object.mel_grill:
#         #         n_mels = 80
#         #         fmin = 80.0
#         #         fmax = min(16000.0, feature_object.sr / 2.0)
#         #     else:
#         #         n_mels = feature_object.n_mels
#         #         fmin = feature_object.fmin
#         #         fmax = feature_object.fmax if feature_object.fmax is not None else feature_object.sr / 2.0

#         #     return librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)

#         case _:
#             raise ValueError(f"Unsupported feature type: {feature_object.feature}")
# =========================================================================
# HELPERS
# ========================================================================

def _validate_segment(segment):
    """Return a validated ``(start, end)`` pair as ``(float, float)``."""
    if len(segment) != 2:
        raise ValueError(
            f"Segment {segment!r} must have exactly two elements (start, end)."
        )

    start = float(segment[0])
    end = float(segment[1])
    if end <= start:
        raise ValueError(
            f"Segment {segment!r} is invalid: "
            f"end ({end}) must be strictly greater than start ({start})."
        )

    return start, end

def _normalize_segments(segments, integer_frames=True):
    normalized = []
    for segment in segments:

        start, end = _validate_segment(segment)

        if integer_frames:
            start_i = int(start)
            end_i = int(end)
            if float(start_i) != start or float(end_i) != end:
                raise ValueError("segments_to_boundaries expects integer frame indices")
            normalized.append((start_i, end_i))
        else:
            normalized.append((start, end))

    normalized.sort(key=lambda x: x[0])
    return normalized


def _assert_contiguous_segments(segments):
    if segments[0][0] != 0:
        raise ValueError("contiguous segments must start at 0")

    previous_end = segments[0][0]
    for start, end in segments:
        if start != previous_end:
            raise ValueError("segments must be contiguous and non-overlapping")
        previous_end = end

# =============================================================================
# H NORMALIZATION
# =============================================================================

def normalize_H(H, H_normalization=None, eps=1e-10):
    match H_normalization:
        case "max":
            normalized_H = H / np.maximum(np.amax(H), eps)

        case "row_max":
            # Normalize H by rows, i.e. divide each row by its own max value.
            normalized_H = H / np.maximum(np.amax(H, axis=1, keepdims=True), eps)

        case "mean":
            normalized_H = H / np.maximum(np.mean(H), eps)

        case "row_mean":
            # Normalize H by rows, i.e. divide each row by its own mean value.
            normalized_H = H / np.maximum(np.mean(H, axis=1, keepdims=True), eps)

        case "l2" | "row_l2":
            # Normalize H by rows using the L2 norm.
            row_norms = np.linalg.norm(H, axis=1, keepdims=True)
            normalized_H = H / np.maximum(row_norms, eps)

        case _:
            normalized_H = H

    return normalized_H
