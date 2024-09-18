# NMF Audio Benchmark

This is a repository aimed at facilitating the benchmarking of NMF-based techniques in the context of audio processing.

## Overview

This toolbox was designed following the fact that new low-rank factorization models are still being developed (in particular under new constraints or objective functions), but testing them under real conditions with audio data is not an easy task. In that spirit, this toolbox was primarily created to:

- Provide a standardized framework for evaluating NMF-based audio processing techniques.
- Offer a collection of audio datasets and pre-processing tools.
- Include a set of baseline NMF algorithms for comparison.
- Enable easy integration of new NMF models for benchmarking.

In particular, this toolbox is primarily designed for people developing new low-rank factorization models. Hence, it should be easy to add new NMF algorithms.

**This toolbox is still under active development. Any help or comment is welcomed!**

## Features

- **Standardized Evaluation Metrics**: Implement common metrics for evaluating the performance of NMF algorithms on audio data.
- **Dataset Handling**: Tools for loading, pre-processing, and managing audio datasets.
- **Baseline Algorithms**: Include implementations of standard NMF algorithms (only beta-divergence NMF without constraints for now).
- **Extensibility**: Easily add new NMF models and compare their performance against existing baselines.

## Installation

To install the toolbox, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/nmf-audio-benchmark.git
cd nmf-audio-benchmark
pip install -r requirements.txt
```

Additional requirements can be installed following the specific task.

## Organization of the Toolbox
The toolbox is organized into several modules:

- algorithms: This module includes the implementations of the NMF algorithms. Ideally, it provides a standardized interface for adding new NMF models. A template file is present in the folder, to help you add a new NMF algorithm.

- tasks: This module defines different tasks for which NMF algorithms can be benchmarked. For now, the tasks are:
    1. **Music Source Separation**: Separating the signals from the different sources in the audio.
    1. **Music Transcription**: Transcribe a spectrogram into notes and their activations. Focus on the piano for now. Note: as for now, this task is done pretty naïvely, and could be enhanced.
    1. **Music Structure Analysis**: Estimates the structure of a song.

Example notebooks are available for each task, to help you get into the code.

- dataloaders: This module contains utilities for loading and pre-processing audio datasets. It includes functions to handle various audio formats and prepare data for NMF algorithms. Dataloaders are designed following the different tasks. Available dataloaders are:
    1. **Music Source Separation**: MusDB18.
    1. **Music Transcription**: MAPS.
    1. **Music Structure Analysis**: RWC Pop, SALAMI and The Beatles.

- benchmarks: This module contains scripts and tools for benchmarking NMF algorithms. It includes standardized evaluation metrics and comparison tools to assess the performance of different models.

## DOCE
The benchmark uses the doce package for the management of the experiments.
See the documentation for more details about the doce package: https://doce.readthedocs.io/

In a nutshell, benchmarks are handled using the command-line interface.
- Run a benchmark using `python <benchmark.py> -c`
- Evaluate the metrics using `python <benchmark.py> -d`
- Control the plans (different conditions of experiments) using `python <benchmark.py> -s <plan_name>`
- Tag experiments using `python <benchmark.py> -t <tag_name>`

The experiment outputs are stored in a folder named 'experiments_outputs' and stored at the same tree level than the one where you run the benchmark.

TODO: add a fixed tree structure for experiment outputs in the future.

# Contact
Axel Marmoret - axel.marmoret@imt-atlantique.fr

# Future work
## NMF developments
For now, only the unconstrained NMF is developed. Sparse and Min-vol NMF should soon follow.

Feel free to add your own models! This is the main reason for this toolbox.

In addition, State-of-the-Art models (such as [1]) should be added to the toolbox.

## Tasks
The current tasks could be enhanced. In particular, the Transcription task, which is done in a naïve way now.

New tasks could also be added, for instance in general audio processing (speech) or bioacoustics (bioacoustics source separation or sound event detection).

## Tensor models
Tensor models constitute a very active literature, which should be tackled in the current toolbox. In particular, tensor models for audio processing already exist (for instance [2, 3]).

## GPU-intended NMF
For now, NMF models run on CPU. GPU are known to be very efficient for matrix computation. Hence, adapting the code for GPU computation (for instance using Tensorly, which enables compatibility with PyTorch) should be a major advantage and time gain.

## References

[1] Bertin, N., Badeau, R., & Vincent, E. (2010). Enforcing harmonicity and smoothness in Bayesian non-negative matrix factorization applied to polyphonic music transcription. IEEE Transactions on Audio, Speech, and Language Processing, 18(3), 538-549.

[2] Ozerov, A., & Févotte, C. (2009). Multichannel nonnegative matrix factorization in convolutive mixtures for audio source separation. IEEE transactions on audio, speech, and language processing, 18(3), 550-563.

[3] Marmoret, A., Bertin, N., & Cohen, J. (2019). Multi-Channel Automatic Music Transcription Using Tensor Algebra. arXiv preprint arXiv:2107.11250.
