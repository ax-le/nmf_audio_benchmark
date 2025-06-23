# NMF Audio Benchmark

This is a repository aimed at facilitating the benchmarking of NMF-based techniques in the context of audio processing.

## Overview

This toolbox was developed in response to the ongoing development of new low-rank factorization models (in particular under new constraints or objective functions), but testing them under real conditions with audio data is not an easy task. In that spirit, this toolbox was primarily created to:

- Provide a standardized framework for evaluating NMF-based audio processing techniques.
- Offer a collection of audio datasets and pre-processing tools.
- Include a set of baseline NMF algorithms for comparison.
- Enable easy integration of new NMF models for benchmarking.

This toolbox is mainly intended for researchers developing new low-rank factorization models. Hence, it should be easy to add new NMF algorithms (at least, this is intended; hence, don't hesitate to make comments).

**This toolbox is still under active development. Any help or comment is welcomed!**

## Features

- **Standardized Evaluation Metrics**: Implement common metrics for evaluating the performance of NMF algorithms on audio data.
- **Dataset Handling**: Tools for loading, pre-processing, and managing audio datasets.
- **Baseline Algorithms**: Include implementations of standard NMF algorithms (only beta-divergence NMF without constraints for now).
- **Extensibility**: Easily add new NMF models and compare their performance against existing baselines.

## Installation

The code was developed using Python 3.12, and numpy version 1.26.4. Using numpy v2.* may cause errors when installing some dependencies, sorry for the inconvenience.

To install the toolbox, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/nmf-audio-benchmark.git
cd nmf-audio-benchmark
pip install -r requirements.txt
```

Additional requirements can be installed depending on the specific task (e.g., `requirements_msa.txt` or `requirements_mss.txt`).
## Organization of the Toolbox
The toolbox is organized into several modules:

- algorithms: This module includes the implementations of the NMF algorithms. Ideally, it provides a standardized interface for adding new NMF models. A template file is present in the folder, to help you add a new NMF algorithm.

- tasks: This module defines different tasks for which NMF algorithms can be benchmarked. For now, the tasks are:
    1. **Music Source Separation**: Separating the signals from the different sources in the audio [VVG18].
    1. **Music Transcription**: Transcribe a spectrogram into notes and their activations [Ben+18]. Currently, this task focuses on the piano and is implemented in a rather naïve way - it could be significantly improved. Future work should tackle this aspect.
    1. **Music Structure Analysis**: Estimates the structure of a song [Nie+20].

Example notebooks are available for each task, to help you get into the code.

- dataloaders: This module contains utilities for loading and pre-processing audio datasets. It includes functions to handle various audio formats and prepare data for NMF algorithms. Dataloaders are organized according to the different tasks. Available dataloaders are:
    1. **Music Source Separation**: MusDB18 [MusDB].
    1. **Music Transcription**: MAPS [MAPS].
    1. **Music Structure Analysis**: RWC Pop [RWCPOP], SALAMI [SALAMI] and The Beatles [Beatles].

- benchmarks: This module contains scripts and tools for benchmarking NMF algorithms. It includes standardized evaluation metrics and comparison tools to assess the performance of different models. Benchmarks are parametrized using the `Hydra` toolbox [Hydra].

The modular design of the toolbox makes it easy to add new components. In particular, you can integrate a new algorithm with confidence that it will work with existing datasets and tasks - or add a new task that's compatible with all current algorithms and datasets. The downside is that this flexibility comes at the cost of a rigid structure, which may not always suit new developments. As a result, significant refactoring may be required to integrate certain future components.

# Contact
Axel Marmoret - axel.marmoret@imt-atlantique.fr

# Future work
## NMF developments
For now, only the unconstrained NMF is developed. Sparse and Min-vol NMF should soon follow.

Feel free to add your own models! This is the main reason for this toolbox.

In addition, State-of-the-Art models (such as [BBV10]) should be added to the toolbox.

## Tasks
The current tasks could be enhanced. In particular, the Transcription task, which is done in a naïve way now.

New tasks could also be added, for instance in general audio processing (speech) or bioacoustics (bioacoustics source separation or sound event detection).

## Tensor models
Tensor models constitute a very active literature, which should be tackled in the current toolbox. In particular, tensor models for audio processing already exist (for instance [OF09, MBC19]).

## GPU-intended NMF
For now, NMF models run on CPUs. GPUs are known to be very efficient for matrix computation. Hence, adapting the code for GPU computation (for instance using Tensorly [Tensorly], which enables compatibility with PyTorch) should be a major advantage and time gain.

## References

### Tasks

[VVG18] Vincent, E., Virtanen, T., & Gannot, S. (Eds.). (2018). Audio source separation and speech enhancement. John Wiley & Sons.

[Ben+18] Benetos, E., Dixon, S., Duan, Z., & Ewert, S. (2018). Automatic music transcription: An overview. IEEE Signal Processing Magazine, 36(1), 20-30.

[Nie+20] Nieto, O., Mysore, G. J., Wang, C. I., Smith, J. B., Schlüter, J., Grill, T., & McFee, B. (2020). Audio-based music structure analysis: Current trends, open challenges, and applications. Transactions of the International Society for Music Information Retrieval, 3(1).

### Datasets

[MusDB] Rafii, Z., Liutkus, A., Stöter, F. R., Mimilakis, S. I., & Bittner, R. (2017). The MUSDB18 corpus for music separation.

[MAPS] Emiya, V., Bertin, N., David, B., & Badeau, R. (2010). MAPS-A piano database for multipitch estimation and automatic transcription of music.

[RWCPOP] Goto, M., Hashiguchi, H., Nishimura, T., & Oka, R. (2002). RWC Music Database: Popular, classical and jazz music databases. In Proceedings of the International Society for Music Information Retrieval Conference (ISMIR) (pp. 287-288).

[SALAMI] Smith, J. B. L., Burgoyne, J. A., Fujinaga, I., De Roure, D., & Downie, J. S. (2011). Design and creation of a large-scale database of structural annotations. In Proceedings of the International Society for Music Information Retrieval Conference (ISMIR) (pp. 555-560). Miami, FL.

[Beatles] Harte, C. (2010). Towards automatic extraction of harmony information from music signals (Doctoral dissertation). Queen Mary University of London. -- http://isophonics.net/content/reference-annotations-beatles

### Benchmarks

[Hydra] Yadan, O. (2019). Hydra – A framework for elegantly configuring complex applications [Computer software]. GitHub. https://github.com/facebookresearch/hydra

### Future Work

[BBV10] Bertin, N., Badeau, R., & Vincent, E. (2010). Enforcing harmonicity and smoothness in Bayesian non-negative matrix factorization applied to polyphonic music transcription. IEEE Transactions on Audio, Speech, and Language Processing, 18(3), 538-549.

[OF09] Ozerov, A., & Févotte, C. (2009). Multichannel nonnegative matrix factorization in convolutive mixtures for audio source separation. IEEE transactions on audio, speech, and language processing, 18(3), 550-563.

[MBC19] Marmoret, A., Bertin, N., & Cohen, J. (2019). Multi-Channel Automatic Music Transcription Using Tensor Algebra. arXiv preprint arXiv:2107.11250.

[Tensorly] Kossaifi, J., Panagakis, Y., Anandkumar, A., & Pantic, M. (2019). TensorLy: Tensor learning in Python. Journal of Machine Learning Research, 20(26), 1–6. https://github.com/tensorly/tensorly

