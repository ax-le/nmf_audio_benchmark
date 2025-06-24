title: 'NMF Audio Benchmark: A Toolbox for Benchmarking Nonnegative Matrix Factorization in Audio Processing'
tags:

Python

audio processing

nonnegative matrix factorization

benchmarking

music information retrieval
authors:

name: Axel Marmoret
corresponding: true
affiliation: 1
orcid: 0000-0000-0000-0000 # replace with actual ORCID if available
affiliations:

name: IMT Atlantique, France
index: 1
date: 23 June 2025
bibliography: paper.bib

Summary

Nonnegative Matrix Factorization (NMF) is a simple yet powerful technique for modeling and analyzing audio data [@]. Its applications range from music source separation to transcription and structure analysis.

In the most recent yeas, NMF-based have been supplanted by deep learning-based techniques, with a major 

 Despite extensive research on algorithmic improvements and model variations, benchmarking these models in realistic audio contexts remains challenging due to the lack of standardized evaluation frameworks and accessible tooling.

NMF Audio Benchmark is a Python toolbox designed to bridge this gap. It provides a modular, extensible, and task-oriented framework for evaluating NMF-based methods on real-world audio tasks. It includes standardized evaluation metrics, dataset handlers, baseline algorithms, and tools for configuring and running reproducible experiments.

Statement of need

NMF-based methods have been widely used in audio signal processing, particularly for tasks such as source separation, music transcription, and structural analysis. These techniques are valued for their simplicity, interpretability, and flexibility in incorporating various constraints or divergence measures.



The audio research community lacks a consistent and modular benchmark for evaluating NMF models across different audio tasks. While many novel approaches introduce improvements in optimization, constraints, or objective functions, comparing them under common real-world conditions remains cumbersome and inconsistent.

nmf-audio-benchmark addresses this issue by:

Offering a unified benchmarking pipeline across tasks like music source separation, transcription, and structure analysis.

Providing dataset loaders and pre-processing tools for widely used datasets such as MusDB18 [@MusDB], MAPS [@MAPS], RWC Pop [@RWCPOP], SALAMI [@SALAMI], and The Beatles [@Beatles].

Implementing baseline NMF algorithms with beta-divergence and no constraints for initial comparisons.

Using Hydra [@Hydra] to manage configurations and experiment workflows.

Enabling simple extension with new NMF algorithms and tasks.

This toolbox is aimed primarily at researchers developing novel NMF methods, offering them a solid basis for evaluation and comparison. Its modular design also enables integration with future developments, including tensor models, GPU acceleration, and new application domains.

Features

Standardized Evaluation: Unified performance metrics for all supported tasks.

Dataset Integration: Support for common audio datasets with task-specific pre-processing.

Baseline Implementations: Reference NMF algorithms for benchmarking.

Hydra-based Configuration: Elegant and reproducible experiment control.

Modular Design: Easily extensible with new algorithms, datasets, or tasks.

Tasks Supported

Music Source Separation — Separate sources from mixed audio signals [@VVG18].

Music Transcription — Detect and localize notes in time and frequency [@Benetos2018].

Music Structure Analysis — Segment songs into structural parts [@Nieto2020].

Future Work

Constrained NMF models: Sparse, min-vol, and Bayesian NMF variants [@BBV10].

Extension to Tensor Models: Integrating TensorLy [@Tensorly] for GPU-accelerated tensor factorization [@OF09; @MBC19].

New Tasks: Speech enhancement, bioacoustics, and sound event detection.

Improved Transcription Models: Replace current naïve implementation.

Acknowledgements

This project was developed at IMT Atlantique. Thanks to the contributors of the datasets and open-source packages that made this work possible. Feedback and contributions are welcomed via GitHub.

References

