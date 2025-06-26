---
title: 'NMF Audio Benchmark: A Toolbox for Benchmarking Nonnegative Matrix Factorization in Audio Processing'
tags:
- Python
- audio processing
- music information retrieval
- nonnegative matrix factorization
- benchmarking

authors:
- name: Axel Marmoret
corresponding: true
affiliation: 1
orcid: 0000-0001-6928-7490

affiliations:
- name: IMT Atlantique, Lab-STICC, UMR CNRS 6285, Brest, France
index: 1

date: 23 June 2025
bibliography: paper.bib
---

# Summary

Nonnegative Matrix Factorization (NMF) is a numerical linear algebra technique, with applications for modeling and analyzing audio data [@lee1999learning; @gillis2020nonnegative]. It has been successfully applied to a range of Music Information Retrieval (MIR) tasks, and notably music source separation [@virtanen2003sound; @vincent2018audio], automatic music transcription [@smaragdis2003non; @bertin2010enforcing; @benetos2018automatic], and music structure analysis [@nieto2020segmentationreview; @marmoret2022barwise]. It has also been applied to other audio domains, such as Speech Processing [@mohammadiha2013supervised; @shimada2019unsupervised] and ecoacoustics [@lin2017improving; @lin2020source], but, in a first approximation, we focus towards MIR applications, with the hope that future work will expand to numerous audio domains.

Over the past decade, the interest in NMF for MIR applications has diminished, in a large part due to the superior empirical performance of deep learning approaches in tasks where NMF was standard, such as source separation [@hennequin2020spleeter; @rouard2023hybrid], transcription [@bittner2022lightweight; @cwitkowitz2024timbre; @riley2024high], and structure analysis [@grill2015music; @buisson2024self]. As a result, recent MIR literature has featured relatively few NMF-based methods. Nevertheless, NMF retains several appealing characteristics --- such as interpretability, low computational requirements, and unsupervised learning capabilities --- that make it particularly suitable for scenarios where deep learning may be less effective. For instance, domains with scarce annotated data (for instance music with rare or non-standard instruments), or for historical recordings with atypical and/or degraded recording conditions could benefit from NMF developments. Hence, we argue that NMF remains a complementary modeling approach, and continued research on its use in MIR appears relevant.

Meanwhile, methodological research on NMF continues to evolve independently of its application in MIR. Recent developments include new algorithmic constraints (_e.g.,_ minimum-volume NMF [@leplat2020blind], multi-objective NMF [@gillis2021distributionally], and multi-resolution NMF [@leplat2022multi]) and novel paradigms such as deep NMF [@le2015deep; @leplat2024deep]. These innovations are sometimes demonstrated using audio tasks, yet their evaluation typically involves limited experimental settings --- frequently relying on a small number of examples rather than comprehensive benchmarking datasets (_e.g.,_ [@leplat2020blind; @gillis2021distributionally; @leplat2022multi]). This may reflect the inherent challenge of designing standard benchmarking protocols that require expertise in both MIR and numerical linear algebra.

To address this challenge, we introduce `nmf_audio_benchmark`, a Python toolbox designed to support systematic and reproducible evaluation of NMF-based methods in audio processing. The framework offers standardized evaluation metrics, integrated dataset handling, baseline algorithms, and tools for experiment management.

By positioning itself at the intersection of the MIR and numerical linear algebra communities, this toolbox aims to:
- Enable MIR researchers to systematically assess the performance of recent NMF-based techniques using established evaluation protocols.
- Assist researchers in numerical linear algebra by providing a ready-to-use benchmarking pipeline, allowing them to focus on algorithmic innovation without the overhead of developing application-specific evaluation frameworks.

# Statement of need

We believe that the numerical linear algebra research community lacks an easy- and ready-to-use platform for evaluating NMF models across different audio tasks, where NMF has been particularly important in the past decades. While many novel approaches in NMF methodology introduce improvements in optimization, constraints, or objective functions, comparing them under common real-world conditions remains cumbersome. 

`nmf_audio_benchmark` addresses this issue by:

- Implementing baseline NMF algorithms.

- Enabling simple extension with new NMF algorithms, tasks, and datasets.

- Providing code handling the MIR aspects of the task once the NMF decomposition is computed.

- Providing dataset loaders and pre-processing tools for standard datasets. All MIR dataloaders are based on the `mirdata` [@mirdata] toolbox.

- Leverages the standard Hydra [@yadan2019Hydra] toolbox to define benchmarks.

In practice, new NMF models are often demonstrated on audio tasks, where they may show improved performance. Providing a shared framework for evaluation and comparison makes it easier to benchmark such methods against existing baselines. This allows researchers in numerical linear algebra to focus on algorithmic development, while presenting results in line with standards in the audio literature.

## Why NMF?

This toolbox is designed with recent developments in NMF in mind. It is primarily intended for researchers developing new NMF-like methods, to help them evaluate their models on real audio tasks. While deep learning dominates many current approaches, NMF has played a key role in audio processing over the past decades and remains relevant, particularly for out-of-distribution settings or data-scarce domains, or in scenarios where interpretability, simplicity, or incorporating specific prior knowledge are needed. Hence, we believe that NMF-like methods, in particular refined models, may be still relevant for solving audio tasks.

We also believe that NMF remains valuable for MIR practitioners, especially in domains where deep learning models struggle. By making it easier to evaluate and compare NMF methods, this toolbox may help reintroduce NMF into practical workflows and help solve practical audio and MIR tasks.

## Why not deep learning methods?

Deep learning methods currently dominate the landscape of MIR and audio modeling, thanks to their strong performance across many tasks — including those covered in this benchmark — such as source separation [@hennequin2020spleeter; @rouard2023hybrid], transcription [@bittner2022lightweight; @cwitkowitz2024timbre; @riley2024high], and structure analysis [@grill2015music; @buisson2024self]. These models are typically application-driven, and new methods are often benchmarked directly against existing approaches in scientific papers. A benchmarking toolbox may hence appear irrelevant for deep learning methods. Furthermore, deep learning models evolve rapidly and rely on diverse paradigms making it difficult to design a unified, lasting benchmarking framework that stays relevant over time.

By contrast, NMF is expected to remain useful in settings where deep learning is less effective --- for example, when annotated data is scarce, domain conditions are unusual, or interpretability is important. In such cases, meaningful comparisons should be made under consistent assumptions and experimental conditions. Benchmarking in this context is not about competing with deep models, but about evaluating the potential of a constraint or a paradigm compared to very similar methods.

While deep learning baselines could eventually be included for context, we believe the primary focus of this toolbox should remain on comparing NMF-based methods. Benchmarking here is intended to support the development and evaluation of new constraints, objective functions, or algorithmic paradigms within the NMF framework. Comparisons with deep models --- which often solve tasks with different goals, priors, and data requirements --- are best left to individual researchers, depending on the framing and scope of their work.

## Why Audio?

Audio is a particularly suitable domain for NMF: many audio signals exhibit additive structures, repeating patterns, and nonnegativity (_e.g.,_ magnitudes of time-frequency representations), making NMF a natural modeling choice for audio signals. These properties, along with the interpretability of NMF components, have led to its longstanding success in audio tasks. As a result, evaluating new NMF methods on audio not only provides practical benchmarks but also leverages a domain where NMF has strong conceptual grounding and proven relevance.

# Design of the toolbox

The toolbox is separated in four modules:
- algorithms --- the NMF algorithms.

- tasks --- the audio tasks.

- dataloaders --- the dataloaders for supported datasets. Each dataloader is associated with a task, to pre-process data and annotations according to the requirements of the task.

- benchmarks --- the benchmark defining code. Benchmarks are supported by Hydra [@yadan2019Hydra].

The modular design is justified by the fact that, as such, it is possible to add any component (algorithm, task, dataset, or benchmark) without having to tackle the other aspects. Hence, we hope that communities specialized in one of these aspects will be able to contribute.

# Acknowledgements

We would like to thank the contributors of the datasets and open-source packages that made this work possible. Feedback and contributions are welcomed and encouraged via GitHub.

# References

