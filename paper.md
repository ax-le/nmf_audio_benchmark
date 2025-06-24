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

# Summary

Nonnegative Matrix Factorization (NMF) is a numerical linear algebra technique, with applications for modeling and analyzing audio data [@lee1999learning; @gillis2020nonnegative]. It has been successfully applied to a range of Music Information Retrieval (MIR) tasks, and notably music source separation [@virtanen2003sound; @vincent2018audio], automatic music transcription [@smaragdis2003non; @bertin2010enforcing; @benetos2018automatic], and music structure analysis [@nieto2020segmentationreview; @marmoret2022barwise]. It has also been applied to other audio domains, such as Speech Processing [@mohammadiha2013supervised; shimada2019unsupervised] and ecoacoustics [@lin2017improving; @lin2020source], but, in a first approximation, we focus towards MIR applications, with the hope that future work will expand to numerous audio domains.

Over the past decade, the interest in NMF for MIR applications has diminished, in a large part due to the superior empirical performance of deep learning approaches in tasks where NMF was standard, such as source separation [@hennequin2020spleeter; @rouard2023hybrid], transcription [@bittner2022lightweight; @cwitkowitz2024timbre; @riley2024high], and structure analysis [@grill2015music; @buisson2024self]. As a result, recent MIR literature has featured relatively few NMF-based methods. Nevertheless, NMF retains several appealing characteristics --- such as interpretability, low computational requirements, and unsupervised learning capabilities --- that make it particularly suitable for scenarios where deep learning may be less effective. For instance, domains with scarce annotated data (for instance music with rare or non-standard instruments), or for historical recordings with atypical and/or degraded recording conditions could benefit from NMF developments. In this context, we believe that NMF remains a complementary modeling approach, and continued research on its use in MIR appears relevant.

Meanwhile, methodological research on NMF continues to evolve independently of its application in MIR. Recent developments include new algorithmic constraints (_e.g.,_ minimum-volume NMF [@leplat2020blind], multi-objective NMF [@gillis2021distributionally], and multi-resolution NMF [@leplat2022multi]) and novel paradigms such as deep NMF [@le2015deep; @leplat2024deep]. These innovations are sometimes demonstrated using audio tasks, yet their evaluation typically involves limited experimental settings --- frequently relying on a small number of examples rather than comprehensive benchmarking datasets (_e.g.,_ [@leplat2020blind; @gillis2021distributionally; @leplat2022multi]). This may reflect the inherent challenge of designing standard benchmarking protocols that require expertise in both MIR and numerical linear algebra.

To address this challenge, we introduce `nmf_audio_benchmark`, a Python toolbox designed to support systematic and reproducible evaluation of NMF-based methods in audio processing. The framework offers standardized evaluation metrics, integrated dataset handling, baseline algorithms, and tools for experiment management.

By positioning itself at the intersection of the MIR and numerical linear algebra communities, this toolbox aims to:
- Enable MIR researchers to systematically assess the performance of recent NMF-based techniques using established evaluation protocols.
- Assist researchers in numerical linear algebra by providing a ready-to-use benchmarking pipeline, allowing them to focus on algorithmic innovation without the overhead of developing application-specific evaluation frameworks.

# Statement of need

We believe that the numerical linear algebra research community lacks a consistent and modular benchmark for evaluating NMF models across different audio tasks, where NMF has been particularly important in the past decades. While many novel approaches introduce improvements in optimization, constraints, or objective functions, comparing them under common real-world conditions remains cumbersome. 

`nmf_audio_benchmark` addresses this issue by:

- Offering a unified benchmarking pipeline across different audio tasks.

- Providing dataset loaders and pre-processing tools for standard datasets. All MIR dataloaders are based on the `mirdata` [@mirdata] toolbox.

- Implementing baseline NMF algorithms.

- Using Hydra [@Hydra] to manage configurations and experiment workflows.

- Enabling simple extension with new NMF algorithms, tasks, and datasets.

This toolbox is aimed primarily at researchers developing novel NMF-like methods, offering them a basis for evaluation and comparison with other models in audio tasks. We also believe that an easy-to-use platform for evaluating NMF methods would benefit MIR researchers, as this method may work in domains where the most recent deep learning models could fail (in particular domains where annotated data is scarce).

# Technical Details

## Design of the toolbox

The toolbox is separated in four modules:
- algorithms --- the NMF algorithms.
    - For now, it only supports the unconstrained NMF with beta-divergence, without additional regularization or constraints. It contains two implementations: one from scikit-learn [@pedregosa2011scikit], and one from nn_fac [@marmoret2020nn_fac].

- tasks --- the audio tasks.
    - For now, only three MIR tasks are supported: Music Source Separation, Music Transcription and Music Structure Analysis. See Section "Tasks (and datasets) Supported" for more details.

- dataloaders --- the dataloaders for supported datasets. Each dataloader is associated with a task, to pre-process data and annotations according to the requirements of the task.
    - For now, only five datasets are supported. See Section "Tasks (and datasets) Supported" for more details.

- benchmarks --- the benchmark defining code. Benchmarks are supported by Hydra [@Yadan2019Hydra].

The modular design is justified by the fact that, as such, it is possible to add any component (algorithm, task, dataset, or benchmark) without having to tackle the other aspects. Hence, we hope that communities specialized in one of these aspects will be able to contribute.

## Tasks (and datasets) Supported

- Music Source Separation --- Separate sources from mixed audio signals [@vincent2018audio].
    - Supports the MusDB18 [@MusDB] dataset for now.

- Music Transcription --- Detect and localize notes in time and frequency [@Benetos2018].
    - Supports the MAPS [@MAPS] dataset for now.

- Music Structure Analysis --- Segment songs into structural parts [@Nieto2020].
    - Supports the RWC POP [@RWCPOP], SALAMI [@SALAMI] and The Beatles [@Beatles] datasets for now.

## Outputs

The outputs are the logs of the benchmark, and metrics obtained by using the NMF on a paticular task with a particular dataset.

## Installation
The toolbox may be installed by downloading the source code on Github, and by then using pip in local mode (`pip install -e /path/to/nmf_audio_benchmark`). (Note: the "-e" is intended for developers of the toolbox, and we recommend using it in order to adapt some bits of code to your needs.)

## Future Work

- Constrained NMF models: Sparse NMF [@le2015sparse, @cohen2025efficient], minimum-volume [@leplat2020blind], Convolutive NMF [@smaragdis2004nmfd; @wu2022semi] and other constrained NMF variants [@bertin2010enforcing; @gillis2021distributionally; @leplat2022multi].

- Extension to Tensor Models, _e.g._ [@ozerov2009multichannel; @marmoret2019multi; @marmoret2020uncovering].

- GPU usage for matrix multiplications: for now only CPU computation is supported, while GPU computation would be probably be faster. One way to implement such feature could be to integrate TensorLy [@tensorly].

- New Tasks: Speech enhancement, ecoacoustics, and sound event detection in general.

- Improved Transcription Models: Replace current naïve implementation.

# Acknowledgements

We would like to thank the contributors of the datasets and open-source packages that made this work possible. Feedback and contributions are welcomed and encouraged via GitHub.

# References

