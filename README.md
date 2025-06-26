# NMF Audio Benchmark

This is a repository aimed at facilitating the benchmarking of NMF-based techniques in the context of audio processing.

## TL;DR

This toolbox was developed in response to the ongoing development of new low-rank factorization models (in particular under new constraints or objective functions), but testing them under real conditions with audio data is not an easy task. In that spirit, this toolbox was primarily created to:

- Provide a standardized framework for evaluating NMF-based audio processing techniques.
- Offer a collection of audio datasets and pre-processing tools.
- Include a set of baseline NMF algorithms for comparison.
- Enable easy integration of new NMF models for benchmarking.

This toolbox is mainly intended for researchers developing new low-rank factorization models. Hence, it should be easy to add new NMF algorithms (at least, this is intended; hence, don't hesitate to make comments).

**This toolbox is still under active development. Any help or comment is welcomed!**

## Summary

Nonnegative Matrix Factorization (NMF) is a numerical linear algebra technique, with applications for modeling and analyzing audio data [@lee1999learning; @gillis2020nonnegative]. It has been successfully applied to a range of Music Information Retrieval (MIR) tasks, and notably music source separation [@virtanen2003sound; @vincent2018audio], automatic music transcription [@smaragdis2003non; @bertin2010enforcing; @benetos2018automatic], and music structure analysis [@nieto2020segmentationreview; @marmoret2022barwise]. It has also been applied to other audio domains, such as Speech Processing [@mohammadiha2013supervised; shimada2019unsupervised] and ecoacoustics [@lin2017improving; @lin2020source], but, in a first approximation, we focus towards MIR applications, with the hope that future work will expand to numerous audio domains.

Over the past decade, the interest in NMF for MIR applications has diminished, in a large part due to the superior empirical performance of deep learning approaches in tasks where NMF was standard, such as source separation [@hennequin2020spleeter; @rouard2023hybrid], transcription [@bittner2022lightweight; @cwitkowitz2024timbre; @riley2024high], and structure analysis [@grill2015music; @buisson2024self]. As a result, recent MIR literature has featured relatively few NMF-based methods. Nevertheless, NMF retains several appealing characteristics --- such as interpretability, low computational requirements, and unsupervised learning capabilities --- that make it particularly suitable for scenarios where deep learning may be less effective. For instance, domains with scarce annotated data (for instance music with rare or non-standard instruments), or for historical recordings with atypical and/or degraded recording conditions could benefit from NMF developments. Hence, we argue that NMF remains a complementary modeling approach, and continued research on its use in MIR appears relevant.

Meanwhile, methodological research on NMF continues to evolve independently of its application in MIR. Recent developments include new algorithmic constraints (_e.g.,_ minimum-volume NMF [@leplat2020blind], multi-objective NMF [@gillis2021distributionally], and multi-resolution NMF [@leplat2022multi]) and novel paradigms such as deep NMF [@le2015deep; @leplat2024deep]. These innovations are sometimes demonstrated using audio tasks, yet their evaluation typically involves limited experimental settings --- frequently relying on a small number of examples rather than comprehensive benchmarking datasets (_e.g.,_ [@leplat2020blind; @gillis2021distributionally; @leplat2022multi]). This may reflect the inherent challenge of designing standard benchmarking protocols that require expertise in both MIR and numerical linear algebra.

To address this challenge, we introduce `nmf_audio_benchmark`, a Python toolbox designed to support systematic and reproducible evaluation of NMF-based methods in audio processing. The framework offers standardized evaluation metrics, integrated dataset handling, baseline algorithms, and tools for experiment management.

By positioning itself at the intersection of the MIR and numerical linear algebra communities, this toolbox aims to:
- Enable MIR researchers to systematically assess the performance of recent NMF-based techniques using established evaluation protocols.
- Assist researchers in numerical linear algebra by providing a ready-to-use benchmarking pipeline, allowing them to focus on algorithmic innovation without the overhead of developing application-specific evaluation frameworks.

## Why this toolbox?

We believe that the numerical linear algebra research community lacks an easy- and ready-to-use platform for evaluating NMF models across different audio tasks, where NMF has been particularly important in the past decades. While many novel approaches in NMF methodology introduce improvements in optimization, constraints, or objective functions, comparing them under common real-world conditions remains cumbersome. 

`nmf_audio_benchmark` addresses this issue by:

- Implementing baseline NMF algorithms.

- Enabling simple extension with new NMF algorithms, tasks, and datasets.

- Providing code handling the MIR aspects of the task once the NMF decomposition is computed.

- Providing dataset loaders and pre-processing tools for standard datasets. All MIR dataloaders are based on the `mirdata` [@mirdata] toolbox.

In practice, new NMF models are often demonstrated on audio tasks, where they may show improved performance. Providing a shared framework for evaluation and comparison makes it easier to benchmark such methods against existing baselines. This allows researchers in numerical linear algebra to focus on algorithmic development, while presenting results in line with standards in the audio literature.

### Why NMF?

This toolbox is designed with recent developments in NMF in mind. It is primarily intended for researchers developing new NMF-like methods, to help them evaluate their models on real audio tasks. While deep learning dominates many current approaches, NMF has played a key role in audio processing over the past decades and remains relevant, particularly for out-of-distribution settings or data-scarce domains, or in scenarios where interpretability, simplicity, or incorporating specific prior knowledge are needed. Hence, we believe that NMF-like methods, in particular refined models, may be still relevant for solving audio tasks.

We also believe that NMF remains valuable for MIR practitioners, especially in domains where deep learning models struggle. By making it easier to evaluate and compare NMF methods, this toolbox may help reintroduce NMF into practical workflows and help solve practical audio and MIR tasks.

### Why not deep learning methods?

Deep learning methods currently dominate the landscape of MIR and audio modeling, thanks to their strong performance across many tasks — including those covered in this benchmark — such as source separation [@hennequin2020spleeter; @rouard2023hybrid], transcription [@bittner2022lightweight; @cwitkowitz2024timbre; @riley2024high], and structure analysis [@grill2015music; @buisson2024self]. These models are typically application-driven, and new methods are often benchmarked directly against existing approaches in scientific papers. A benchmarking toolbox may hence appear irrelevant for deep learning methods. Furthermore, deep learning models evolve rapidly and rely on diverse paradigms making it difficult to design a unified, lasting benchmarking framework that stays relevant over time.

By contrast, NMF is expected to remain useful in settings where deep learning is less effective --- for example, when annotated data is scarce, domain conditions are unusual, or interpretability is important. In such cases, meaningful comparisons should be made under consistent assumptions and experimental conditions. Benchmarking in this context is not about competing with deep models, but about evaluating the potential of a constraint or a paradigm compared to very similar methods.

While deep learning baselines could eventually be included for context, we believe the primary focus of this toolbox should remain on comparing NMF-based methods. Benchmarking here is intended to support the development and evaluation of new constraints, objective functions, or algorithmic paradigms within the NMF framework. Comparisons with deep models --- which often solve tasks with different goals, priors, and data requirements --- are best left to individual researchers, depending on the framing and scope of their work.

### Why Audio?

Audio is a particularly suitable domain for NMF: many audio signals exhibit additive structures, repeating patterns, and nonnegativity (_e.g.,_ magnitudes of time-frequency representations), making NMF a natural modeling choice for audio signals. These properties, along with the interpretability of NMF components, have led to its longstanding success in audio tasks. As a result, evaluating new NMF methods on audio not only provides practical benchmarks but also leverages a domain where NMF has strong conceptual grounding and proven relevance.

## Technical Details

### Design of the toolbox

The toolbox is separated in four modules:
- algorithms --- the NMF algorithms.
    - For now, it only supports the unconstrained NMF with beta-divergence, without additional regularization or constraints. It contains two implementations: one from scikit-learn [@pedregosa2011scikit], and one from nn_fac [@marmoret2020nn_fac].

- tasks --- the audio tasks.
    - For now, only three MIR tasks are supported: Music Source Separation, Music Transcription and Music Structure Analysis. See Section "Tasks (and datasets) Supported" for more details.

- dataloaders --- the dataloaders for supported datasets. Each dataloader is associated with a task, to pre-process data and annotations according to the requirements of the task.
    - For now, only five datasets are supported. See Section "Tasks (and datasets) Supported" for more details.

- benchmarks --- the benchmark defining code. Benchmarks are supported by Hydra [@Yadan2019Hydra].

The modular design is justified by the fact that, as such, it is possible to add any component (algorithm, task, dataset, or benchmark) without having to tackle the other aspects. Hence, we hope that communities specialized in one of these aspects will be able to contribute.

### Tasks (and datasets) Supported

- Music Source Separation --- Separate sources from mixed audio signals [@vincent2018audio].
    - Supports the MusDB18 [@MusDB] dataset for now.

- Music Transcription --- Detect and localize notes in time and frequency [@Benetos2018].
    - Supports the MAPS [@MAPS] dataset for now.

- Music Structure Analysis --- Segment songs into structural parts [@Nieto2020].
    - Supports the RWC POP [@RWCPOP], SALAMI [@SALAMI] and The Beatles [@Beatles] datasets for now.

### Outputs

The outputs are the logs of the benchmark, and metrics obtained by using the NMF on a paticular task with a particular dataset.

### Installation

The code was developed using Python 3.12, and numpy version 1.26.4. Using numpy v2.* may cause errors when installing some dependencies, sorry for the inconvenience.

To install the toolbox, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/nmf-audio-benchmark.git
cd nmf-audio-benchmark
pip install -r requirements.txt
```

Additional requirements can be installed depending on the specific task (e.g., `requirements_msa.txt` or `requirements_mss.txt`).

## Future Work

### NMF developments
Constrained NMF models: Sparse NMF [@le2015sparse, @cohen2025efficient], minimum-volume [@leplat2020blind], Convolutive NMF [@smaragdis2004nmfd; @wu2022semi] and other constrained NMF variants [@bertin2010enforcing; @gillis2021distributionally; @leplat2022multi].

### Tasks
The current tasks could be enhanced. In particular, the Transcription and Source Separation tasks, which are done in a rather naïve way now.

New tasks could also be added, for instance in general audio processing (speech) or ecoacoustics (ecoacoustics source separation or sound event detection).

### Datasets
Many datasets could be added.

### Tensor models
Tensor models constitute a very active literature, which should be tackled in the current toolbox. In particular, tensor models for audio processing already exist (_e.g.,_ [@ozerov2009multichannel; @marmoret2019multi; @marmoret2020uncovering]).

### GPU-intended NMF
For now, NMF models run on CPUs. GPUs are known to be very efficient for matrix computation. Hence, adapting the code for GPU computation (for instance using Tensorly [@tensorly], which enables compatibility with PyTorch) should be a major advantage and time gain.

## Acknowledgements

We would like to thank the contributors of the datasets and open-source packages that made this work possible. Feedback and contributions are welcomed and encouraged via GitHub.

## Contact
Axel Marmoret - axel.marmoret@imt-atlantique.fr


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

