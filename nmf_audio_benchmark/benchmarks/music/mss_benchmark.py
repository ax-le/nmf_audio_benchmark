"""
Benchmark definition for the Music Source Separation task.
Details for Music Source Separation with NMF can be found in [1, Chap. 8.1]

The benchmark uses the doce package for the management of the experiments.
See the documentation for more details about the doce package: https://doce.readthedocs.io/

In a nutshell, the benchmark is defined by the following steps:
- Define the experiment with the doce.Experiment object.
- Define the constant parameters using experiment.__name__ = value
- Define the parameters to vary using experiment.add_plan('__name__', **params).
    - Note: several plans allow to define specific parameters for different parts of the experiment.
- Define the step function, which will be called for each setting of the experiment.
    - The step function should take two arguments: setting and experiment, which coontain the parameters defined in the previous steps.
    - The step function should return the results of the experiment, which will be saved in the output directory.
- Run the experiment using the Command Line Interface
    - The CLI will run the step function for each setting of the experiment.
    - To actually run the experiment, use the command: python msa_benchmark.py -c
    - See more details details on the CLI in the doce documentation: https://doce.readthedocs.io/en/latest/tutorial.html
- To analyze the results, use the doce package to load the results and analyze them.
    - Either using the CLI: python mss_benchmark.py -d
    - Or analyzing the results in a Python script. See "utils.visualize_outputs.py" for an example.

References:
[1] Vincent, E., Virtanen, T., & Gannot, S. (Eds.). (2018). Audio source separation and speech enhancement. John Wiley & Sons.
"""
# Define the dataloaders
import nmf_audio_benchmark.dataloaders.music.mss_dataloader as mss_dl
# Define the task
import nmf_audio_benchmark.tasks.music.mss as mss
# Define the algorithm
import nmf_audio_benchmark.algorithms.nn_fac_algos as nn_fac_algos

import tqdm
import numpy as np
import doce

# Instantiante the experiment.
experiment = doce.Experiment(name = 'mss_benchmark',
                             purpose = 'Benchmarking NMF methods on the Music Source Separation task',
                             author = 'Axel Marmoret',
                             address = 'axel.marmoret@imt-atlantique.fr',
                             )

# Path to the dataset
datapath = '/home/a23marmo/datasets/musdb18'

# Dataset class. The object will be initialized in the step function.
dataset_object = mss_dl.MusDBDataloader

# Store the outputs, logs and other files in a specific directory
experiment.set_path('output', f'./experiments_outputs/{experiment.name}/{dataset_object.name}/')

# Fixed parameters for the experiment. Will not appear in the name of the output files
experiment.feature = "stft_complex"
experiment.nb_sources = 4

# Set the parameters which are going to be varied in the experiment
experiment.add_plan('harmonic',
    # dataset params
    # No parameters for the dataset

    # NMF params
    rank = [88],
    beta = [2,1,0],
    nmf_type = "unconstrained",
    init = ["harmonic"],

    # Task params
    # no params, due to the cross validation
)

experiment.add_plan('nndsvd',
    # dataset params
    # No parameters for the dataset

    # NMF params
    rank = [10,20,40,60],
    beta = [2,1,0],
    nmf_type = "unconstrained",
    init = ["nndsvd"],

    # Task params
    # No parameters for the task
)

def step(setting, experiment):
    # Instanciate the dataset object
    dataset = dataset_object(datapath=datapath, feature = experiment.feature, cache_path = f"{datapath}/cache", chunk_duration=30)

    # Instanciate the NMF object
    nmf = nn_fac_algos.nn_fac_NMF(setting.rank, setting.beta, init = setting.init, nmf_type=setting.nmf_type,
                                  normalize=[True, False])

    # Instanciate the source separation object
    source_separation_object = mss.MusicSourceSeparation(feature_object=dataset.feature_object, nb_sources = experiment.nb_sources, phase_retrieval="original_phase")
    
    # Compute the source separation. Scores are stored in a dictionary, sourcewise. The key "average" contains the average score for all sources.
    dict_all_si_sdr = compute_source_separation(dataset, nmf, source_separation_object)

    # Save the results of this benchmark
    save_path = f"{experiment.path.output}/{setting.identifier()}"

    for key in dict_all_si_sdr: # Save the results for each source and for the average
        np.save(f'{save_path}_si_sdrs_{key}.npy', np.array(dict_all_si_sdr[key]))

# Define the metrics which can be displayed using python <this_benchmark>.py -d
experiment.set_metric(
    name = 'si_sdrs_average',
    percent=False,
    higher_the_better= True,
)

# Defines metrics accordiing to each sources available in the dataset.
dataset = dataset_object(datapath=datapath, feature = "stft") # Generating a dataset to access the stems labels
for stem_label in dataset.all_stems: # Parsing the sources
    experiment.set_metric(
        name = f'si_sdrs_{stem_label}',
        func = np.nanmean, # Because sources which are never estiamted are set to NaN
        percent=False,
        higher_the_better= True,
    )

# Actually compute the source separation
def compute_source_separation(dataset, nmf, source_separation_object):
    """
    Compute the source separation for all the songs in the dataset, using the nmf and the source separation object.
    """
    # Initialize the dictionary with NaN scores for each source.
    dict_all_si_sdr = {}
    dict_all_si_sdr["average"] = []
    for stem_label in dataset.all_stems:
        dict_all_si_sdr[stem_label] = [np.nan]

    # Iterate over all the songs in the dataset
    for idx_song in tqdm.tqdm(range(len(dataset))):
        # One song
        track_id, (mag, phase), stems, stems_labels = dataset[idx_song]

        # Compute NMF
        W, H = nmf.run(data=mag, feature_object=dataset.feature_object)

        # Estimate the individual sources
        estimated_sources = source_separation_object.predict(W, H, phase=phase)

        # Compute the scores and store them
        si_sdr, snr, idx_argmax = source_separation_object.score(estimated_sources, stems)

        # Store the average SI-SDR for all estimated sources
        dict_all_si_sdr["average"].append(np.mean(si_sdr))

        # Store the SI-SDR source-wise
        result_sourcewise_si_sdr = mss.average_scores_sourcewise(si_sdr, idx_argmax, stems_labels)

        # Update the dictionary, with source-wise scores.
        for key in result_sourcewise_si_sdr:
            if dict_all_si_sdr[key][0] is np.nan: # First time that this source is estimated
                dict_all_si_sdr[key][0] = result_sourcewise_si_sdr[key] # Replace the None value by the first result
            else:
                dict_all_si_sdr[key].append(result_sourcewise_si_sdr[key]) # Append the score to the list

    return dict_all_si_sdr

if __name__ == "__main__":
    # Invoke the command line management of the doce package
    doce.cli.main(experiment = experiment,
                  func = step)