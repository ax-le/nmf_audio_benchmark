"""
Benchmark definition for the Music Structure Analysis task.
Details for Music Structure Analysis with NMF can be found in [1], [2 Chap. 5.3].
Regarding the segmentation algorithm once NMF is computed, one can refer to [3], an article dedicated to the CBM segmentation algorithm, in addition to [2, Chap 3].

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
    - Either using the CLI: python msa_benchmark.py -d
    - Or analyzing the results in a Python script. See "utils.visualize_outputs.py" for an example.

References:
[1] Marmoret, A., Cohen, J. E., & Bimbot, F. (2022). "Barwise Compression Schemes for Audio-Based Music Structure Analysis." Sound and Music Computing 2022. 2022.
[2] Marmoret, A. (2022). Unsupervised Machine Learning Paradigms for the Representation of Music Similarity and Structure (Doctoral dissertation, Université Rennes 1).
[3] Marmoret, A., Cohen, J. E., & Bimbot, F. (2023). Barwise Music Structure Analysis with the Correlation Block-Matching Segmentation Algorithm. Transactions of the International Society for Music Information Retrieval (TISMIR), 6(1), 167-185.
"""
# Define the dataloaders
import nmf_audio_benchmark.dataloaders.msa_dataloader as msa_dl
# Define the task
import nmf_audio_benchmark.tasks.msa as msa
# Define the algorithm
import nmf_audio_benchmark.algorithms.nn_fac_algos as nn_fac_algos

import tqdm
import numpy as np
import doce

# Path to the dataset
datapath = '/home/a23marmo/datasets/rwcpop' # TODO: to change

# Dataset class. The object will be initialized in the step function.
dataset_object = msa_dl.RWCPopDataloader

# Instantiante the experiment.
experiment = doce.Experiment(name = 'msa_benchmark',
                             purpose = 'Benchmarking NMF methods on the Music Structure Analysis task',
                             author = 'Axel Marmoret',
                             address = 'axel.marmoret@imt-atlantique.fr',
                             )

# Store the outputs, logs and other files in a specific directory
experiment.set_path('output', f'./experiments_outputs/{experiment.name}/{dataset_object.name}/')

# Fixed parameters for the experiment. Will not appear in the name of the output files
experiment.init = "nndsvd"

# Set the parameters which are going to be varied in the experiment
experiment.add_plan('plan',
    rank = [10,20],
    beta = [2,1,0],
    feature = ["mel", "nn_log_mel"],
    nmf_type = "unconstrained"
)

def step(setting, experiment):
    # Instanciate the dataset object
    dataset = dataset_object(datapath, feature = setting.feature, cache_path = f"{datapath}/cache")

    # Instanciate the NMF object
    nmf = nn_fac_algos.nn_fac_NMF(setting.rank, setting.beta, init = experiment.init, nmf_type=setting.nmf_type,
                                  normalize=[False, True])
    
    # Instanciate the segmentation algorithm object (CBM algorithm, see [2, Chapter 3])
    segmentation_algorithm = msa.CBMEstimator(similarity_function="cosine", penalty_weight=0, penalty_func="modulo8", bands_number=7)

    f_mes_05, f_mes_3 = compute_segmentation(dataset, nmf, segmentation_algorithm)

    # Save the results of this benchmark
    save_path = f"{experiment.path.output}/{setting.identifier()}"
    np.save(f'{save_path}_f_mes_05.npy', np.array(f_mes_05))
    np.save(f'{save_path}_f_mes_3.npy', np.array(f_mes_3))


# Define the metrics which can be displayed using python <this_benchmark>.py -d
experiment.set_metric(
  name = 'f_mes_05',
  percent=True,
  higher_the_better= True,
)

# experiment.set_metric(
#   name = 'std_f_mes_05',
#   output='f_mes_05',
#   func = np.std,
# )

experiment.set_metric(
  name = 'f_mes_3',
  percent=True,
  higher_the_better= True,
  significance = True,
)

# experiment.set_metric(
#   name = 'std_f_mes_3',
#   output='f_mes_3',
#   func = np.std,
# )


# The routine to compute the segmentation. It is called in the step function.
def compute_segmentation(dataset, nmf, segmentation_algorithm):
    """
    Actually compute the segmentation of the dataset using the NMF and the segmentation algorithm.
    """
    # Empty lists for the scores
    all_scores_05 = []
    all_scores_3 = []

    # Iterate over all the songs in the dataset
    for idx_song in tqdm.tqdm(range(len(dataset))):
        # One song
        track_id, bars, barwise_tf_matrix, annotations_intervals = dataset[idx_song]

        # Compute NMF
        W, H = nmf.run(data=barwise_tf_matrix)

        # Predict the boundaries and convert them in seconds
        segments_estimated = segmentation_algorithm.predict_in_seconds(W, bars)
        
        # file_name = f"song-{track_id}_method{nmf_object.name}_segmentation_alg-{segmentation_algorithm.algorithm_name}.txt".replace(" ", "_")
        # dataset.save_segments(segments_estimated, file_name) # TODO

        # Compute the scores and store them
        tol_05, tol_3 = segmentation_algorithm.score(segments_estimated, annotations_intervals)
        # all_scores_05.append(tol_05)
        # all_scores_3.append(tol_3)
        all_scores_05.append(tol_05[2]) # Keep only the F measure
        all_scores_3.append(tol_3[2]) # Keep only the F measure
    
    return all_scores_05, all_scores_3

def compute_segmentation_with_cross_validation(dataset, nmf, segmentation_algorithm, cross_val_dict):
    # Cross validation is implemented in the transcription benchmark. TODO: implement it here.
    raise NotImplementedError("TODO. Implement the cross validation here.")

if __name__ == "__main__":
    # Invoke the command line management of the doce package
    doce.cli.main(experiment = experiment,
                  func = step)
