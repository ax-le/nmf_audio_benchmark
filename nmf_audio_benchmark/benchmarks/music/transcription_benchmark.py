"""
Benchmark definition for the Music Transcription task.
Details for the transcription task with NMF can be found in [1]. The transcription algorithm is derived from [2].   

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
    - Either using the CLI: python transcription_benchmark.py -d
    - Or analyzing the results in a Python script. See "utils.visualize_outputs.py" for an example.

References:
[1] Smaragdis, P., and Judith C. B.. "Non-negative matrix factorization for polyphonic music transcription." 2003 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (IEEE Cat. No. 03TH8684). IEEE, 2003.
[2] Marmoret, A., Bertin, N., & Cohen, J. (2019). Multi-Channel Automatic Music Transcription Using Tensor Algebra. arXiv preprint arXiv:2107.11250.
"""
# Define the dataloaders
import nmf_audio_benchmark.dataloaders.music.transcription_dataloader as tr_dl
# Define the task
import nmf_audio_benchmark.tasks.music.transcription as tr
# Define the algorithm
import nmf_audio_benchmark.algorithms.nn_fac_algos as nn_fac_algos

# Code for the cross validation
import nmf_audio_benchmark.utils.find_hyperparameters as hyperparams_helper
from sklearn.base import clone

import tqdm
import numpy as np
import doce


# Instantiante the experiment.
experiment = doce.Experiment(name = 'transcription_benchmark',
                             purpose = 'Benchmarking NMF methods on the Music Transcription task',
                             author = 'Axel Marmoret',
                             address = 'axel.marmoret@imt-atlantique.fr',
                             )

# Path to the dataset
datapath = '/home/a23marmo/datasets/MAPS'

# Dataset class. The object will be initialized in the step function.
dataset_object = tr_dl.MAPSDataloader

# Store the outputs, logs and other files in a specific directory
experiment.set_path('output', f'./experiments_outputs/{experiment.name}/{dataset_object.name}/')

# Fixed parameters for the experiment. Will not appear in the name of the output files
experiment.init = "nndsvd"

experiment.time_tolerance = 0.05

# Feature parameters
experiment.feature = "stft"
experiment.n_fft = 3528 # Weird, but values from [1]
experiment.hop_length = 882

# Set the parameters which are going to be varied in the experiment
experiment.add_plan('harmonic',
    # dataset params
    subfolder = ["ENSTDkCl", "AkPnBcht"],

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
    subfolder = ["ENSTDkAm", "AkPnBcht"],

    # NMF params
    rank = [20, 40],
    beta = [2,1],#,0],
    nmf_type = "unconstrained",
    init = ["nndsvd"],

    # Task params
    # no params, due to the cross validation
)

experiment.task_default_values = {
    'threshold':0.01,
    'salience_shift_autocorrelation':0.3,
    'adaptative_threshold':True,
    'H_normalization':True,
}

# Cross validation parameters
# Cross validation is only possible for the task itself for now. TODO: extend it for the NMF parameters?
experiment.cross_validation = True
experiment.cv = 4

experiment.task_param_grid = {
    'threshold':np.arange(0.001, 0.02, 0.001),
    'salience_shift_autocorrelation':np.arange(0.1, 0.7, 0.1),
    'adaptative_threshold':[True, False]
}

# The core function of the experiment
def step(setting, experiment):
    # Instanciate the dataset object
    dataset = dataset_object(datapath=datapath, feature = experiment.feature, subfolder = setting.subfolder, n_fft = experiment.n_fft, hop_length = experiment.hop_length)

    # Instanciate the NMF object
    nmf = nn_fac_algos.nn_fac_NMF(setting.rank, setting.beta, init = setting.init, nmf_type=setting.nmf_type,
                                  normalize=[True, False])
    
    # Save the results of this benchmark
    save_path = f"{experiment.path.output}/{setting.identifier()}"
    
    if experiment.cross_validation: # Compute the transcription with cross validation for the hyperparameters of the transcription algorithm
        # Initialize the default transcription algorithm, in order to set parameter which are not going to be varied in the cross validation
        default_transcription_algorithm = tr.Transcription(feature_object=dataset.feature_object, salience_shift_autocorrelation = experiment.task_default_values['salience_shift_autocorrelation'],threshold = experiment.task_default_values['threshold'],
                                                           H_normalization = experiment.task_default_values['H_normalization'], adaptative_threshold = experiment.task_default_values['adaptative_threshold'], verbose=False)
        # Run the transcription with cross validation
        final_results, best_params_cv = compute_transcription_with_cross_validation(dataset, nmf, default_transcription_algorithm, experiment.task_param_grid, cv=experiment.cv)

        # Save the best parameters
        np.save(f'{save_path}_best_params_cv.npy', np.array(best_params_cv))

        # Save the results of this benchmark
        all_accuracies = final_results[:, 1]
        all_f1 = final_results[:, 0]

    else:
        # Initialize the transcription algorithm
        transcription_algorithm = tr.Transcription(feature_object=dataset.feature_object, salience_shift_autocorrelation = experiment.task_default_values['salience_shift_autocorrelation'],threshold = experiment.task_default_values['threshold'],
                                                   H_normalization = experiment.task_default_values['H_normalization'], adaptative_threshold = experiment.task_default_values['adaptative_threshold'], verbose=False)
        
        # Compute the transcription with this algorithm
        all_accuracies, all_f1 = compute_transcription(dataset, nmf, transcription_algorithm)

    # Save the results of this benchmark
    np.save(f'{save_path}_accuracies.npy', np.array(all_accuracies))
    np.save(f'{save_path}_f1s.npy', np.array(all_f1))

# set the metrics

experiment.set_metric(
  name = 'accuracies',
  percent=True,
  higher_the_better= True,
  significance = True,
  precision = 10
)

experiment.set_metric(
  name = 'f1s',
  percent=True,
  higher_the_better= True,
  significance = True,
  precision = 10
)

# %% Scripts to compute the transcription
def compute_transcription(dataset, nmf, transcription_algorithm):
    """
    Compute the transcription for all the songs in the dataset, with this particular set of parameters.
    """
    # Empty lists for the scores
    all_accuracies = []
    all_f1 = []

    # Iterate over all the songs in the dataset
    for idx_song in tqdm.tqdm(range(len(dataset))):
        # One song
        track_id, spectrogram, annotations = dataset[idx_song]

        # Compute NMF
        W, H = nmf.run(data=spectrogram, feature_object=dataset.feature_object)

        # Compute the transcription
        estimated_transcription = transcription_algorithm.predict(W, H)

        # Compute the metrics
        f_mes, accuracy = transcription_algorithm.score(estimated_transcription, annotations, time_tolerance=experiment.time_tolerance)

        # Compute the metrics
        all_f1.append(f_mes)
        all_accuracies.append(accuracy)
    
    return all_accuracies, all_f1

def compute_transcription_with_cross_validation(dataset, nmf, default_transcription_algorithm, param_grid, cv=4):
    """
    Cross validation to find the best parameters for the transcription algorithm.
    A second option could be hyperparameter optimization, using hyperopt for example, but it may be considered as "cheating".

    It would have been easier to use the GridSearchCV from scikit-learn, but it is cumbersome to adapt the current code to it.
    In particular, it requires to have a single fit for the whole dataset, which is not the mentality of the current code.
    """

    # Empty lists for the scores
    all_results = []

    param_combinations = list(hyperparams_helper.generate_param_grid(param_grid))

    def evaluate_one_set_params(W, H, annotations, params):
        # Clone the default transcription algorithm and update the parameters
        transcription_algorithm = clone(default_transcription_algorithm)
        transcription_algorithm.update_params(params)

        # Compute the transcription with the new parameters
        estimated_transcription = transcription_algorithm.predict(W, H)
        f_mes, accuracy = transcription_algorithm.score(estimated_transcription, annotations, time_tolerance=experiment.time_tolerance)
        return (f_mes, accuracy)
    
    # Iterate over all the songs in the dataset
    for idx_song in tqdm.tqdm(range(len(dataset))):
        # One song
        track_id, spectrogram, annotations = dataset[idx_song]

        # Compute NMF
        W, H = nmf.run(data=spectrogram, feature_object=dataset.feature_object)

        # Compute the transcription for all params
        all_results.append([evaluate_one_set_params(W, H, annotations, params) for params in param_combinations])

    # Find the best results in the cross validation scheme
    final_results, final_best_params = hyperparams_helper.find_best_results_in_cv_scheme(all_results, cv, param_combinations, nb_metrics=2, test_metric_idx=1)

    # Return the final results, obtained via cross-validation
    return final_results, final_best_params

if __name__ == "__main__":
    # Invoke the command line management of the doce package
    doce.cli.main(experiment = experiment,
                  func = step)
    
    # experiment.perform(
    #   experiment.selector,
    #   step,
    #   nb_jobs=1,
    # #   log_file_name=log_file_name,
    # #   progress=args.progress,
    # #   mail_interval=float(args.mail)
    #   )