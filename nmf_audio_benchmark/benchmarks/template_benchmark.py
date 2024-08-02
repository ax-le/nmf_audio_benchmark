"""
Template Benchmark, which can be followed to define a new benchmark.

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
    - Either using the CLI: python <this_benchmark>.py -d
    - Or analyzing the results in a Python script. See "utils.visualize_outputs.py" for an example.
"""

# Define the dataloaders
import nmf_audio_benchmark.dataloaders.template_dataloader as dl
# Define the algorithm
import nmf_audio_benchmark.algorithms.template_nmf as nmf
# Define the task
import nmf_audio_benchmark.tasks.template_task as task

import tqdm
import numpy as np
import doce

# Path to the dataset
datapath = '/path/for/the/dataset' # TODO: to change

# Dataset class. The object will be initialized in the step function.
dataset_object = dl.TemplateDataloader

# Instantiante the experiment.
experiment = doce.Experiment(name = 'template_benchmark',
                             purpose = 'A template to add a new benchmark',
                             author = 'Axel Marmoret',
                             address = 'axel.marmoret@imt-atlantique.fr',
                             )

# Store the outputs, logs and other files in a specific directory
experiment.set_path('output', f'./experiments_outputs/{experiment.name}/{dataset_object.name}/')

# Fixed parameters for the experiment. Will not appear in the name of the output files
experiment.fixed_parameter = "fixed"
experiment.feature = "stft"
experiment.init = "random"

# Set the parameters which are going to be varied in the experiment
experiment.add_plan('a_plan',
    varying_paramter = ["value_1", "value_2"],
    rank=[1, 2, 3],
    beta=[2, 1]
)

def step(setting, experiment):
    # Instanciate the dataset object
    dataset = dataset_object(datapath, feature = experiment.feature) # Fixed parameters are accessed via the experiment object

    # Instanciate the NMF object
    nmf = nmf.TemplateNMF(setting.rank, setting.beta, # Varying parameters are accessed via the setting object
                          init = experiment.init, nmf_type="default")
    
    # Instanciate the task algorithm object
    task_algorithm = task.TemplateTask()

    metrics = compute_task(dataset, nmf, task_algorithm)

    # Save the results of this benchmark
    save_path = f"{experiment.path.output}/{setting.identifier()}"
    np.save(f'{save_path}_metric.npy', np.array(metrics))

# Define the metrics which can be displayed using python <this_benchmark>.py -d
experiment.set_metric(
  name = 'metric',
  percent=True,
  higher_the_better= True,
)

experiment.set_metric(
  name = 'std_metric',
  output='metric',
  func = np.std,
)


def compute_task(dataset, nmf, task_algorithm):
    """
    Actually compute the task on the dataset, using the NMF and the task algorithm.
    """
    # Empty lists for the scores
    all_metrics = []

    # Iterate over all the songs in the dataset
    for idx_song in tqdm.tqdm(range(len(dataset))):
        # One song
        track_id, spectrogram, annotations = dataset[idx_song]

        # Compute NMF
        W, H = nmf.run(data=spectrogram)

        # Predict the estimmations for this task
        predictions = task_algorithm.predict(W, H)

        # Compute the scores
        metrics = task_algorithm.score(predictions, annotations)
        all_metrics.append(metrics)
    
    return all_metrics

def compute_segmentation_with_cross_validation(dataset, nmf, task_algorithm, cross_val_dict):
    # Cross validation is implemented in the transcription benchmark. TODO: implement it here.
    raise NotImplementedError("TODO. Implement the cross validation here.")

if __name__ == "__main__":
    # Invoke the command line management of the doce package
    doce.cli.main(experiment = experiment,
                  func = step)
