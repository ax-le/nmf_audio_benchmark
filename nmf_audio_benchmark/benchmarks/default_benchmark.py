"""
Template Benchmark, which can be followed to launch a benchmark.

The benchmark uses the hydra package for the management of the experiments.
See the documentation for more details about the hydra package: https://hydra.cc/docs/intro/

In a nutshell, the benchmark is defined by the following steps:
- Instantiate the dataset object, which is a dataloader for this particular dataset.
- Instantiate the NMF object, which is the algorithm that should be benchmarked.
- Instantiate the task object, which is the task on which is computed the benchmark.
- Compute the metrics for the task on the dataset, using a method defined in the task object.

All the details are not defined in Python code, but in a configuration file, which are YAML files. This follows the hydra standards.
See the configuration file in the current directory, named "template_config.yaml".
You may also check actual configuration files in the "music" directory, which are used to benchmark the MSA, MSS, and Music Transcription tasks.
"""

import numpy as np
import tqdm

# Load the dataloaders
import nmf_audio_benchmark.dataloaders
# Load the algorithms
import nmf_audio_benchmark.algorithms
# Load the tasks
import nmf_audio_benchmark.tasks

import hydra
from hydra.utils import instantiate
import logging

# A logger for this file
log = logging.getLogger(__name__)

# Inform the hydra package about the configuration file
# The configuration file is located in the current directory, and is named "template_config.yaml"
@hydra.main(version_base=None, config_path="./music", config_name="mss_config")
def launch_benchmark(cfg, save_metrics=True):
    log.info(f"Running benchmark with config: {cfg}\n")

    # Instanciate the dataset object
    log.info(f"Using dataset: {cfg.dataloader.object._target_} with parameters: {cfg.dataloader.object}\n")
    dataloader_object = instantiate(cfg.dataloader.object)

    # Instanciate the NMF object
    log.info(f"Using NMF model: {cfg.algorithm.object._target_} with parameters: {cfg.algorithm.object}\n")
    nmf_object = instantiate(cfg.algorithm.object)
    
    # Instanciate the task algorithm object
    log.info(f"Benchmarking on task: {cfg.task.object._target_} with parameters: {cfg.task.object}\n")
    task_object = instantiate(cfg.task.object)

    # Compute the metrics for the task
    metrics = task_object.compute_task_on_dataset(dataloader_object, nmf_object) # I could add *args and **kwargs to the function to pass additional parameters if needed

    # Print the metrics
    if metrics is None or metrics == {}:
        log.debug("No metrics computed, check the task implementation.")
    else:
        log.info(f"Metrics computed (in benchmark.py): {metrics}")

    if save_metrics:
        # Get the output directory from the hydra configuration
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        job_name = hydra.core.hydra_config.HydraConfig.get().job.name

        metrics["config"] = cfg

        # Save the metrics to a file, if needed
        # For example, use numpy to save to a .npy file
        np.save(f"{output_dir}/{job_name}_metrics.npy", metrics)

        # Or you can use pandas to save the metrics to a .csv file
        # import pandas as pd
        # df = pd.DataFrame(metrics)
        # df.to_csv(f"{output_dir}/{job_name}_metrics.csv", index=False)

    # You may also want to plot the metrics, for example using matplotlib
    # import matplotlib.pyplot as plt
    # plt.plot(metrics)
    # plt.savefig("metrics.png")

    # Or you can use seaborn to plot the metrics
    # import seaborn as sns
    # sns.lineplot(data=metrics)
    # plt.savefig("metrics_seaborn.png")

    # Or you can use plotly to plot the metrics
    # import plotly.express as px
    # fig = px.line(metrics)
    # fig.write_html("metrics_plotly.html")

    # Anyway, you can use the metrics as you want, for example to compare the performance of different algorithms or to visualize the results.

    return metrics

if __name__ == "__main__":
    # Launch the benchmark with the hydra package
    metrics = launch_benchmark()

    print("Benchmark completed.")
    # print(f"Metrics computed: {metrics}") ## For some reason, this does not work, while metrics is instanciated in launch_benchmark(), I don't know why.