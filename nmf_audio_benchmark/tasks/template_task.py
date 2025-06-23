"""
A template task, that you ay use as a basis for adding a new task to the nmf_audio_benchmark package.

Each task is based on 4 functions, that need to be implemented in any task to work seamlessly with the rest of the package:
- __init__ : Constructor of the class.
- predict : Perform predictions, i.e. what will be used to transform the W, H factors of NMF into predictions for the task.
- score : Compute the score of the predictions.
- compute_task_on_dataset : Iterator to compute the task on the dataset, and define how to handle the particular metrics associated with this task.
"""

import tqdm

from nmf_audio_benchmark.tasks.base_task import *

class TemplateTask(BaseTask):
    def __init__(self):
        """
        Constructor of the TemplateTask class.
        """
        pass

    def predict(self, W, H):
        """
        Perform Predictions.
        """
        raise NotImplementedError("This method should be implemented.") from None

    def score(self, predictions, annotations):
        """
        Compute the score of the predictions.
        """
        raise NotImplementedError("This method should be implemented.") from None

    def compute_task_on_dataset(self, dataset, nmf, verbose=False):
        """
        Compute the task on the dataset, using the NMF and the task algorithm.
        """
        return compute_task_on_dataset(dataset, nmf, self, verbose=verbose)

def compute_task_on_dataset(dataset, nmf, task_algorithm, verbose=False):
    """
    Actually compute the task on the dataset, using the NMF and the task algorithm.
    """
    # Empty lists for the scores
    all_metrics = {}

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

        if verbose:
            # Log the computed metrics
            print(f"Computed metrics for track {track_id}: {metrics}")

        # Store the metrics
        all_metrics[track_id] = metrics

    if verbose:
        # Log the final metrics
        print(f"All metrics computed. Songs processed: {len(all_metrics)}")
        print(f"Final metrics: {all_metrics}")


    return all_metrics