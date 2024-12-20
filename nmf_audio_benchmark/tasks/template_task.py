"""
A template task, to be used as a basis for adding a new task to the nmf_audio_benchmark package.

It uses 3 functions:
- __init__ : Constructor of the TemplateTask class.
- predict : Perform Predictions, i.e. what will be used to transform the W, H factors of NMF into predictions for the task.
- score : Compute the score of the predictions.
"""

class TemplateTask():
    def __init__(self):
        """
        Constructor of the TemplateTask class.
        """
        TODO

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