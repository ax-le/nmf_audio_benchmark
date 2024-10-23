"""
A base task, to act as the default task.

It uses 3 functions:
- __init__ : Constructor of the TemplateTask class.
- predict : Perform Predictions, i.e. what will be used to transform the W, H factors of NMF into predictions for the task.
- score : Compute the score of the predictions.

This is largely inspired from the scikit-learn API: https://scikit-learn.org/stable/auto_examples/developing_estimators/sklearn_is_fitted.html, Author: Kushan <kushansharma1@gmail.com>, License: BSD 3 clause
"""

class BaseTask():
    def __init__(self):
        """
        Constructor of the BaseTask class.
        """
        raise NotImplementedError("This method should be implemented.") from None

    def predict(self, W, H):
        """
        Perform predictions, based on the W and H matrices found by NMF.
        """
        raise NotImplementedError("This method should be implemented.") from None

    def score(self, predictions, annotations):
        """
        Compute the score of the predictions.
        """
        raise NotImplementedError("This method should be implemented.") from None