# -*- coding: utf-8 -*-
"""
@author: a23marmo

Defining common plotting functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from base_audio.common_plot import *

def plot_measure_with_annotations(measure, annotations, color = "red"):
    """
    Plots the measure (typically activations for H) with the segmentation annotation.
    """
    plt.plot(np.arange(len(measure)),measure, color = "black")
    for x in annotations:
        plt.plot([x, x], [0,np.amax(measure)], '-', linewidth=1, color = color)
    plt.show()
    
def plot_measure_with_annotations_and_prediction(measure, annotations, frontiers_predictions, title = "Title"):
    """
    Plots the measure (typically activations for H) with the segmentation annotation and the estimated segmentation.
    """
    plt.title(title)
    plt.plot(np.arange(len(measure)),measure, color = "black")
    ax1 = plt.axes()
    ax1.axes.get_yaxis().set_visible(False)
    for x in annotations:
        plt.plot([x, x], [0,np.amax(measure)], '-', linewidth=1, color = "red")
    for x in frontiers_predictions:
        if x in annotations:
            plt.plot([x, x], [0,np.amax(measure)], '-', linewidth=1, color = "#8080FF")#"#17becf")
        else:
            plt.plot([x, x], [0,np.amax(measure)], '-', linewidth=1, color = "orange")
    plt.show()