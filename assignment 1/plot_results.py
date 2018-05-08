"""
Quick and dirty script to plot the results of trainings jointly in one graph.
"""

# STD
from collections import defaultdict
import codecs
from random import sample

# EXT
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import numpy as np

# All relevant results that are going to be plotted
RESULT_FILES = [
    "./models/simple_model1_results.txt",
    "./models/vb_alpha_0.01_results.txt",
    "./models/vb_alpha_0.1_results.txt",
    "./models/vb_alpha_1_results.txt",
    "./models/uniform_model2_results.txt",
    "./models/continue_model2_results.txt",
    "./models/random_model2_run1_results.txt"
]


def aggregate_results(*result_paths):
    """
    Aggregate training data from different models regarding the same metric.
    """
    metric_data = defaultdict(dict)

    def _parse_line(line):
        metric, raw_data = line.split("\t")
        raw_data = raw_data.split(" ")
        data = list(map(float, raw_data))
        return metric, data

    def _get_model_name(result_path):
        return result_path.replace("./models/", "").replace("_results.txt", "")

    for result_path in result_paths:
        model_name = _get_model_name(result_path)

        with codecs.open(result_path, "rb", "utf-8") as result_file:
            for line in result_file.readlines():
                metric, data = _parse_line(line)
                metric_data[metric][model_name] = data

    return metric_data


def plot_metrics(metric_data, save_path):
    linestyles = ("solid", "dashed", "dashdot", "dotted")

    for metric, model_data in metric_data.items():
        plt.figure()
        data_length = len(list(model_data.values())[0])

        for current_model, current_model_data in model_data.items():
            x_axis = np.array(range(1, data_length+1))
            smoothed_axis = np.linspace(x_axis.min(), x_axis.max(), 50)

            smoothed_data = spline(x_axis, current_model_data, smoothed_axis)
            plt.plot(smoothed_axis, smoothed_data, label=current_model, linestyle="solid")

        plt.xlabel("Iteration")
        plt.xticks(range(1, data_length + 1))
        plt.ylabel(metric)
        plt.legend(loc=1 if metric == "AER" else 4)
        plt.savefig("{}{}.png".format(save_path, metric.lower()))


if __name__ == "__main__":
    metric_data = aggregate_results(*RESULT_FILES)
    plot_metrics(metric_data, "./models/")
