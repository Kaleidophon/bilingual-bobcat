"""
Analyze results (plot losses, do a qualitative analyzes etc.).
"""

# STD
import codecs

# EXT
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import numpy as np


def compare_translations(reference_path, **model_output_paths):
    model_output_files = [codecs.open(path, "rb", "utf-8") for path in model_output_paths.values()]
    model_names = list(model_output_paths.keys())

    with codecs.open(reference_path, "rb", "utf8") as reference_file:
        reference_lines = reference_file.readlines()
        model_output_lines = [file.readlines() for file in model_output_files]
        model_output_lines = list(zip(*model_output_lines))

        for reference_sentence, model_output_sentences in zip(reference_lines, model_output_lines):
            print("Reference: " + reference_sentence.strip())

            for i, model_sentence in enumerate(model_output_sentences):
                print("{}: {}".format(model_names[i], model_sentence.strip()))

            print("")

    for file in model_output_files:
        file.close()


def plot_losses(save_path, **model_data_paths):
    # Get data
    model_data = {model_name: load_model_data(path) for model_name, path in model_data_paths.items()}
    model_training_losses = {model_name: model_data.item().get("train_loss") for model_name, model_data in model_data.items()}
    model_val_losses = {model_name: model_data.item().get("val_loss") for model_name, model_data in model_data.items()}

    # Create plots
    plot_metrics(model_training_losses, save_path, "training loss")
    plot_metrics(model_val_losses, save_path, "validation loss")


def plot_metrics(model_data, save_path, metric):
    data_length = len(list(model_data.values())[0])
    plt.figure()

    for current_model, current_model_data in model_data.items():
        x_axis = np.array(range(1, data_length+1))
        smoothed_axis = np.linspace(x_axis.min(), x_axis.max(), 50)

        smoothed_data = spline(x_axis, current_model_data, smoothed_axis)
        plt.plot(smoothed_axis, smoothed_data, label=current_model, linestyle="solid")

    plt.xlabel("Iteration")
    plt.xticks(range(1, data_length + 1))
    plt.ylabel(metric)
    plt.legend(loc=1)
    plt.savefig("{}{}.png".format(save_path, metric.replace(" ", "_")))


def load_model_data(data_path):
    return np.load(data_path)


if __name__ == "__main__":
    #compare_translations("./results/ref_debug.en", test_model="./results/eval_out_debug.txt")

    plot_losses("./results/", test_model="./results/debug.npy")
