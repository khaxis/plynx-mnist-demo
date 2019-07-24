#!/usr/bin/env python
import argparse
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_examples(data):
    mismatches = defaultdict(list)
    matches = defaultdict(list)

    for i in range(data.shape[0]):
        predicted = data[i, 0]
        target = data[i, 1]
        if target == predicted:
            matches[predicted].append(i)
        else:
            mismatches[predicted].append(i)

    return matches, mismatches


def plot_details(data, selected_images, N=15):
    f, axarr = plt.subplots(N, 10, figsize=(10, 20))

    for target in range(10):
        j = 0
        for j, index in enumerate(selected_images[target]):
            d = data[index, 2:]
            d = np.reshape(d, (28, 28))
            axarr[j, target].imshow(d)
            if j == 0:
                axarr[j, target].set_title(target)
            axarr[j, target].axis('off')
            if j == N - 1:
                break
        while j < N:
            axarr[j, target].axis('off')
            j += 1


def build_plot(input, delimiter, out_cnf, out_matches, out_mismatches):

    data = np.genfromtxt(
            input,
            dtype=int,
            delimiter=delimiter
            )

    predictions = data[:, [0]]
    targets = data[:, [1]]
    labels = np.unique([targets, predictions])
    cnf_matrix = confusion_matrix(targets, predictions, labels=labels)
    print(cnf_matrix)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True,
                          title='Normalized confusion matrix')

    plt.savefig(out_cnf)

    matches, mismatches = get_examples(data)

    plot_details(data, matches)
    plt.savefig(out_matches)

    plot_details(data, mismatches)
    plt.savefig(out_mismatches)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simple train algorithm')
    parser.add_argument('-i', '--input', help='Input file', required=True)
    parser.add_argument('-c', '--out-cnf', help='Plot confusion matrix', required=True)
    parser.add_argument('-s', '--out-mismatches', help='Plot mistakes', required=True)
    parser.add_argument('-m', '--out-matches', help='Plot correctly predicted images', required=True)
    parser.add_argument('-d', '--delimiter', help='Data separator', default=',')
    args = parser.parse_args()

    exit(build_plot(**vars(args)))
