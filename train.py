#!/usr/bin/env python
import argparse
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import pickle


def train(input, output, delimiter, target_column):

    data = np.genfromtxt(
            input,
            dtype=int,
            delimiter=delimiter
            )

    targets = data[:, 0]
    features = np.delete(data, target_column, axis=1)
    classifier = ExtraTreesClassifier(
        n_estimators=50,
        max_depth=None,
        min_samples_split=1.0,
        random_state=0,
        verbose=True,
        n_jobs=-1)
    # We learn the digits on the first half of the digits
    classifier.fit(features, targets)
    with open(output, 'wb') as fp:
        pickle.dump(classifier, fp)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simple train algorithm')
    parser.add_argument('-i', '--input', help='Input file', required=True)
    parser.add_argument('-o', '--output', help='Output model', required=True)
    parser.add_argument('-d', '--delimiter', help='Data separator', default=',')
    parser.add_argument('-t', '--target-column', help='Target column', default=0, type=int)
    args = parser.parse_args()

    exit(train(**vars(args)))
