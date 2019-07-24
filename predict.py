#!/usr/bin/env python
import argparse
import numpy as np
import pickle


def predict(input, output, model, delimiter, target_column):

    data = np.genfromtxt(
            input,
            dtype=int,
            delimiter=delimiter
            )

    targets = data[:, [target_column]]
    features = np.delete(data, target_column, axis=1)
    with open(model, 'rb') as fp:
        classifier = pickle.load(fp)

    predictions = classifier.predict(features).reshape(targets.shape)
    np.savetxt(output, np.concatenate([predictions, targets, features], axis=1), delimiter=delimiter, fmt='%d')

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simple train algorithm')
    parser.add_argument('-i', '--input', help='Input file', required=True)
    parser.add_argument('-o', '--output', help='Output file', required=True)
    parser.add_argument('-m', '--model', help='Model', required=True)
    parser.add_argument('-d', '--delimiter', help='Data separator', default=',')
    parser.add_argument('-t', '--target-column', help='Target column', default=0, type=int)
    args = parser.parse_args()

    exit(predict(**vars(args)))
