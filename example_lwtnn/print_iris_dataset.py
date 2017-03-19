#!/usr/bin/env python


from sklearn.datasets import load_iris


# Load iris dataset
dataset = load_iris()

# Print values to stdout
for x, y in zip(dataset['data'], dataset['target']):
    print('Inputs: {0:.2f} {1:.2f} {2:.2f} {3:.2f} -> Target: {4}'.format(
        x[0], x[1], x[2], x[3], dataset.target_names[y]))
