# -*- encoding: utf-8 -*-

import sys
import os
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from data_management.data_manager import DataManager
import argparse
import tempfile
import time

from tpot import TPOTClassifier


def main(dataset_name, dataset_id):
    dm = DataManager()
    dm.read_data(dataset_name, test_split=0.2)

    start_time = time.time()
    generation = 0

    automl = TPOTClassifier(
        generations=1,
        scoring='balanced_accuracy',
        max_eval_time_mins=100, # how long should this seed fit process run
        warm_start=True,
        cv=5,
        verbosity=0
    )
    with open(os.path.join(tempfile.gettempdir(), "result.csv"), "w") as f:
        while time.time() - start_time < 92400:
            print(time.time() - start_time, generation)

            automl.fit(dm.X_train, dm.Y_train)
            score = automl.score(dm.X_test, dm.Y_test)

            generation += 1

            print(time.time() - start_time, generation, score, sep=",", file=f)


if __name__ == '__main__':
    dataset_id = int(sys.argv[-1]) - 1

    with open("openml_datasets.txt", "r") as f:
        dataset_name = list(f)[dataset_id].strip()
    main(dataset_name, dataset_id)
