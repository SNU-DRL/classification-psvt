import argparse
import math
import os
from glob import glob

import numpy as np
import pandas as pd


def get_95_CI(metrics_list, n=10):
    mean = np.nanmean(metrics_list)
    std = np.nanstd(metrics_list, ddof=1)

    lower_bound = mean - 1.96 * std / math.sqrt(n)
    upper_bound = mean + 1.96 * std / math.sqrt(n)

    return mean, std, lower_bound, upper_bound


def parse_args():
    parser = argparse.ArgumentParser("Aggregate results of all folds")
    parser.add_argument(
        "--result_path",
        type=str,
        default="./cv_results",
        help="path for resulting cross validation folds",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    fold_results = glob(f"{args.result_path}/val_fold_*")

    results_df = pd.read_csv(os.path.join(fold_results[0], "scores.csv")).iloc[-1:]
    results_df = results_df.reset_index(drop=True)
    for fold_result in fold_results[1:]:
        result_df = pd.read_csv(os.path.join(fold_result, "scores.csv"))
        results_df.loc[len(results_df)] = result_df.iloc[-1]

    results_df = results_df.drop("epoch", axis=1)
    results_df.to_csv(os.path.join(args.result_path, "scores.csv"), index=False)

    processed_df = pd.concat(
        [results_df.mean().to_frame(), results_df.std().to_frame()], axis=1
    )
    processed_df.columns = ["mean", "stddev"]
    processed_df = processed_df.round(4)

    final_df = (
        processed_df.drop(
            ["train_loss", "val_loss", "auprc_baseline", "tn", "fp", "fn", "tp"]
        )
        .stack()
        .to_frame()
        .T
    )
    final_df.columns = ["_".join(col) for col in final_df.columns.values]

    final_df.to_csv(os.path.join(args.result_path, "all_scores.csv"), index=False)

    print(
        "AUROC: {:.3f}±{:.3f} ({:.3f}-{:.3f})".format(
            *get_95_CI(results_df.auroc.values, 10)
        )
    )
    print(
        "AUPRC: {:.3f}±{:.3f} ({:.3f}-{:.3f})".format(
            *get_95_CI(results_df.auprc.values, 10)
        )
    )
    print(
        "F1-score: {:.3f}±{:.3f} ({:.3f}-{:.3f})".format(
            *get_95_CI(results_df.f1_score.values, 10)
        )
    )
    print(
        "Sensitivity: {:.3f}±{:.3f} ({:.3f}-{:.3f})".format(
            *get_95_CI(results_df.sensitivity.values, 10)
        )
    )
    print(
        "Specificity: {:.3f}±{:.3f} ({:.3f}-{:.3f})".format(
            *get_95_CI(results_df.specificity.values, 10)
        )
    )
    print(
        "PPV: {:.3f}±{:.3f} ({:.3f}-{:.3f})".format(
            *get_95_CI(results_df.precision.values, 10)
        )
    )
    print(
        "NPV: {:.3f}±{:.3f} ({:.3f}-{:.3f})".format(
            *get_95_CI(results_df.npv.values, 10)
        )
    )
    print(
        "Accuracy: {:.3f}±{:.3f} ({:.3f}-{:.3f})".format(
            *get_95_CI(results_df.accuracy.values, 10)
        )
    )
