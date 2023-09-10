import argparse
import os
import pickle
import random

import numpy as np

from src.ECGXMLReader import ECGXMLReader


def parse_args():
    parser = argparse.ArgumentParser("Building cross-validation folds")
    parser.add_argument(
        "--c_avrt_path",
        type=str,
        required=True,
        help="path for concealed AVRT samples (label 0)",
    )
    parser.add_argument(
        "--avnrt_path",
        type=str,
        required=True,
        help="path for AVNRT samples (label 1)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./cv_folds",
        help="path for resulting cross validation folds",
    )
    parser.add_argument("--num_folds", type=int, default=10, help="number of folds")
    parser.add_argument("--random_seed", type=int, default=0, help="random_seed")

    args = parser.parse_args()

    print(args)
    return args


def setup_seed(args):
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)


def read_ecg_xml(path):
    xml_files = os.listdir(path)
    total_data = []

    for xml_file in xml_files:
        total_data.append(ECGXMLReader(os.path.join(path, xml_file)))

    return total_data


def build_cv_folds(path_0, path_1, save_path, num_folds=10):
    """
    build cross validation folds with specified num_folds
    save each fold's data list (list of ECGXMLReader)
    """

    # Read data from path_0 (Concealed AVRT)
    if path_0 is not None:
        _total_data_0 = read_ecg_xml(path_0)
        total_data_0 = list(filter(lambda x: x.num_samples == 5000, _total_data_0))
        np.random.shuffle(total_data_0)
    else:
        total_data_0 = []
    print(f"Total # of Concealed AVRT: {len(total_data_0)}")

    # Read data from path_1 (AVNRT)
    if path_1 is not None:
        _total_data_1 = read_ecg_xml(path_1)
        total_data_1 = list(filter(lambda x: x.num_samples == 5000, _total_data_1))
        np.random.shuffle(total_data_1)
    else:
        total_data_1 = []
    print(f"Total # of AVNRT: {len(total_data_1)}")

    data_0_folds = np.array_split(total_data_0, num_folds)
    data_1_folds = np.array_split(total_data_1, num_folds)

    for fold_idx in range(num_folds):
        each_fold_0 = data_0_folds[fold_idx].tolist()
        each_fold_1 = data_1_folds[fold_idx].tolist()

        print(f"# of Concealed AVRT in fold {fold_idx}: {len(each_fold_0)}")
        print(f"# of AVNRT in fold {fold_idx}: {len(each_fold_1)}")
        print(f"# of samples in fold {fold_idx}: {len(each_fold_0) + len(each_fold_1)}")
        np.random.shuffle(each_fold_0)
        np.random.shuffle(each_fold_1)
        with open(os.path.join(save_path, f"fold_{fold_idx}_C-AVRT.pkl"), "wb") as f:
            pickle.dump(each_fold_0, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(save_path, f"fold_{fold_idx}_AVNRT.pkl"), "wb") as f:
            pickle.dump(each_fold_1, f, pickle.HIGHEST_PROTOCOL)


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    setup_seed(args)

    build_cv_folds(
        path_0=args.c_avrt_path,
        path_1=args.avnrt_path,
        save_path=args.save_path,
        num_folds=args.num_folds,
    )


if __name__ == "__main__":
    main()
