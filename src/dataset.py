import os
import pickle
import random
from collections import OrderedDict
from glob import glob

import numpy as np
import torch

LEADSETS = {
    12: ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    8: ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"],
    6: ["I", "II", "III", "aVR", "aVL", "aVF"],
    5: ["V1", "V2", "V3", "V4", "V5", "V6"],
    1: ["I"],
}


def build_datasets_from_cv_folds(cv_folds_path, valid_fold_num, num_leads=12):
    """
    build dataset from splited folds
    use fold_num as validation set, others as training set
    """
    if not os.path.exists(cv_folds_path):
        raise FileNotFoundError("Path for cross-validation folds is not valid")

    fold_files_0 = glob(os.path.join(cv_folds_path, "*_C-AVRT.pkl"))
    validation_fold_0 = os.path.join(cv_folds_path, f"fold_{valid_fold_num}_C-AVRT.pkl")
    training_folds_0 = [x for x in fold_files_0 if x != validation_fold_0]

    fold_files_1 = glob(os.path.join(cv_folds_path, "*_AVNRT.pkl"))
    validation_fold_1 = os.path.join(cv_folds_path, f"fold_{valid_fold_num}_AVNRT.pkl")
    training_folds_1 = [x for x in fold_files_1 if x != validation_fold_1]

    # Training set
    train_data_0 = []
    for training_fold in training_folds_0:
        with open(training_fold, "rb") as f:
            train_data_0.extend(pickle.load(f))

    train_data_1 = []
    for training_fold in training_folds_1:
        with open(training_fold, "rb") as f:
            train_data_1.extend(pickle.load(f))

    random.shuffle(train_data_0)
    random.shuffle(train_data_1)

    print(f"# of Concealed AVRT in training set: {len(train_data_0)}")
    print(f"# of AVNRT in training set: {len(train_data_1)}")
    train_dataset = Dataset_PSVT(train_data_0, train_data_1, num_leads=num_leads)

    # Validation set
    valid_data_0 = []
    with open(validation_fold_0, "rb") as f:
        valid_data_0.extend(pickle.load(f))

    valid_data_1 = []
    with open(validation_fold_1, "rb") as f:
        valid_data_1.extend(pickle.load(f))

    print(f"# of Concealed AVRT in validation set: {len(valid_data_0)}")
    print(f"# of AVNRT in validation set: {len(valid_data_1)}")
    valid_dataset = Dataset_PSVT(
        valid_data_0,
        valid_data_1,
        num_leads=num_leads,
        mean_dict=train_dataset.mean_dict,
    )

    return train_dataset, valid_dataset


def merge_leads_into_matrix(leads_dict, num_leads):
    selected_leads_dict = {}

    for lead in LEADSETS[num_leads]:
        selected_leads_dict[lead] = leads_dict[lead]

    standard_lead_order = [
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]
    ordered_leads_dict = OrderedDict(
        sorted(
            selected_leads_dict.items(), key=lambda x: standard_lead_order.index(x[0])
        )
    )

    recordings = []

    for lead, recording in ordered_leads_dict.items():
        recordings.append(recording)

    merged = np.stack(recordings, axis=0)
    return merged


def preprocess_recording(recording):
    recording = torch.tensor(recording, dtype=torch.float32)
    recording = recording - recording.mean(dim=1, keepdim=True)
    recording = recording / (recording.std(dim=1, keepdim=True) + 1e-6)
    return recording


def get_gender_vector(num):
    if num == 0:
        return np.array([1, 0])
    elif num == 1:
        return np.array([0, 1])
    else:
        return np.array([0, 0])


class Dataset_PSVT(torch.utils.data.Dataset):
    def __init__(
        self,
        ecg_list_0,
        ecg_list_1,
        transform=lambda x: x,
        num_leads=12,
        mean_dict=None,
    ):
        sample_list_0 = [
            {
                "x": ecg_reader.getAllVoltages(),
                "y": 0,
                "id": ecg_reader.id,
                "age": ecg_reader.Age,
                "gender": ecg_reader.Gender,
                "hr": ecg_reader.HeartRate,
                "features": ecg_reader.Features,
                "diagnosis_features": ecg_reader.DiagnosisFeatures,
            }
            for ecg_reader in ecg_list_0
        ]
        sample_list_1 = [
            {
                "x": ecg_reader.getAllVoltages(),
                "y": 1,
                "id": ecg_reader.id,
                "age": ecg_reader.Age,
                "gender": ecg_reader.Gender,
                "hr": ecg_reader.HeartRate,
                "features": ecg_reader.Features,
                "diagnosis_features": ecg_reader.DiagnosisFeatures,
            }
            for ecg_reader in ecg_list_1
        ]

        sample_list = sample_list_0 + sample_list_1
        np.random.shuffle(sample_list)

        self.mean_dict = {}
        self.recordings = list(
            map(
                lambda x: preprocess_recording(
                    merge_leads_into_matrix(x["x"], num_leads)
                ),
                sample_list,
            )
        )
        self.labels = list(map(lambda x: x["y"], sample_list))
        self.ids = list(map(lambda x: x["id"], sample_list))

        # Age
        ages = list(map(lambda x: x["age"], sample_list))
        if mean_dict is not None:
            mean_age = mean_dict["age"]
        else:
            mean_age = np.nanmean(ages)
        self.ages = list(
            map(
                lambda x: np.array([x / 100])
                if not np.isnan(x)
                else np.array([mean_age / 100]),
                ages,
            )
        )

        # Gender
        self.genders = list(map(lambda x: get_gender_vector(x["gender"]), sample_list))

        # Heart rate
        hrs = list(map(lambda x: x["hr"], sample_list))
        if mean_dict is not None:
            mean_hr = mean_dict["hr"]
        else:
            mean_hr = np.nanmean(hrs)
        self.hrs = list(
            map(
                lambda x: np.array([x / 100])
                if not np.isnan(x)
                else np.array([mean_hr / 100]),
                hrs,
            )
        )

        # Other features
        feature_vectors = np.array(list(map(lambda x: x["features"], sample_list)))
        if mean_dict is not None:
            mean_features = mean_dict["features"]
        else:
            mean_features = np.nanmean(feature_vectors, axis=0)
        for feature_idx in range(len(mean_features)):
            feature_vectors[:, feature_idx] = np.nan_to_num(
                feature_vectors[:, feature_idx], nan=mean_features[feature_idx]
            )
        self.features = feature_vectors / 100

        self.diagnosis_features = np.array(
            list(map(lambda x: x["diagnosis_features"], sample_list))
        )

        self.num_recordings = len(self.labels)
        self.num_leads = num_leads
        self.transform = transform

        self.mean_dict["age"] = mean_age
        self.mean_dict["hr"] = mean_hr
        self.mean_dict["features"] = mean_features

    def __len__(self):
        return self.num_recordings

    def __getitem__(self, idx):
        return {
            "id": self.ids[idx],
            "recording": self.transform(self.recordings[idx]).unsqueeze(0),
            "label": torch.tensor(self.labels[idx]),
            "feature": torch.tensor(
                np.concatenate(
                    [
                        self.ages[idx],
                        self.genders[idx],
                        self.hrs[idx],
                        self.features[idx],
                        self.diagnosis_features[idx],
                    ]
                ),
                dtype=torch.float32,
            ),
        }


def get_dataloader(dataset, batch_size, num_workers=1):
    return torch.utils.data.DataLoader(
        dataset,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=False,
    )
