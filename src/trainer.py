# Reference: https://xylambda.github.io/blog/python/pytorch/machine-learning/2021/01/04/pytorch_trainer.html

import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics
from tqdm import tqdm


class Trainer:
    """
    Parameters
    ----------
    model : torch.nn.Module
    criterion : torch.nn.modules.loss
    optimizer : torch.optim
    scheduler : torch.optim.lr_scheduler
    log_kwargs : dict
        - Dictionary for paths
    """

    def __init__(self, model, criterion, optimizer, scheduler, log_kwargs, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_kwargs = log_kwargs
        self.device = device

        self.model.to(self.device)
        self.valid_metrics_df = pd.DataFrame(
            columns=[
                "epoch",
                "train_loss",
                "val_loss",
                "auroc",
                "auprc",
                "auprc_baseline",
                "accuracy",
                "sensitivity",
                "specificity",
                "precision",
                "npv",
                "f1_score",
                "threshold",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
        )
        self.test_metrics_df = pd.DataFrame(
            columns=[
                "auroc",
                "auprc",
                "auprc_baseline",
                "accuracy",
                "sensitivity",
                "specificity",
                "precision",
                "npv",
                "f1_score",
                "threshold",
                "tn",
                "fp",
                "fn",
                "tp",
                "test_loss",
            ]
        )

    def fit(self, train_loader, val_loader, epochs):
        """
            Training & Validation
        """
        # track total training time
        total_start_time = time.time()

        # ---- train process ----
        for epoch in range(1, epochs + 1):
            # train
            train_loss = self._train(train_loader)
            print(
                f"- [{epoch:03d}/{epochs:03d}] Train loss: {train_loss:.4f}, learning rate: {self.scheduler.get_last_lr()[0]:.6f}"
            )

            # validate
            val_loss, val_metrics_dict, val_result = self._validate(
                val_loader, is_test=False
            )
            print(f"- [{epoch:03d}/{epochs:03d}] Validation loss: {val_loss:.4f}")
            epoch_metrics_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            update_metrics_dict = {**epoch_metrics_dict, **val_metrics_dict}
            self._log(
                update_metrics_dict, self.log_kwargs["scorefile_path"], is_test=False
            )
            self.scheduler.step()

        # last epoch
        val_result_df = pd.DataFrame.from_records(
            list(zip(*val_result)), columns=["id", "label", "prob", "pred"]
        )
        val_result_df.to_csv(self.log_kwargs["prediction_path"])

        total_time = time.time() - total_start_time

        # final message
        print(f"End of training. Total time: {round(total_time, 5)} seconds")
        torch.save(self.model, self.log_kwargs["model_path"])

    def test(self, test_loader):
        """
            Test only
        """
        test_loss, test_metrics_dict, test_result = self._validate(
            test_loader, is_test=True
        )
        test_result_df = pd.DataFrame.from_records(
            list(zip(*test_result)), columns=["id", "label", "prob", "pred"]
        )
        test_result_df.to_csv(self.log_kwargs["prediction_path"])
        test_metrics_dict = {**test_metrics_dict, "test_loss": test_loss}
        self._log(test_metrics_dict, self.log_kwargs["scorefile_path"], is_test=True)

    def _log(self, update_metrics_dict, metrics_save_path, is_test=False):
        if is_test:  # test
            self.test_metrics_df = pd.concat(
                [self.test_metrics_df, pd.Series(update_metrics_dict).to_frame().T],
                ignore_index=True,
            )
            self.test_metrics_df = self.test_metrics_df.astype(
                {"tn": "int", "fp": "int", "fn": "int", "tp": "int"}
            )
            self.test_metrics_df.round(4).to_csv(metrics_save_path, index=False)
        else:  # validation
            self.valid_metrics_df = pd.concat(
                [self.valid_metrics_df, pd.Series(update_metrics_dict).to_frame().T],
                ignore_index=True,
            )
            self.valid_metrics_df = self.valid_metrics_df.astype(
                {"epoch": "int", "tn": "int", "fp": "int", "fn": "int", "tp": "int"}
            )
            self.valid_metrics_df.round(4).to_csv(metrics_save_path, index=False)

    def _train(self, loader):
        self.model.train()
        total_loss = 0.0

        for idx_batch, data_batch in enumerate(pbar := tqdm(loader)):
            ids, x, y, feature = (
                data_batch["id"],
                data_batch["recording"],
                data_batch["label"],
                data_batch["feature"],
            )
            x, y, feature = (
                x.to(self.device),
                y.to(self.device),
                feature.to(self.device),
            )
            y_hat = self.model(x, feature)

            self.optimizer.zero_grad()
            loss = self._compute_loss(y_hat, y)
            loss.backward()
            self.optimizer.step()

            total_loss += y_hat.shape[0] * loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        total_loss /= len(loader.dataset)
        return total_loss

    def _validate(self, loader, is_test=False):
        self.model.eval()
        total_loss = 0.0

        ids = []
        probs = []
        labels = []

        threshold = None
        if is_test:
            threshold = self.model.threshold

        with torch.no_grad():
            for idx_batch, data_batch in enumerate(pbar := tqdm(loader)):
                _ids, x, y, feature = (
                    data_batch["id"],
                    data_batch["recording"],
                    data_batch["label"],
                    data_batch["feature"],
                )
                x, y, feature = (
                    x.to(self.device),
                    y.to(self.device),
                    feature.to(self.device),
                )
                y_hat = self.model(x, feature)
                _probs = F.softmax(y_hat, dim=1)

                loss = self._compute_loss(y_hat, y)
                total_loss += y_hat.shape[0] * loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                ids.extend(_ids)
                labels.extend(y)
                probs.extend(_probs[:, 1])

            total_loss /= len(loader.dataset)

        labels = torch.tensor(labels).detach().cpu().numpy()
        probs = torch.tensor(probs).detach().cpu().numpy()
        metrics_dict = self._compute_metrics(labels, probs, threshold)

        if not is_test:
            threshold = metrics_dict["threshold"]
            self.model.threshold = threshold

        preds = (probs > threshold).astype(int)
        return (
            total_loss,
            metrics_dict,
            (ids, labels.tolist(), probs.tolist(), preds.tolist()),
        )

    def _compute_loss(self, y_hat, y):
        loss = self.criterion(y_hat, y)

        # apply regularization if any
        # loss += penalty.item()

        return loss

    def _compute_metrics(self, labels, probs, threshold=None):
        num_valid = len(labels)
        if threshold is None:
            threshold = self._compute_threshold(labels, probs)
        preds = probs > threshold

        # Accuracy
        num_corrrects = (preds == labels).sum()
        accuracy = num_corrrects / num_valid

        # Sensitivity, Specificity, Precision, NPV, F1-score
        tn, fp, fn, tp = metrics.confusion_matrix(labels, preds).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        npv = tn / (tn + fn)
        f1_score = 2 * precision * sensitivity / (precision + sensitivity)

        # AUROC, AUPRC
        roc_auc_score = metrics.roc_auc_score(labels, probs)
        precisions, recalls, thresholds = metrics.precision_recall_curve(labels, probs)
        auc_prc_score = metrics.auc(recalls, precisions)
        auprc_baseline = labels.sum() / num_valid

        metrics_dict = {
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "npv": npv,
            "f1_score": f1_score,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "auprc_baseline": auprc_baseline,
            "auroc": roc_auc_score,
            "auprc": auc_prc_score,
            "threshold": threshold,
        }

        return metrics_dict

    def _compute_threshold(self, labels, probs):
        # Find a threshold that maximizes Youden's J statistics
        fpr, tpr, thresholds = metrics.roc_curve(labels, probs)
        J = tpr - fpr
        threshold_idx = np.argmax(J)
        threshold = thresholds[threshold_idx]
        return threshold
