import argparse
import os

import torch

from src.dataset import build_datasets_from_cv_folds, get_dataloader
from src.models import ModelWrapper
from src.setup import setup
from src.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser("arguments for training")

    # Model selection
    parser.add_argument(
        "--model", type=str, default="ecg_resnet34_psvt", help="model(from scratch)"
    )
    parser.add_argument(
        "--from_pretrained",
        dest="from_pretrained",
        action="store_true",
        help="start from pre-trained model",
    )
    parser.add_argument(
        "--pretrained_model_path", type=str, help="path for pre-trained model path"
    )

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    # Dataset
    parser.add_argument(
        "--cv_folds_path",
        type=str,
        required=True,
        help="cross-validation directory path",
    )
    parser.add_argument(
        "--valid_fold_num",
        type=int,
        default=0,
        help="validation fold number",
    )
    parser.add_argument(
        "--num_leads", type=int, default=12, help="number of leads used"
    )

    # Other settings
    parser.add_argument("--result_path", type=str, default="./cv_results")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--gpu_num", type=str, default="0")

    # features
    parser.add_argument(
        "--use_features", action="store_true", help="use feature vector as model input"
    )
    parser.add_argument(
        "--feature_dim", type=int, default=25, help="number of features used"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=25,
        help="network hidden dimension for features",
    )

    args = parser.parse_args()
    print(args)
    return args


def main():
    ### Setup
    args = parse_args()
    device = setup(args)

    result_path = f"{args.result_path}/val_fold_{args.valid_fold_num}"
    os.makedirs(result_path, exist_ok=True)

    train_dataset, valid_dataset = build_datasets_from_cv_folds(
        cv_folds_path=args.cv_folds_path,
        valid_fold_num=args.valid_fold_num,
        num_leads=args.num_leads,
    )

    # Concealed AVRT gets label 0, AVNRT gets label 1
    print(f"# of training samples: {len(train_dataset)}")
    print(f"# of validation samples: {len(valid_dataset)}")

    train_loader = get_dataloader(train_dataset, args.batch_size)
    valid_loader = get_dataloader(valid_dataset, args.batch_size)

    if args.from_pretrained:
        backbone = torch.load(args.pretrained_model_path)
    else:
        backbone = getattr(__import__("src.models", fromlist=[""]), args.model)()

    model = ModelWrapper(
        backbone,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        use_features=args.use_features,
    )

    ### Training & Validation
    log_kwargs = {
        "result_path": result_path,
        "scorefile_path": os.path.join(result_path, "scores.csv"),
        "model_path": os.path.join(result_path, "model.pt"),
        "prediction_path": os.path.join(result_path, "prediction.csv"),
    }

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1
    )

    trainer = Trainer(model, criterion, optimizer, scheduler, log_kwargs, device)
    trainer.fit(train_loader, valid_loader, args.epochs)


if __name__ == "__main__":
    main()
