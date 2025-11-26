import argparse

import lightning as L
import numpy as np
import random
import torch

from datamodules import NumPyDataModule, GraphDataModule
from gnn.models import GNNModel
from mlp.models import MLPModel

from lightning.pytorch.callbacks import ModelCheckpoint


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    L.seed_everything(seed, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["gnn", "mlp"],
        default="gnn")
    parser.add_argument(
        "--layer-name",
        type=str,
        choices=["GraphSAGE", "GAT", "GATv2"],
        default="GraphSAGE")
    parser.add_argument(
        "--eval-metric",
        type=str,
        choices=["accuracy", "auc", "f1", "mse"],
        default="accuracy")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--input-dropout", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--test-batch-size", type=int, default=4096 * 2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1108)
    parser.add_argument("--test-on-gpu", action="store_true")
    parser.add_argument("--test-after-train", action="store_true")
    parser.add_argument("--checkpoint-path", type=str, default=".")
    args = parser.parse_args()

    fix_seed(args.seed)

    if args.model_type == "gnn":
        datamodule = GraphDataModule(
            data_path=args.data_path,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            num_workers=args.num_workers,
        )
        modelmodule = GNNModel(
            in_channels=datamodule.train_dataset[0]["x"].shape[-1],
            hidden_channels=args.hidden_channels,
            out_channels=args.num_classes,
            layer_name=args.layer_name,
            num_layers=args.num_layers,
            in_dropout=args.input_dropout,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            eval_metric=args.eval_metric,
        )
    else:
        datamodule = NumPyDataModule(
            data_path=args.data_path,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            num_workers=args.num_workers,
        )
        modelmodule = MLPModel(
            in_channels=datamodule.train_dataset[0]["x"].shape[-1],
            hidden_channels=args.hidden_channels,
            out_channels=args.num_classes,
            num_layers=args.num_layers,
            in_dropout=args.input_dropout,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            eval_metric=args.eval_metric,
        )

    # Train and validate.
    best_mode = "min" if args.eval_metric == "mse" else "max"
    trainer = L.Trainer(
        default_root_dir=args.checkpoint_path,
        devices=args.num_devices,
        accelerator="gpu",
        strategy="ddp",
        max_epochs=args.max_epochs,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=1,
        accumulate_grad_batches=1,
        enable_progress_bar=True,
        callbacks=[
            ModelCheckpoint(
                save_top_k=1,
                save_last=True,
                mode=best_mode,
                monitor="val_result",
                save_weights_only=True,
                save_on_train_epoch_end=False,
                filename="{epoch}-{step}-{val_loss:.3f}-{val_result:.3f}",
            ),
        ],
    )
    trainer.fit(modelmodule, datamodule=datamodule)
    
    if args.test_after_train:
        # Test on a single device using the best validation checkpoint.
        accelerator = "gpu" if args.test_on_gpu else "cpu"
        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        tester = L.Trainer(
            devices=1,
            accelerator=accelerator,
            enable_progress_bar=True,
        )
        tester.test(modelmodule, datamodule=datamodule, ckpt_path=best_ckpt_path)
