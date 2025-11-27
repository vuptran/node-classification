import lightning as L

import torch
import torch.nn as nn

from torchmetrics import (
    AUROC,
    F1Score,
    Accuracy,
    MeanSquaredError
)


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=3,
        in_dropout=0.1,
        dropout=0.5,
        batch_norm=True,
        residual=True,
    ):
        super(MultiLayerPerceptron, self).__init__()
        self.num_layers = num_layers
        self.in_dropout = in_dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if self.batch_norm:
            self.bn_layers = nn.ModuleList()
            self.bn_layers.extend([
                nn.BatchNorm1d(hidden_channels)
                for _ in range(num_layers)
            ])
        self.linear_layers = nn.ModuleList()
        self.proj_layers = nn.ModuleList()

        channels = in_channels
        for _ in range(num_layers):
            self.linear_layers.append(
                nn.Linear(channels, hidden_channels)
            )
            self.proj_layers.append(
                nn.Linear(channels, hidden_channels)
            )
            channels = hidden_channels
    
        self.input_dropout = nn.Dropout(in_dropout)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)
        self.linear_head = nn.Linear(hidden_channels, out_channels)
                                  
    def forward(self, x, return_embeddings=False):
        if self.in_dropout:
            x = self.input_dropout(x)
        
        for i in range(self.num_layers):
            z = x
            x = self.linear_layers[i](x)
            if self.residual:
                z = self.proj_layers[i](z)
                x = x + z
            if self.batch_norm:
                x = self.bn_layers[i](x)
            x = self.activation(x)
            x = self.dropout(x)

        linear_predictions = self.linear_head(x)
        if return_embeddings:
            return linear_predictions, x
        else:
            return linear_predictions


class MLPModel(L.LightningModule):
    def __init__(
        self,
        in_channels,
        hidden_channels=128,
        out_channels=1,
        residual=True,
        batch_norm=True,
        num_layers=3,
        in_dropout=0.1,
        dropout=0.5,
        learning_rate=0.001,
        eval_metric="accuracy",
    ):
        super().__init__()
        self.model = MultiLayerPerceptron(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            residual=residual,
            batch_norm=batch_norm,
            num_layers=num_layers,
            in_dropout=in_dropout,
            dropout=dropout,
        )
        self.test_outputs = []
        self.eval_metric = eval_metric
        self.criterion = nn.CrossEntropyLoss()
        if eval_metric in ["f1", "accuracy"]:
            METRICS = {
                "f1": F1Score,
                "accuracy": Accuracy,
            }
            self.metric = METRICS[eval_metric](
                task="multiclass",
                num_classes=out_channels,
                average="micro",
            )
        elif eval_metric == "auc":
            self.metric = AUROC(task="binary")
        elif eval_metric == "mse":
            self.criterion = nn.MSELoss()
            self.metric = MeanSquaredError()
        self.learning_rate = learning_rate

    def forward(self, x, y):
        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        return loss, outputs

    def forward_step(self, batch, batch_idx, step_mode):
        x = batch["x"]
        labels = batch["y"]
        loss, outputs = self.forward(x, labels)
        if step_mode in ["val", "test"]:
            if self.eval_metric in ["f1", "accuracy"]:
                outputs = outputs.argmax(dim=-1)
            elif self.eval_metric == "auc":
                outputs = torch.softmax(outputs, dim=-1)[:, 1]
        return loss, outputs, labels
    
    def training_step(self, batch, batch_idx):
        loss, outputs, labels = self.forward_step(
            batch, batch_idx, step_mode="train")
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs, labels = self.forward_step(
            batch, batch_idx, step_mode="val")
        self.metric(outputs, labels)
        self.log_dict(
            {"val_loss": loss, "val_result": self.metric},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
    
    def test_step(self, batch, batch_idx):
        loss, outputs, labels = self.forward_step(
            batch, batch_idx, step_mode="test")
        self.test_outputs.append(
            {"loss": loss.item(), "predictions": outputs, "labels": labels}
        )

    def on_test_epoch_end(self):
        predictions = torch.cat(
            [output["predictions"] for output in self.test_outputs]
        )
        labels = torch.cat(
            [output["labels"] for output in self.test_outputs]
        )
        test_loss = sum(
            output["loss"] for output in self.test_outputs
        ) / len(self.test_outputs)
        test_metric = self.metric(predictions, labels)
        self.log_dict(
            {"test_loss": test_loss, f"test_{self.eval_metric}": test_metric}
        )
        self.test_outputs.clear()
        self.metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate)
        return optimizer
