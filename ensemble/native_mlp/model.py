import os
import random

import numpy as np
import torch
import torch.nn as nn

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

ECG_FEATURES = 65
SPO2_FEATURES = 27
TOTAL_FEATURES = 92


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=RANDOM_SEED):
    """Set all random seeds used by the model."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# MLP
# ============================================================

class MLP(nn.Module):

    def __init__(self, input_features):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_features,
                128,
            ),

            # LayerNorm is more stable for tabular data than
            # BatchNorm because it does not depend on batch size.
            nn.LayerNorm(128),

            nn.GELU(),

            nn.Dropout(0.25),

            nn.Linear(
                128,
                64,
            ),

            nn.LayerNorm(64),

            nn.GELU(),

            nn.Dropout(0.25),

            nn.Linear(
                64,
                1,
            ),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


# ============================================================
# MLP Classifier
# ============================================================

class MLPClassifier:

    def __init__(
        self,
        input_features,
        device,
        epochs=30,
        batch_size=512,
        learning_rate=0.001,
    ):

        self.input_features = int(input_features)
        self.device = device

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.imputer = None
        self.scaler = None
        self.model = None

        self.feature_importances_ = None

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def _validate_features(self, x):

        x = np.asarray(
            x,
            dtype=np.float32,
        )

        if x.ndim != 2:
            raise ValueError(
                f"Expected 2D feature matrix, "
                f"got shape {x.shape}"
            )

        if x.shape[1] != self.input_features:
            raise ValueError(
                f"Expected {self.input_features} features, "
                f"got {x.shape[1]}"
            )

        return x

    def _validate_labels(self, y):

        y = np.asarray(
            y,
            dtype=np.float32,
        ).ravel()

        if not np.isin(
            y,
            [0.0, 1.0],
        ).all():

            raise ValueError(
                "Labels must contain only 0 and 1."
            )

        return y

    # --------------------------------------------------------
    # Imputation
    # --------------------------------------------------------

    def _fit_imputer(self, x):

        self.imputer = SimpleImputer(
            strategy="mean"
        )

        self.imputer.fit(x)

    def _impute(self, x):

        return self.imputer.transform(x)

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    def _fit_scaler(self, x):

        self.scaler = StandardScaler()

        self.scaler.fit(x)

    def _scale(self, x):

        return self.scaler.transform(x)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    def fit(
        self,
        x,
        y,
        pos_weight,
    ):

        x = self._validate_features(x)
        y = self._validate_labels(y)

        if len(x) != len(y):
            raise ValueError(
                f"Feature/label length mismatch: "
                f"{len(x)} samples vs {len(y)} labels"
            )

        if len(x) == 0:
            raise ValueError(
                "Cannot train on an empty dataset."
            )

        # ----------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------

        self._fit_imputer(x)

        x = self._impute(x)

        self._fit_scaler(x)

        x = self._scale(x)

        # Check preprocessing did not create invalid values.
        if not np.isfinite(x).all():
            raise ValueError(
                "Non-finite values remain after "
                "imputation and scaling."
            )

        
        # ----------------------------------------------------
        # Tensors
        # ----------------------------------------------------

        # Keep dataset tensors on CPU.
        #
        # This is required when using pin_memory=True.
        # Individual batches are moved to CUDA below.
        x_tensor = torch.tensor(
            x,
            dtype=torch.float32,
        )

        y_tensor = torch.tensor(
            y,
            dtype=torch.float32,
        )

        dataset = torch.utils.data.TensorDataset(
            x_tensor,
            y_tensor,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        self.model = MLP(
            input_features=self.input_features
        ).to(self.device)

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                pos_weight,
                dtype=torch.float32,
                device=self.device,
            )
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        self.model.train()

        print(
            f"Training MLP "
            f"({self.input_features} features, "
            f"{self.epochs} epochs, "
            f"{len(loader)} batches/epoch)"
        )

        print(
            f"  Samples: {len(dataset)}"
        )

        print(
            f"  Positive samples: {int(y.sum())}"
        )

        print(
            f"  Negative samples: "
            f"{int(len(y) - y.sum())}"
        )

        print(
            f"  Positive class weight: "
            f"{pos_weight:.4f}"
        )

        for epoch in range(self.epochs):

            total_loss = 0.0

            for batch_x, batch_y in loader:

                # Move only the current batch to the
                # requested device.
                batch_x = batch_x.to(
                    self.device,
                    non_blocking=True,
                )

                batch_y = batch_y.to(
                    self.device,
                    non_blocking=True,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                logits = self.model(
                    batch_x
                )

                loss = criterion(
                    logits,
                    batch_y,
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

                total_loss += (
                    loss.item()
                    * len(batch_x)
                )

            avg_loss = (
                total_loss
                / len(dataset)
            )

            print(
                f"  Epoch "
                f"{epoch + 1:02d}/{self.epochs} "
                f"loss={avg_loss:.4f}"
            )

        print("Training complete.")

        # ----------------------------------------------------
        # Feature importance
        # ----------------------------------------------------

        self._calculate_feature_importance(
            x
        )

        self.model.eval()



    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    def predict_proba(self, x):

        if self.model is None:
            raise RuntimeError(
                "Model has not been trained."
            )

        x = self._validate_features(x)

        x = self._impute(x)
        x = self._scale(x)

        if not np.isfinite(x).all():
            raise ValueError(
                "Non-finite values encountered "
                "during prediction."
            )

        x_tensor = torch.tensor(
            x,
            dtype=torch.float32,
            device=self.device,
        )

        probabilities = []

        self.model.eval()

        with torch.no_grad():

            for start in range(
                0,
                len(x_tensor),
                self.batch_size,
            ):

                batch = x_tensor[
                    start:start + self.batch_size
                ]

                logits = self.model(
                    batch
                )

                probs = torch.sigmoid(
                    logits
                )

                probabilities.append(
                    probs.cpu().numpy()
                )

        if not probabilities:
            return np.empty(
                (0, 2),
                dtype=np.float32,
            )

        p_apnoea = np.concatenate(
            probabilities
        )

        p_normal = 1.0 - p_apnoea

        return np.column_stack(
            [
                p_normal,
                p_apnoea,
            ]
        )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    def _calculate_feature_importance(
        self,
        x,
    ):

        max_samples = 4096

        if len(x) > max_samples:

            rng = np.random.default_rng(
                RANDOM_SEED
            )

            indices = rng.choice(
                len(x),
                size=max_samples,
                replace=False,
            )

            x_sample = x[indices]

        else:

            x_sample = x

        x_tensor = torch.tensor(
            x_sample,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Keep the model in evaluation mode.
        #
        # The old implementation used train() here because
        # this was previously needed for the temporal model.
        # For this MLP it is unnecessary and could modify
        # BatchNorm running statistics.
        #
        # The current MLP uses LayerNorm anyway, but keeping
        # evaluation mode makes feature importance completely
        # non-destructive.
        # ----------------------------------------------------

        self.model.eval()

        self.model.zero_grad(
            set_to_none=True
        )

        logits = self.model(
            x_tensor
        )

        output = logits.mean()

        output.backward()

        if x_tensor.grad is None:
            raise RuntimeError(
                "Could not calculate input gradients "
                "for feature importance."
            )

        importance = (
            x_tensor.grad
            .abs()
            .mean(dim=0)
            .detach()
            .cpu()
            .numpy()
        )

        total = importance.sum()

        if total > 0:
            importance /= total

        self.feature_importances_ = (
            importance
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    def save_model(
        self,
        path,
    ):

        if self.model is None:
            raise RuntimeError(
                "Cannot save an untrained model."
            )

        directory = os.path.dirname(
            path
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        torch.save(
            {
                "model_state_dict":
                    self.model.state_dict(),

                "input_features":
                    self.input_features,

                "imputer_statistics":
                    self.imputer.statistics_,

                "scaler_mean":
                    self.scaler.mean_,

                "scaler_scale":
                    self.scaler.scale_,
            },
            path,
        )

        print(
            f"Model saved to: {path}"
        )


# ============================================================
# Device
# ============================================================

def choose_device(device_name="cpu"):
    """Select the requested PyTorch device."""

    if device_name == "cuda":

        if torch.cuda.is_available():

            device = torch.device(
                "cuda"
            )

            print(
                "Using CUDA: "
                f"{torch.cuda.get_device_name(0)}"
            )

            return device

        print(
            "CUDA requested but unavailable. "
            "Falling back to CPU."
        )

    device = torch.device(
        "cpu"
    )

    print("Using CPU.")

    return device


# ============================================================
# Main training interface
# ============================================================

def fit_model(
    x,
    y,
    device,
    weight=None,
):

    set_seed()

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=np.float32,
    ).ravel()

    if x.ndim != 2:
        raise ValueError(
            f"Expected 2D feature matrix, "
            f"got shape {x.shape}"
        )

    if len(x) != len(y):
        raise ValueError(
            f"Feature/label length mismatch: "
            f"{len(x)} vs {len(y)}"
        )

    input_features = x.shape[1]

    # --------------------------------------------------------
    # Calculate class weight automatically when one was not
    # explicitly supplied.
    # --------------------------------------------------------

    if weight is None:

        positive = np.sum(
            y == 1
        )

        negative = np.sum(
            y == 0
        )

        if positive == 0:
            raise ValueError(
                "Training data contains no "
                "positive samples."
            )

        if negative == 0:
            raise ValueError(
                "Training data contains no "
                "negative samples."
            )

        weight = np.sqrt(
            negative / positive
        )

    model = MLPClassifier(
        input_features=input_features,
        device=device,
    )

    model.fit(
        x,
        y,
        pos_weight=weight,
    )

    return model, device