import os
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold
from sklearn.metrics import (
	f1_score,
	precision_score,
	recall_score,
	accuracy_score
)

from tqdm import tqdm

from cnn_model import MultiScaleAttentionModel


# ============================================================
# CONFIG
# ============================================================

CACHE_DIR = os.path.join(
	os.path.dirname(os.path.abspath(__file__)),
	"feature_cache_v2"
)

MODEL_DIR = os.path.join(
	os.path.dirname(os.path.abspath(__file__)),
	"saved_models"
)

os.makedirs(
	MODEL_DIR,
	exist_ok=True
)


NUM_PATIENTS = 100

WINDOWS = [
	31,
	61,
	91,
	121
]

TRAIN_SAMPLES_PER_PATIENT = 1500

POSITIVE_FRACTION = 0.30

N_FOLDS = 5

MAX_EPOCHS = 12

PATIENCE = 4

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

BATCH_SIZE = 512

TRAIN_SEED = 42

FOLD_SEED = 42


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
	"cuda"
	if torch.cuda.is_available()
	else "cpu"
)

USE_AMP = (
	DEVICE.type == "cuda"
)

print("=" * 70)
print("DEVICE")
print("=" * 70)
print(DEVICE)

if DEVICE.type == "cuda":

	print(
		"GPU:",
		torch.cuda.get_device_name(0)
	)

print(
	"Mixed precision:",
	USE_AMP
)

print()


# ============================================================
# SEED
# ============================================================

def set_seed(seed):

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	if torch.cuda.is_available():

		torch.cuda.manual_seed_all(
			seed
		)


set_seed(
	TRAIN_SEED
)


# ============================================================
# LOAD PATIENT
# ============================================================

def load_patient(
	patient_id
):

	path = os.path.join(
		CACHE_DIR,
		f"patient_{patient_id + 1:03d}.npz"
	)

	data = np.load(
		path,
		allow_pickle=False
	)

	return (
		data["features"],
		data["labels"]
	)


# ============================================================
# NORMALISATION
# ============================================================

def normalise_features(
	features
):

	x = features.astype(
		np.float32,
		copy=True
	)

	# --------------------------------------------------------
	# Basic SpO2
	# --------------------------------------------------------

	x[0] = (
		x[0] - 100.0
	) / 10.0

	x[1] /= 2.0
	x[2] /= 2.0

	# --------------------------------------------------------
	# Long-term means
	# channels 3-6
	# --------------------------------------------------------

	for i in range(
		3,
		7
	):

		x[i] = (
			x[i] - 100.0
		) / 10.0

	# --------------------------------------------------------
	# Long-term std
	# channels 7-10
	# --------------------------------------------------------

	for i in range(
		7,
		11
	):

		x[i] /= 2.0

	# --------------------------------------------------------
	# Long-term range
	# channels 11-14
	# --------------------------------------------------------

	for i in range(
		11,
		15
	):

		x[i] /= 5.0

	# --------------------------------------------------------
	# Long-term minimum
	# channels 15-18
	# --------------------------------------------------------

	for i in range(
		15,
		19
	):

		x[i] = (
			x[i] - 100.0
		) / 10.0

	# --------------------------------------------------------
	# Relative to mean
	# channels 19-22
	# --------------------------------------------------------

	for i in range(
		19,
		23
	):

		x[i] /= 5.0

	# --------------------------------------------------------
	# Distance from recent minimum
	# channels 23-25
	# --------------------------------------------------------

	for i in range(
		23,
		26
	):

		x[i] /= 5.0

	# --------------------------------------------------------
	# Drop counts
	# channels 26-28
	# --------------------------------------------------------

	for i in range(
		26,
		29
	):

		x[i] /= 10.0

	# --------------------------------------------------------
	# ECG
	# channels 29-34
	#
	# Wait: there are six ECG channels, but our feature
	# count must be 35 if all six are included.
	# --------------------------------------------------------

	# These indices are corrected below.
	return x


# ============================================================
# CORRECT FEATURE NORMALISATION
# ============================================================

def normalise_features(
	features
):

	x = features.astype(
		np.float32,
		copy=True
	)

	# --------------------------------------------------------
	# 0-2: SpO2 / derivatives
	# --------------------------------------------------------

	x[0] = (
		x[0] - 100.0
	) / 10.0

	x[1] /= 2.0
	x[2] /= 2.0

	# --------------------------------------------------------
	# 3-6: means
	# --------------------------------------------------------

	for i in range(
		3,
		7
	):

		x[i] = (
			x[i] - 100.0
		) / 10.0

	# --------------------------------------------------------
	# 7-10: standard deviations
	# --------------------------------------------------------

	for i in range(
		7,
		11
	):

		x[i] /= 2.0

	# --------------------------------------------------------
	# 11-14: ranges
	# --------------------------------------------------------

	for i in range(
		11,
		15
	):

		x[i] /= 5.0

	# --------------------------------------------------------
	# 15-18: minimums
	# --------------------------------------------------------

	for i in range(
		15,
		19
	):

		x[i] = (
			x[i] - 100.0
		) / 10.0

	# --------------------------------------------------------
	# 19-22: SpO2 - mean
	# --------------------------------------------------------

	for i in range(
		19,
		23
	):

		x[i] /= 5.0

	# --------------------------------------------------------
	# 23-25: SpO2 - recent minimum
	# --------------------------------------------------------

	for i in range(
		23,
		26
	):

		x[i] /= 5.0

	# --------------------------------------------------------
	# 26-28: drop counts
	# --------------------------------------------------------

	for i in range(
		26,
		29
	):

		x[i] /= 10.0

	# --------------------------------------------------------
	# 29: HR
	# 30: dHR
	# 31: RR
	# 32: dRR
	# 33: HR deviation
	# 34: RR deviation
	# --------------------------------------------------------

	x[29] = (
		x[29] - 70.0
	) / 30.0

	x[30] /= 10.0

	x[31] = (
		x[31] - 1.0
	) / 0.5

	x[32] /= 0.25

	x[33] /= 15.0

	x[34] /= 0.25

	return x


# ============================================================
# DATASET
# ============================================================

class WindowDataset(
	Dataset
):

	def __init__(
		self,
		indices
	):

		self.indices = indices

		self.patient_cache = {}

	def _get_patient(
		self,
		patient_id
	):

		if patient_id not in self.patient_cache:

			features, labels = load_patient(
				patient_id
			)

			features = normalise_features(
				features
			)

			self.patient_cache[
				patient_id
			] = (
				features,
				labels
			)

		return self.patient_cache[
			patient_id
		]

	def __len__(self):

		return len(
			self.indices
		)

	def __getitem__(
		self,
		idx
	):

		patient_id, second = (
			self.indices[idx]
		)

		features, labels = self._get_patient(
			patient_id
		)

		outputs = []

		for window in WINDOWS:

			half = window // 2

			start = (
				second - half
			)

			end = (
				second + half + 1
			)

			if start < 0:

				left = -start

				window_data = np.pad(
					features[
						:,
						0:end
					],
					(
						(0, 0),
						(left, 0)
					),
					mode="edge"
				)

			elif end > features.shape[1]:

				right = (
					end
					- features.shape[1]
				)

				window_data = np.pad(
					features[
						:,
						start:
					],
					(
						(0, 0),
						(0, right)
					),
					mode="edge"
				)

			else:

				window_data = features[
					:,
					start:end
				]

			outputs.append(
				torch.from_numpy(
					window_data
				)
			)

		return (
			outputs,
			torch.tensor(
				float(labels[second]),
				dtype=torch.float32
			)
		)


# ============================================================
# TRAINING INDICES
# ============================================================

def make_training_indices(
	patient_ids,
	rng
):

	all_indices = []

	for patient_id in patient_ids:

		_, labels = load_patient(
			patient_id
		)

		positive = np.flatnonzero(
			labels == 1
		)

		negative = np.flatnonzero(
			labels == 0
		)

		if len(positive) == 0:

			n_samples = min(
				TRAIN_SAMPLES_PER_PATIENT,
				len(negative)
			)

			chosen = rng.choice(
				negative,
				size=n_samples,
				replace=False
			)

			all_indices.extend(
				[
					(patient_id, int(i))
					for i in chosen
				]
			)

			continue

		n_positive = int(
			TRAIN_SAMPLES_PER_PATIENT
			* POSITIVE_FRACTION
		)

		n_negative = (
			TRAIN_SAMPLES_PER_PATIENT
			- n_positive
		)

		n_positive = min(
			n_positive,
			len(positive)
		)

		n_negative = min(
			n_negative,
			len(negative)
		)

		chosen_positive = rng.choice(
			positive,
			size=n_positive,
			replace=(
				len(positive)
				< n_positive
			)
		)

		chosen_negative = rng.choice(
			negative,
			size=n_negative,
			replace=(
				len(negative)
				< n_negative
			)
		)

		all_indices.extend(
			[
				(patient_id, int(i))
				for i in chosen_positive
			]
		)

		all_indices.extend(
			[
				(patient_id, int(i))
				for i in chosen_negative
			]
		)

	rng.shuffle(
		all_indices
	)

	return all_indices


# ============================================================
# VALIDATION INDICES
# ============================================================

def make_validation_indices(
	patient_ids
):

	all_indices = []

	for patient_id in patient_ids:

		_, labels = load_patient(
			patient_id
		)

		all_indices.extend(
			[
				(patient_id, second)
				for second in range(
					len(labels)
				)
			]
		)

	return all_indices


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
	model,
	loader,
	optimizer,
	scaler,
	loss_function
):

	model.train()

	total_loss = 0.0
	total_samples = 0

	progress = tqdm(
		loader,
		desc="Training",
		leave=False
	)

	for inputs, labels in progress:

		inputs = [
			x.to(
				DEVICE,
				non_blocking=True
			)
			for x in inputs
		]

		labels = labels.to(
			DEVICE,
			non_blocking=True
		)

		optimizer.zero_grad(
			set_to_none=True
		)

		with torch.autocast(
			device_type=DEVICE.type,
			enabled=USE_AMP
		):

			logits = model(
				inputs
			)

			loss = loss_function(
				logits,
				labels
			)

		scaler.scale(
			loss
		).backward()

		scaler.step(
			optimizer
		)

		scaler.update()

		batch_size = labels.size(0)

		total_loss += (
			loss.item()
			* batch_size
		)

		total_samples += (
			batch_size
		)

		progress.set_postfix(
			loss=f"{loss.item():.4f}"
		)

	return (
		total_loss
		/ total_samples
	)


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
	model,
	loader
):

	model.eval()

	all_predictions = []
	all_labels = []

	progress = tqdm(
		loader,
		desc="Validation",
		leave=False
	)

	for inputs, labels in progress:

		inputs = [
			x.to(
				DEVICE,
				non_blocking=True
			)
			for x in inputs
		]

		with torch.autocast(
			device_type=DEVICE.type,
			enabled=USE_AMP
		):

			logits = model(
				inputs
			)

			probabilities = torch.sigmoid(
				logits
			)

		all_predictions.append(
			probabilities.float()
			.cpu()
			.numpy()
		)

		all_labels.append(
			labels.numpy()
		)

	predictions = np.concatenate(
		all_predictions
	)

	labels = np.concatenate(
		all_labels
	)

	predicted = (
		predictions >= 0.50
	).astype(np.uint8)

	f1 = f1_score(
		labels,
		predicted,
		zero_division=0
	)

	precision = precision_score(
		labels,
		predicted,
		zero_division=0
	)

	recall = recall_score(
		labels,
		predicted,
		zero_division=0
	)

	accuracy = accuracy_score(
		labels,
		predicted
	)

	return (
		f1,
		precision,
		recall,
		accuracy,
		predictions,
		labels
	)


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
	predictions,
	labels
):

	best_f1 = -1
	best_threshold = 0.50
	best_precision = 0
	best_recall = 0
	best_accuracy = 0

	for threshold in np.arange(
		0.20,
		0.901,
		0.01
	):

		predicted = (
			predictions >= threshold
		).astype(np.uint8)

		f1 = f1_score(
			labels,
			predicted,
			zero_division=0
		)

		if f1 > best_f1:

			best_f1 = f1
			best_threshold = threshold

			best_precision = precision_score(
				labels,
				predicted,
				zero_division=0
			)

			best_recall = recall_score(
				labels,
				predicted,
				zero_division=0
			)

			best_accuracy = accuracy_score(
				labels,
				predicted
			)

	return (
		best_threshold,
		best_f1,
		best_precision,
		best_recall,
		best_accuracy
	)


# ============================================================
# MAIN
# ============================================================

def main():

	print()
	print("=" * 70)
	print("MULTI-SCALE CNN + BIGRU + ATTENTION")
	print("33-CHANNEL DESATURATION HISTORY EXPERIMENT")
	print("=" * 70)
	print()

	patient_ids = np.arange(
		NUM_PATIENTS
	)

	kfold = KFold(
		n_splits=N_FOLDS,
		shuffle=True,
		random_state=FOLD_SEED
	)

	fold_predictions = []
	fold_labels = []
	fold_results = []

	# --------------------------------------------------------
	# FOLDS
	# --------------------------------------------------------

	for fold, (
		train_ids,
		val_ids
	) in enumerate(
		kfold.split(patient_ids),
		start=1
	):

		print()
		print("=" * 70)
		print(
			f"FOLD {fold}/{N_FOLDS}"
		)
		print("=" * 70)

		print(
			"Training patients:",
			train_ids + 1
		)

		print(
			"Validation patients:",
			val_ids + 1
		)

		rng = np.random.default_rng(
			TRAIN_SEED + fold
		)

		train_indices = (
			make_training_indices(
				train_ids,
				rng
			)
		)

		val_indices = (
			make_validation_indices(
				val_ids
			)
		)

		print()
		print(
			f"Training windows: "
			f"{len(train_indices):,}"
		)

		print(
			f"Validation windows: "
			f"{len(val_indices):,}"
		)

		train_dataset = WindowDataset(
			train_indices
		)

		val_dataset = WindowDataset(
			val_indices
		)

		train_loader = DataLoader(
			train_dataset,
			batch_size=BATCH_SIZE,
			shuffle=True,
			num_workers=0,
			pin_memory=(
				DEVICE.type == "cuda"
			)
		)

		val_loader = DataLoader(
			val_dataset,
			batch_size=BATCH_SIZE,
			shuffle=False,
			num_workers=0,
			pin_memory=(
				DEVICE.type == "cuda"
			)
		)

		set_seed(
			TRAIN_SEED + fold
		)

		model = MultiScaleAttentionModel(
			num_channels=35
		).to(
			DEVICE
		)

		optimizer = torch.optim.AdamW(
			model.parameters(),
			lr=LEARNING_RATE,
			weight_decay=WEIGHT_DECAY
		)

		loss_function = (
			nn.BCEWithLogitsLoss()
		)

		scaler = torch.amp.GradScaler(
			"cuda",
			enabled=USE_AMP
		)

		best_f1 = -1
		best_epoch = 0
		epochs_without_improvement = 0

		checkpoint_path = os.path.join(
			MODEL_DIR,
			f"attention_33ch_fold_{fold}_best.pt"
		)

		# ----------------------------------------------------
		# TRAINING
		# ----------------------------------------------------

		for epoch in range(
			1,
			MAX_EPOCHS + 1
		):

			loss = train_one_epoch(
				model,
				train_loader,
				optimizer,
				scaler,
				loss_function
			)

			(
				f1,
				precision,
				recall,
				accuracy,
				_,
				_
			) = evaluate(
				model,
				val_loader
			)

			print(
				f"Epoch {epoch}/{MAX_EPOCHS} | "
				f"loss={loss:.4f} | "
				f"F1={f1:.4f} | "
				f"P={precision:.4f} | "
				f"R={recall:.4f} | "
				f"Acc={accuracy:.4f}"
			)

			if f1 > best_f1:

				best_f1 = f1
				best_epoch = epoch
				epochs_without_improvement = 0

				torch.save(
					{
						"model_state_dict":
							model.state_dict(),

						"optimizer_state_dict":
							optimizer.state_dict(),

						"epoch":
							epoch,

						"f1":
							f1,

						"num_channels":
							35,

						"windows":
							WINDOWS
					},
					checkpoint_path
				)

			else:

				epochs_without_improvement += 1

			if (
				epochs_without_improvement
				>= PATIENCE
			):

				print(
					"Early stopping"
				)

				break

		# ----------------------------------------------------
		# BEST MODEL
		# ----------------------------------------------------

		checkpoint = torch.load(
			checkpoint_path,
			map_location=DEVICE
		)

		model.load_state_dict(
			checkpoint[
				"model_state_dict"
			]
		)

		(
			f1,
			precision,
			recall,
			accuracy,
			predictions,
			labels
		) = evaluate(
			model,
			val_loader
		)

		print()
		print(
			f"Best epoch {best_epoch}"
		)

		print(
			f"Best F1 {f1:.4f}"
		)

		print(
			f"Precision {precision:.4f}"
		)

		print(
			f"Recall {recall:.4f}"
		)

		fold_predictions.append(
			predictions
		)

		fold_labels.append(
			labels
		)

		fold_results.append(
			(
				f1,
				precision,
				recall,
				accuracy,
				best_epoch
			)
		)

		del model
		del train_loader
		del val_loader
		del train_dataset
		del val_dataset

		if DEVICE.type == "cuda":

			torch.cuda.empty_cache()

	# ========================================================
	# POOLED
	# ========================================================

	all_predictions = np.concatenate(
		fold_predictions
	)

	all_labels = np.concatenate(
		fold_labels
	)

	predicted = (
		all_predictions >= 0.50
	).astype(np.uint8)

	f1 = f1_score(
		all_labels,
		predicted,
		zero_division=0
	)

	precision = precision_score(
		all_labels,
		predicted,
		zero_division=0
	)

	recall = recall_score(
		all_labels,
		predicted,
		zero_division=0
	)

	accuracy = accuracy_score(
		all_labels,
		predicted
	)

	print()
	print("=" * 70)
	print("POOLED RESULTS @ 0.50")
	print("=" * 70)

	print(
		f"F1        = {f1:.4f}"
	)

	print(
		f"Precision  = {precision:.4f}"
	)

	print(
		f"Recall     = {recall:.4f}"
	)

	print(
		f"Accuracy   = {accuracy:.4f}"
	)

	(
		best_threshold,
		best_f1,
		best_precision,
		best_recall,
		best_accuracy
	) = find_best_threshold(
		all_predictions,
		all_labels
	)

	print()
	print("=" * 70)
	print("BEST THRESHOLD")
	print("=" * 70)

	print(
		f"Threshold  = "
		f"{best_threshold:.2f}"
	)

	print(
		f"F1         = "
		f"{best_f1:.4f}"
	)

	print(
		f"Precision  = "
		f"{best_precision:.4f}"
	)

	print(
		f"Recall     = "
		f"{best_recall:.4f}"
	)

	print(
		f"Accuracy   = "
		f"{best_accuracy:.4f}"
	)

	print()
	print("=" * 70)
	print("FOLD SUMMARY")
	print("=" * 70)

	for i, result in enumerate(
		fold_results,
		start=1
	):

		f1, p, r, acc, epoch = result

		print(
			f"Fold {i}: "
			f"F1={f1:.4f} | "
			f"P={p:.4f} | "
			f"R={r:.4f} | "
			f"Acc={acc:.4f} | "
			f"Best epoch={epoch}"
		)

	# --------------------------------------------------------
	# Save OOF
	# --------------------------------------------------------

	oof_path = os.path.join(
		MODEL_DIR,
		"attention_33ch_oof.npz"
	)

	np.savez_compressed(
		oof_path,
		predictions=all_predictions,
		labels=all_labels,
		threshold=best_threshold
	)

	print()
	print(
		"OOF predictions saved:"
	)

	print(
		oof_path
	)

	print()
	print("=" * 70)
	print("DONE")
	print("=" * 70)


if __name__ == "__main__":
	main()