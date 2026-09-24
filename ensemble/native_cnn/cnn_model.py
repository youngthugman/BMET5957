import torch
import torch.nn as nn


# ============================================================
# CONFIG
# ============================================================

NUM_CHANNELS = 35

BRANCH_CHANNELS = 64
BRANCH_OUTPUT = 128

GRU_HIDDEN = 96

POOL_LENGTH = 16


# ============================================================
# TEMPORAL CNN BRANCH
# ============================================================

class TemporalCNNBranch(nn.Module):

	def __init__(
		self,
		in_channels,
		out_channels=128
	):

		super().__init__()

		self.net = nn.Sequential(

			nn.Conv1d(
				in_channels,
				BRANCH_CHANNELS,
				kernel_size=7,
				padding=3
			),

			nn.BatchNorm1d(
				BRANCH_CHANNELS
			),

			nn.ReLU(
				inplace=True
			),

			nn.Conv1d(
				BRANCH_CHANNELS,
				BRANCH_CHANNELS,
				kernel_size=7,
				padding=3
			),

			nn.BatchNorm1d(
				BRANCH_CHANNELS
			),

			nn.ReLU(
				inplace=True
			),

			nn.MaxPool1d(2),

			nn.Conv1d(
				BRANCH_CHANNELS,
				BRANCH_CHANNELS * 2,
				kernel_size=5,
				padding=2
			),

			nn.BatchNorm1d(
				BRANCH_CHANNELS * 2
			),

			nn.ReLU(
				inplace=True
			),

			nn.Conv1d(
				BRANCH_CHANNELS * 2,
				BRANCH_CHANNELS * 2,
				kernel_size=5,
				padding=2
			),

			nn.BatchNorm1d(
				BRANCH_CHANNELS * 2
			),

			nn.ReLU(
				inplace=True
			),

			nn.MaxPool1d(2),

			nn.Conv1d(
				BRANCH_CHANNELS * 2,
				out_channels,
				kernel_size=5,
				padding=2
			),

			nn.BatchNorm1d(
				out_channels
			),

			nn.ReLU(
				inplace=True
			)
		)

		self.pool = nn.AdaptiveAvgPool1d(
			POOL_LENGTH
		)

	def forward(self, x):

		x = self.net(x)

		x = self.pool(x)

		# [B, C, T] -> [B, T, C]
		x = x.transpose(
			1,
			2
		)

		return x


# ============================================================
# MULTI-SCALE CNN + BIGRU + ATTENTION
# ============================================================

class MultiScaleAttentionModel(
	nn.Module
):

	def __init__(
		self,
		num_channels=NUM_CHANNELS
	):

		super().__init__()

		self.windows = [
			31,
			61,
			91,
			121
		]

		self.branches = nn.ModuleList(
			[
				TemporalCNNBranch(
					num_channels,
					BRANCH_OUTPUT
				)

				for _ in self.windows
			]
		)

		combined_channels = (
			len(self.windows)
			* BRANCH_OUTPUT
		)

		self.bigru = nn.GRU(
			input_size=combined_channels,
			hidden_size=GRU_HIDDEN,
			num_layers=1,
			batch_first=True,
			bidirectional=True
		)

		bigru_output = (
			GRU_HIDDEN * 2
		)

		self.attention = nn.Sequential(

			nn.Linear(
				bigru_output,
				64
			),

			nn.Tanh(),

			nn.Linear(
				64,
				1
			)
		)

		self.classifier = nn.Sequential(

			nn.Linear(
				bigru_output,
				128
			),

			nn.ReLU(
				inplace=True
			),

			nn.Dropout(0.30),

			nn.Linear(
				128,
				64
			),

			nn.ReLU(
				inplace=True
			),

			nn.Dropout(0.20),

			nn.Linear(
				64,
				1
			)
		)

	def forward(self, inputs):

		branch_outputs = []

		for branch, x in zip(
			self.branches,
			inputs
		):

			x = branch(x)

			branch_outputs.append(
				x
			)

		# [B, T, 128] × 4
		x = torch.cat(
			branch_outputs,
			dim=2
		)

		x, _ = self.bigru(x)

		scores = self.attention(
			x
		)

		weights = torch.softmax(
			scores,
			dim=1
		)

		context = torch.sum(
			x * weights,
			dim=1
		)

		logits = self.classifier(
			context
		)

		return logits.squeeze(1)