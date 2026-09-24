import os
import numpy as np
import h5py


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = r"C:\Users\My Computer\Documents\UNI stuff\MachineLearning\ProjectTrainData.mat"

CACHE_DIR = os.path.join(
	os.path.dirname(os.path.abspath(__file__)),
	"feature_cache_v2"
)

FS_ECG = 200

# QRS cleaning
MIN_RR = 0.30
MAX_RR = 3.00
MIN_HR = 20.0
MAX_HR = 200.0

# SpO2 validity
MIN_SPO2 = 50.0
MAX_SPO2 = 100.0


# ============================================================
# UTILITIES
# ============================================================

def ensure_cache_dir():
	os.makedirs(CACHE_DIR, exist_ok=True)


def matlab_ref_to_array(f, ref):
	return np.asarray(f[ref]).squeeze()


def clean_spo2(spo2):

	spo2 = np.asarray(
		spo2,
		dtype=np.float32
	).copy()

	valid = np.isfinite(spo2)
	valid &= (spo2 >= MIN_SPO2)
	valid &= (spo2 <= MAX_SPO2)

	if np.all(valid):
		return spo2

	if np.sum(valid) == 0:
		return np.full_like(
			spo2,
			97.0
		)

	x = np.arange(
		len(spo2),
		dtype=np.float64
	)

	spo2[~valid] = np.interp(
		x[~valid],
		x[valid],
		spo2[valid]
	)

	return spo2


# ============================================================
# ROLLING STATISTICS
# ============================================================

def trailing_mean(x, window):

	result = np.empty_like(
		x,
		dtype=np.float32
	)

	cumsum = np.concatenate(
		[
			np.array([0.0], dtype=np.float64),
			np.cumsum(
				x,
				dtype=np.float64
			)
		]
	)

	for i in range(len(x)):

		start = max(
			0,
			i - window + 1
		)

		count = i - start + 1

		result[i] = (
			cumsum[i + 1]
			- cumsum[start]
		) / count

	return result


def trailing_std(x, window):

	result = np.empty_like(
		x,
		dtype=np.float32
	)

	mean = trailing_mean(
		x,
		window
	)

	squared = x.astype(
		np.float64
	) ** 2

	cumsum = np.concatenate(
		[
			np.array([0.0], dtype=np.float64),
			np.cumsum(
				squared,
				dtype=np.float64
			)
		]
	)

	for i in range(len(x)):

		start = max(
			0,
			i - window + 1
		)

		count = i - start + 1

		mean_square = (
			cumsum[i + 1]
			- cumsum[start]
		) / count

		variance = (
			mean_square
			- float(mean[i]) ** 2
		)

		result[i] = np.sqrt(
			max(variance, 0.0)
		)

	return result


def trailing_min(x, window):

	result = np.empty_like(
		x,
		dtype=np.float32
	)

	for i in range(len(x)):

		start = max(
			0,
			i - window + 1
		)

		result[i] = np.min(
			x[start:i + 1]
		)

	return result


def trailing_max(x, window):

	result = np.empty_like(
		x,
		dtype=np.float32
	)

	for i in range(len(x)):

		start = max(
			0,
			i - window + 1
		)

		result[i] = np.max(
			x[start:i + 1]
		)

	return result


def trailing_range(x, window):

	return (
		trailing_max(x, window)
		-
		trailing_min(x, window)
	)


def trailing_drop_count(
	dspo2,
	window,
	threshold=-2.0
):

	"""
	Count the number of individual one-second drops
	of at least 2 percentage points within the trailing
	window.
	"""

	is_drop = (
		dspo2 <= threshold
	).astype(np.int32)

	cumsum = np.concatenate(
		[
			np.array([0], dtype=np.int64),
			np.cumsum(
				is_drop,
				dtype=np.int64
			)
		]
	)

	result = np.empty(
		len(dspo2),
		dtype=np.float32
	)

	for i in range(len(dspo2)):

		start = max(
			0,
			i - window + 1
		)

		result[i] = (
			cumsum[i + 1]
			- cumsum[start]
		)

	return result


# ============================================================
# ECG FEATURES
# ============================================================

def build_ecg_features(
	qrs,
	n_seconds
):

	qrs = np.asarray(
		qrs,
		dtype=np.float64
	).reshape(-1)

	if len(qrs) < 3:

		hr = np.full(
			n_seconds,
			60.0,
			dtype=np.float32
		)

		rr = np.full(
			n_seconds,
			1.0,
			dtype=np.float32
		)

		return (
			hr,
			np.zeros_like(hr),
			rr,
			np.zeros_like(rr)
		)

	qrs = np.sort(qrs)

	rr = np.diff(qrs) / FS_ECG

	rr_times = (
		qrs[:-1]
		+
		qrs[1:]
	) / (
		2.0 * FS_ECG
	)

	valid_rr = np.isfinite(rr)
	valid_rr &= rr >= MIN_RR
	valid_rr &= rr <= MAX_RR

	rr_valid = rr[valid_rr]
	rr_times_valid = rr_times[valid_rr]

	if len(rr_valid) == 0:

		hr = np.full(
			n_seconds,
			60.0,
			dtype=np.float32
		)

		rr_out = np.full(
			n_seconds,
			1.0,
			dtype=np.float32
		)

		return (
			hr,
			np.zeros_like(hr),
			rr_out,
			np.zeros_like(rr_out)
		)

	hr_values = 60.0 / rr_valid

	valid_hr = np.isfinite(
		hr_values
	)

	valid_hr &= (
		hr_values >= MIN_HR
	)

	valid_hr &= (
		hr_values <= MAX_HR
	)

	hr_values = hr_values[valid_hr]
	rr_valid = rr_valid[valid_hr]
	rr_times_valid = rr_times_valid[valid_hr]

	if len(hr_values) == 0:

		hr = np.full(
			n_seconds,
			60.0,
			dtype=np.float32
		)

		rr_out = np.full(
			n_seconds,
			1.0,
			dtype=np.float32
		)

		return (
			hr,
			np.zeros_like(hr),
			rr_out,
			np.zeros_like(rr_out)
		)

	t = np.arange(
		n_seconds,
		dtype=np.float64
	)

	hr = np.interp(
		t,
		rr_times_valid,
		hr_values,
		left=hr_values[0],
		right=hr_values[-1]
	)

	rr_out = np.interp(
		t,
		rr_times_valid,
		rr_valid,
		left=rr_valid[0],
		right=rr_valid[-1]
	)

	hr = np.clip(
		hr,
		MIN_HR,
		MAX_HR
	)

	rr_out = np.clip(
		rr_out,
		MIN_RR,
		MAX_RR
	)

	hr = hr.astype(
		np.float32
	)

	rr_out = rr_out.astype(
		np.float32
	)

	dhr = np.diff(
		hr,
		prepend=hr[0]
	)

	drr = np.diff(
		rr_out,
		prepend=rr_out[0]
	)

	dhr = np.clip(
		dhr,
		-30.0,
		30.0
	)

	drr = np.clip(
		drr,
		-0.5,
		0.5
	)

	return (
		hr,
		dhr.astype(np.float32),
		rr_out,
		drr.astype(np.float32)
	)


# ============================================================
# SPO2 FEATURES
# ============================================================

def build_spo2_features(spo2):

	spo2 = clean_spo2(
		spo2
	)

	# --------------------------------------------------------
	# Basic derivatives
	# --------------------------------------------------------

	dspo2 = np.diff(
		spo2,
		prepend=spo2[0]
	)

	d2spo2 = np.diff(
		dspo2,
		prepend=dspo2[0]
	)

	dspo2 = np.clip(
		dspo2,
		-10.0,
		10.0
	)

	d2spo2 = np.clip(
		d2spo2,
		-10.0,
		10.0
	)

	# --------------------------------------------------------
	# Long-term statistics
	# --------------------------------------------------------

	windows = [
		30,
		60,
		120,
		300
	]

	means = []
	stds = []
	ranges = []
	mins = []

	for window in windows:

		print(
			f"      Calculating "
			f"{window}s statistics..."
		)

		mean = trailing_mean(
			spo2,
			window
		)

		std = trailing_std(
			spo2,
			window
		)

		minimum = trailing_min(
			spo2,
			window
		)

		maximum = trailing_max(
			spo2,
			window
		)

		range_value = (
			maximum - minimum
		)

		means.append(mean)
		stds.append(std)
		mins.append(minimum)
		ranges.append(range_value)

	# --------------------------------------------------------
	# Relative-to-baseline
	# --------------------------------------------------------

	relative_means = [
		spo2 - mean
		for mean in means
	]

	# --------------------------------------------------------
	# Distance above recent minimum
	# --------------------------------------------------------

	distance_min = [
		spo2 - mins[i]
		for i in [
			1,	# 60s
			2,	# 120s
			3	# 300s
		]
	]

	# --------------------------------------------------------
	# Significant drop counts
	# --------------------------------------------------------

	drop_counts = [
		trailing_drop_count(
			dspo2,
			window,
			threshold=-2.0
		)

		for window in [
			60,
			120,
			300
		]
	]

	# --------------------------------------------------------
	# Assemble
	# --------------------------------------------------------

	return (
		[
			spo2,
			dspo2,
			d2spo2
		]
		+
		means
		+
		stds
		+
		ranges
		+
		mins
		+
		relative_means
		+
		distance_min
		+
		drop_counts
	)


# ============================================================
# PATIENT FEATURES
# ============================================================

def build_patient_features(
	spo2,
	qrs
):

	spo2_features = build_spo2_features(
		spo2
	)

	hr, dhr, rr, drr = build_ecg_features(
		qrs,
		len(spo2)
	)

	# 30-second HR baseline
	hr_baseline = trailing_mean(
		hr,
		30
	)

	# 30-second RR baseline
	rr_baseline = trailing_mean(
		rr,
		30
	)

	hr_deviation = (
		hr - hr_baseline
	)

	rr_deviation = (
		rr - rr_baseline
	)

	features = np.stack(
		spo2_features
		+
		[
			hr,
			dhr,
			rr,
			drr,
			hr_deviation,
			rr_deviation
		],
		axis=0
	)

	print(
		f"      Final feature shape: "
		f"{features.shape}"
	)

	return features.astype(
		np.float32
	)


# ============================================================
# BUILD CACHE
# ============================================================

def build_cache():

	ensure_cache_dir()

	print("=" * 70)
	print("BUILDING FEATURE CACHE V2")
	print("=" * 70)
	print(
		f"Data file: {DATA_FILE}"
	)
	print(
		f"Cache: {CACHE_DIR}"
	)
	print()

	with h5py.File(
		DATA_FILE,
		"r"
	) as f:

		spo2_refs = (
			f["SpO2"][:]
			.reshape(-1)
		)

		qrs_refs = (
			f["QRS"][:]
			.reshape(-1)
		)

		class_refs = (
			f["Class"][:]
			.reshape(-1)
		)

		n_patients = len(
			spo2_refs
		)

		print(
			f"Patients: {n_patients}"
		)
		print()

		for patient_idx in range(
			n_patients
		):

			cache_file = os.path.join(
				CACHE_DIR,
				f"patient_{patient_idx + 1:03d}.npz"
			)

			if os.path.exists(
				cache_file
			):

				print(
					f"[{patient_idx + 1:03d}/"
					f"{n_patients}] "
					f"already cached"
				)

				continue

			print(
				f"[{patient_idx + 1:03d}/"
				f"{n_patients}] "
				f"Loading..."
			)

			spo2 = matlab_ref_to_array(
				f,
				spo2_refs[patient_idx]
			)

			qrs = matlab_ref_to_array(
				f,
				qrs_refs[patient_idx]
			)

			labels_raw = matlab_ref_to_array(
				f,
				class_refs[patient_idx]
			)

			spo2 = np.asarray(
				spo2
			).reshape(-1)

			qrs = np.asarray(
				qrs
			).reshape(-1)

			labels_raw = np.asarray(
				labels_raw
			).reshape(-1)

			# ------------------------------------------------
			# Labels
			# ------------------------------------------------

			labels = np.zeros(
				len(labels_raw),
				dtype=np.uint8
			)

			if labels_raw.dtype.kind in (
				"U",
				"S",
				"O"
			):

				for i, value in enumerate(
					labels_raw
				):

					if str(value) == "A":
						labels[i] = 1

			else:

				# ASCII A
				labels[
					labels_raw == 65
				] = 1

				# Already-binary support
				labels[
					labels_raw == 1
				] = 1

			if len(spo2) != len(labels):

				raise ValueError(
					f"Patient {patient_idx + 1}: "
					f"SpO2/Class length mismatch"
				)

			# ------------------------------------------------
			# Features
			# ------------------------------------------------

			features = build_patient_features(
				spo2,
				qrs
			)

			if features.shape[0] != 35:

				raise ValueError(
					f"Expected 35 features, "
					f"got {features.shape[0]}"
				)

			np.savez_compressed(
				cache_file,
				features=features,
				labels=labels
			)

			print(
				f"      Saved | "
				f"{len(labels):,} seconds | "
				f"apnoea={labels.mean():.3f}"
			)

			print()

	print(
		"Feature cache V2 complete."
	)


if __name__ == "__main__":
	build_cache()