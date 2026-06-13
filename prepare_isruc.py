import os
import shutil
import numpy as np
import mne
from scipy.signal import spectrogram
from sklearn.preprocessing import MinMaxScaler

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    LOW_FREQ,
    HIGH_FREQ,
    EPOCH_SECONDS,
    TARGET_SAMPLE_RATE,
    N_PER_SEG,
    N_OVERLAP
)

# =====================================================
# ISRUC paths
# =====================================================
ISRUC_RAW_DIR = os.path.join(RAW_DATA_DIR, "isruc")
ISRUC_PROCESSED_DIR = os.path.join(PROCESSED_DATA_DIR, "isruc")

os.makedirs(ISRUC_PROCESSED_DIR, exist_ok=True)

ISRUC_CHANNEL = "F3-A2"

# ISRUC label mapping
# ISRUC labels:
# 0 = Wake, 1 = N1, 2 = N2, 3 = N3, 5 = REM
# Project labels:
# 0 = Wake, 1 = N1, 2 = N2, 3 = N3, 4 = REM
LABEL_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    5: 4
}


def find_signal_file(subject_folder):
    """
    Finds .edf or .rec signal file, including nested folders.
    If only .rec is found, a temporary .edf copy is created for MNE.
    """

    for root, dirs, files in os.walk(subject_folder):
        for file in files:
            if file.lower().endswith(".edf"):
                return os.path.join(root, file)

        for file in files:
            if file.lower().endswith(".rec"):
                rec_path = os.path.join(root, file)
                edf_path = os.path.join(root, file[:-4] + ".edf")

                if not os.path.exists(edf_path):
                    shutil.copy(rec_path, edf_path)

                return edf_path

    return None


def find_label_file(subject_folder, subject_id):
    """
    Finds ISRUC sleep-stage label file.
    Prefers scorer 1, then scorer 2.
    """

    preferred_names = [
        f"{subject_id}_1.txt",
        f"{subject_id}_2.txt"
    ]

    all_txt_files = []

    for root, dirs, files in os.walk(subject_folder):
        for file in files:
            if file.lower().endswith(".txt"):
                full_path = os.path.join(root, file)
                all_txt_files.append(full_path)

                if file in preferred_names:
                    return full_path

    if len(all_txt_files) > 0:
        return all_txt_files[0]

    return None


def load_isruc_recording(signal_path):
    """
    Loads one ISRUC EDF/REC recording and applies standard preprocessing.
    """

    raw = mne.io.read_raw_edf(signal_path, preload=True, verbose=False)

    if ISRUC_CHANNEL not in raw.ch_names:
        raise ValueError(
            f"Required channel '{ISRUC_CHANNEL}' not found.\n"
            f"Available channels: {raw.ch_names}"
        )

    raw.pick([ISRUC_CHANNEL])
    raw.filter(LOW_FREQ, HIGH_FREQ, verbose=False)
    raw.resample(TARGET_SAMPLE_RATE, verbose=False)

    return raw


def load_labels(label_path):
    """
    Loads ISRUC numerical sleep-stage labels and maps them to project labels.
    """

    labels = []

    with open(label_path, "r") as file:
        for line in file:
            line = line.strip()

            if line == "":
                continue

            try:
                original_label = int(line)

                if original_label in LABEL_MAP:
                    labels.append(LABEL_MAP[original_label])

            except ValueError:
                continue

    return np.array(labels)


def extract_epochs_from_labels(raw, labels):
    """
    Extracts 30-second epochs from ISRUC signal using TXT label sequence.
    """

    signal = raw.get_data()[0]
    sfreq = int(raw.info["sfreq"])
    epoch_samples = sfreq * EPOCH_SECONDS

    max_epochs_from_signal = len(signal) // epoch_samples
    usable_epochs = min(max_epochs_from_signal, len(labels))

    X_epochs = []
    y_labels = []

    for i in range(usable_epochs):
        start = i * epoch_samples
        end = start + epoch_samples

        epoch_signal = signal[start:end]
        label = labels[i]

        X_epochs.append(epoch_signal)
        y_labels.append(label)

    return X_epochs, y_labels, sfreq


def make_spectrogram(epoch_signal, sfreq):
    """
    Converts one EEG epoch into a normalized spectrogram.
    """

    freqs, times, spec = spectrogram(
        epoch_signal,
        fs=sfreq,
        nperseg=N_PER_SEG,
        noverlap=N_OVERLAP
    )

    mask = freqs <= 30
    spec = spec[mask, :]

    spec = np.log1p(spec)

    scaler = MinMaxScaler()
    spec = scaler.fit_transform(spec)

    return spec.astype(np.float32)


def process_subject(subject_folder, subject_id, cohort_name):
    """
    Processes one ISRUC subject into spectrograms and labels.
    """

    signal_path = find_signal_file(subject_folder)
    label_path = find_label_file(subject_folder, subject_id)

    if signal_path is None:
        print(f"No REC/EDF file found for {cohort_name} subject {subject_id}")
        return [], []

    if label_path is None:
        print(f"No label file found for {cohort_name} subject {subject_id}")
        return [], []

    print("\nProcessing ISRUC subject:")
    print("Cohort:", cohort_name)
    print("Subject:", subject_id)
    print("Signal file:", signal_path)
    print("Label file:", label_path)

    try:
        raw = load_isruc_recording(signal_path)
        labels = load_labels(label_path)

        X_epochs, y_labels, sfreq = extract_epochs_from_labels(raw, labels)

        X_specs = []

        for epoch_signal in X_epochs:
            spec = make_spectrogram(epoch_signal, sfreq)
            X_specs.append(spec)

        print(f"Usable epochs: {len(X_specs)}")

        return X_specs, y_labels

    except Exception as e:
        print(f"Error processing ISRUC subject {subject_id}: {e}")
        return [], []


def main():
    if not os.path.exists(ISRUC_RAW_DIR):
        print("ISRUC raw folder not found.")
        print("Expected folder:")
        print(ISRUC_RAW_DIR)
        return

    cohort_folders = [
        folder for folder in os.listdir(ISRUC_RAW_DIR)
        if os.path.isdir(os.path.join(ISRUC_RAW_DIR, folder))
        and folder.startswith("ISRUC_")
    ]

    cohort_folders = sorted(cohort_folders)

    if len(cohort_folders) == 0:
        print("No ISRUC cohort folders found.")
        return

    X_all = []
    y_all = []

    print("Found ISRUC cohorts:", cohort_folders)

    for cohort_name in cohort_folders:
        cohort_path = os.path.join(ISRUC_RAW_DIR, cohort_name)

        subject_folders = [
            folder for folder in os.listdir(cohort_path)
            if os.path.isdir(os.path.join(cohort_path, folder))
            and folder.isdigit()
        ]

        subject_ids = sorted([int(folder) for folder in subject_folders])

        print("\n==============================")
        print(f"Processing cohort: {cohort_name}")
        print(f"Found subjects: {len(subject_ids)}")
        print("==============================")

        for subject_id in subject_ids:
            subject_folder = os.path.join(cohort_path, str(subject_id))

            X_subject, y_subject = process_subject(
                subject_folder,
                subject_id,
                cohort_name
            )

            X_all.extend(X_subject)
            y_all.extend(y_subject)

    if len(X_all) == 0:
        print("No ISRUC spectrograms were generated.")
        return

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    X_all = X_all[:, np.newaxis, :, :]

    X_save_path = os.path.join(ISRUC_PROCESSED_DIR, "X_isruc.npy")
    y_save_path = os.path.join(ISRUC_PROCESSED_DIR, "y_isruc.npy")

    np.save(X_save_path, X_all)
    np.save(y_save_path, y_all)

    print("\nISRUC preprocessing completed successfully.")
    print("Saved X:", X_save_path)
    print("Saved y:", y_save_path)
    print("X shape:", X_all.shape)
    print("y shape:", y_all.shape)

    unique, counts = np.unique(y_all, return_counts=True)

    print("\nClass distribution:")
    for label, count in zip(unique, counts):
        stage = ["Wake", "N1", "N2", "N3", "REM"][int(label)]
        print(f"Class {label} ({stage}): {count}")


if __name__ == "__main__":
    main()