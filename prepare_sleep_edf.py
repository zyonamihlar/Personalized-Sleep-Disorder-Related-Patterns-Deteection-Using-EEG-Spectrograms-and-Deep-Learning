import os
import numpy as np
import mne
from scipy.signal import spectrogram
from sklearn.preprocessing import MinMaxScaler

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    EEG_CHANNEL,
    LOW_FREQ,
    HIGH_FREQ,
    EPOCH_SECONDS,
    TARGET_SAMPLE_RATE,
    N_PER_SEG,
    N_OVERLAP
)

# =====================================================
# Sleep-EDF paths
# =====================================================
SLEEP_EDF_RAW_DIR = os.path.join(RAW_DATA_DIR, "sleep_edf")
SLEEP_EDF_PROCESSED_DIR = os.path.join(PROCESSED_DATA_DIR, "sleep_edf")

os.makedirs(SLEEP_EDF_PROCESSED_DIR, exist_ok=True)

# Sleep-EDF label mapping
# Project label format:
# 0 = Wake, 1 = N1, 2 = N2, 3 = N3, 4 = REM
LABEL_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,   # merge old stage 4 with N3
    "Sleep stage R": 4
}


def find_sleep_edf_pairs():
    """
    Finds PSG signal files and matching Hypnogram annotation files.
    """

    files = [
        file for file in os.listdir(SLEEP_EDF_RAW_DIR)
        if file.lower().endswith(".edf")
    ]

    signal_files = [
        file for file in files
        if "hypnogram" not in file.lower()
    ]

    annotation_files = [
        file for file in files
        if "hypnogram" in file.lower()
    ]

    pairs = []

    for signal_file in signal_files:
        subject_code = signal_file[:6]
        matched_annotation = None

        for annotation_file in annotation_files:
            if annotation_file.startswith(subject_code):
                matched_annotation = annotation_file
                break

        if matched_annotation is not None:
            pairs.append((
                os.path.join(SLEEP_EDF_RAW_DIR, signal_file),
                os.path.join(SLEEP_EDF_RAW_DIR, matched_annotation)
            ))
        else:
            print(f"No matching hypnogram found for {signal_file}")

    return pairs


def load_sleep_edf_recording(signal_path, annotation_path):
    """
    Loads one Sleep-EDF PSG file and its corresponding hypnogram.
    """

    raw = mne.io.read_raw_edf(signal_path, preload=True, verbose=False)
    annotations = mne.read_annotations(annotation_path)
    raw.set_annotations(annotations)

    print("Available channels:", raw.ch_names)

    if EEG_CHANNEL not in raw.ch_names:
        raise ValueError(
            f"Required channel '{EEG_CHANNEL}' not found.\n"
            f"Available channels: {raw.ch_names}"
        )

    raw.pick([EEG_CHANNEL])
    raw.filter(LOW_FREQ, HIGH_FREQ, verbose=False)
    raw.resample(TARGET_SAMPLE_RATE, verbose=False)

    return raw


def extract_epochs_from_annotations(raw):
    """
    Extracts 30-second EEG epochs and corresponding labels
    from Sleep-EDF hypnogram annotations.
    """

    signal = raw.get_data()[0]
    sfreq = int(raw.info["sfreq"])
    epoch_samples = sfreq * EPOCH_SECONDS

    X_epochs = []
    y_labels = []

    for annotation in raw.annotations:
        label_name = annotation["description"]

        if label_name not in LABEL_MAP:
            continue

        label = LABEL_MAP[label_name]

        start_sample = int(annotation["onset"] * sfreq)
        duration_samples = int(annotation["duration"] * sfreq)

        usable_epochs = duration_samples // epoch_samples

        for i in range(usable_epochs):
            start = start_sample + i * epoch_samples
            end = start + epoch_samples

            if end <= len(signal):
                epoch_signal = signal[start:end]
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


def process_recording(signal_path, annotation_path):
    """
    Processes one Sleep-EDF recording into spectrograms and labels.
    """

    print("\nProcessing Sleep-EDF recording:")
    print("Signal file:", signal_path)
    print("Annotation file:", annotation_path)

    try:
        raw = load_sleep_edf_recording(signal_path, annotation_path)
        X_epochs, y_labels, sfreq = extract_epochs_from_annotations(raw)

        X_specs = []

        for epoch_signal in X_epochs:
            spec = make_spectrogram(epoch_signal, sfreq)
            X_specs.append(spec)

        print(f"Usable epochs: {len(X_specs)}")

        return X_specs, y_labels

    except Exception as e:
        print(f"Error processing recording: {e}")
        return [], []


def main():
    if not os.path.exists(SLEEP_EDF_RAW_DIR):
        print("Sleep-EDF raw folder not found.")
        print("Expected folder:")
        print(SLEEP_EDF_RAW_DIR)
        return

    pairs = find_sleep_edf_pairs()

    if len(pairs) == 0:
        print("No Sleep-EDF PSG/Hypnogram pairs found.")
        return

    X_all = []
    y_all = []

    print(f"Found {len(pairs)} Sleep-EDF PSG/Hypnogram pairs.")

    for signal_path, annotation_path in pairs:
        X_recording, y_recording = process_recording(
            signal_path,
            annotation_path
        )

        X_all.extend(X_recording)
        y_all.extend(y_recording)

    if len(X_all) == 0:
        print("No Sleep-EDF spectrograms were generated.")
        return

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    X_all = X_all[:, np.newaxis, :, :]

    X_save_path = os.path.join(SLEEP_EDF_PROCESSED_DIR, "X_sleep_edf.npy")
    y_save_path = os.path.join(SLEEP_EDF_PROCESSED_DIR, "y_sleep_edf.npy")

    np.save(X_save_path, X_all)
    np.save(y_save_path, y_all)

    print("\nSleep-EDF preprocessing completed successfully.")
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