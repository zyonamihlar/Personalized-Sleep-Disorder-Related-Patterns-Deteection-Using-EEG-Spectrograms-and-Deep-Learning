import os
import numpy as np
from config import BASE_DIR

SLEEP_EDF_X = os.path.join(BASE_DIR, "data", "processed", "sleep_edf", "X_sleep_edf.npy")
SLEEP_EDF_Y = os.path.join(BASE_DIR, "data", "processed", "sleep_edf", "y_sleep_edf.npy")

ISRUC_X = os.path.join(BASE_DIR, "data", "processed", "isruc", "X_isruc.npy")
ISRUC_Y = os.path.join(BASE_DIR, "data", "processed", "isruc", "y_isruc.npy")

COMBINED_DIR = os.path.join(BASE_DIR, "data", "processed", "combined")
os.makedirs(COMBINED_DIR, exist_ok=True)

X_COMBINED = os.path.join(COMBINED_DIR, "X_combined.npy")
Y_COMBINED = os.path.join(COMBINED_DIR, "y_combined.npy")


def main():
    print("Loading Sleep-EDF...")
    X_sleep = np.load(SLEEP_EDF_X)
    y_sleep = np.load(SLEEP_EDF_Y)

    print("Loading ISRUC...")
    X_isruc = np.load(ISRUC_X)
    y_isruc = np.load(ISRUC_Y)

    print("Sleep-EDF:", X_sleep.shape, y_sleep.shape)
    print("ISRUC:", X_isruc.shape, y_isruc.shape)

    if X_sleep.shape[1:] != X_isruc.shape[1:]:
        raise ValueError(
            f"Shape mismatch: Sleep-EDF {X_sleep.shape[1:]} vs ISRUC {X_isruc.shape[1:]}"
        )

    X_combined = np.concatenate([X_sleep, X_isruc], axis=0)
    y_combined = np.concatenate([y_sleep, y_isruc], axis=0)

    print("Combined:", X_combined.shape, y_combined.shape)

    np.save(X_COMBINED, X_combined)
    np.save(Y_COMBINED, y_combined)

    # Verify files immediately
    X_test = np.load(X_COMBINED)
    y_test = np.load(Y_COMBINED)

    print("Saved and verified successfully.")
    print("Verified X:", X_test.shape)
    print("Verified y:", y_test.shape)

    unique, counts = np.unique(y_test, return_counts=True)
    print("\nClass distribution:")
    for label, count in zip(unique, counts):
        stage = ["Wake", "N1", "N2", "N3", "REM"][int(label)]
        print(f"{label} ({stage}): {count}")


if __name__ == "__main__":
    main()