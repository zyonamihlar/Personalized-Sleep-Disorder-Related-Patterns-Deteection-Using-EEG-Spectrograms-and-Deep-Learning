import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    BATCH_SIZE,
    NUM_CLASSES,
    CLASS_NAMES
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleSleepCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_70_15_15_split(X, y):
    # 70% training, 30% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Split remaining 30% into 15% validation and 15% testing
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(model_path, X_test, y_test):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = SimpleSleepCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    test_ds = EEGDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)

            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    results = {
        "accuracy": accuracy_score(all_true, all_preds),
        "precision": precision_score(all_true, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_true, all_preds, average="weighted", zero_division=0),
        "f1": f1_score(all_true, all_preds, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(all_true, all_preds),
        "classification_report": classification_report(
            all_true,
            all_preds,
            target_names=CLASS_NAMES,
            zero_division=0
        ),
        "predictions": np.array(all_preds),
        "true_labels": np.array(all_true)
    }

    return results


def simple_disorder_logic(pred_labels):
    total_epochs = len(pred_labels)

    if total_epochs == 0:
        return {
            "wake_ratio": 0,
            "rem_ratio": 0,
            "deep_ratio": 0,
            "possible_insomnia_pattern": False,
            "possible_fragmented_sleep_pattern": False
        }

    wake_ratio = np.sum(pred_labels == 0) / total_epochs
    deep_ratio = np.sum(pred_labels == 3) / total_epochs
    rem_ratio = np.sum(pred_labels == 4) / total_epochs

    insomnia_flag = wake_ratio > 0.35
    fragmented_flag = (rem_ratio < 0.10) or (deep_ratio < 0.10)

    return {
        "wake_ratio": wake_ratio,
        "rem_ratio": rem_ratio,
        "deep_ratio": deep_ratio,
        "possible_insomnia_pattern": insomnia_flag,
        "possible_fragmented_sleep_pattern": fragmented_flag
    }


def print_results(title, results):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")
    print(f"F1-score : {results['f1']:.4f}")

    print("\nClassification Report:")
    print(results["classification_report"])

    print("Confusion Matrix:")
    print(results["confusion_matrix"])

    disorder_results = simple_disorder_logic(results["predictions"])

    print("\nPreliminary Disorder Indicators:")
    print(f"Wake ratio: {disorder_results['wake_ratio']:.4f}")
    print(f"REM ratio : {disorder_results['rem_ratio']:.4f}")
    print(f"N3 ratio  : {disorder_results['deep_ratio']:.4f}")
    print(
        "Possible insomnia-related pattern:",
        "Yes" if disorder_results["possible_insomnia_pattern"] else "No"
    )
    print(
        "Possible fragmented sleep pattern:",
        "Yes" if disorder_results["possible_fragmented_sleep_pattern"] else "No"
    )


def main():
    X_path = os.path.join(PROCESSED_DATA_DIR, "combined", "X_combined.npy")
    y_path = os.path.join(PROCESSED_DATA_DIR, "combined", "y_combined.npy")

    if not os.path.exists(X_path):
        raise FileNotFoundError(f"Combined X file not found: {X_path}")

    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Combined y file not found: {y_path}")

    print("Loading combined dataset...")
    X = np.load(X_path)
    y = np.load(y_path)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    X_train, X_val, X_test, y_train, y_val, y_test = create_70_15_15_split(X, y)

    print("\nDataset Split:")
    print(f"Training set   : {X_train.shape[0]} samples")
    print(f"Validation set : {X_val.shape[0]} samples")
    print(f"Testing set    : {X_test.shape[0]} samples")

    generic_model_path = os.path.join(MODEL_DIR, "generic_sleep_cnn.pth")
    personalized_model_path = os.path.join(MODEL_DIR, "personalized_sleep_cnn.pth")

    generic_results = evaluate_model(generic_model_path, X_test, y_test)
    personalized_results = evaluate_model(personalized_model_path, X_test, y_test)

    print_results("Generic CNN Model Results", generic_results)
    print_results("Personalized CNN Model Results", personalized_results)


if __name__ == "__main__":
    main()