import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

from config import (
    PROCESSED_DATA_DIR, MODEL_DIR,
    BATCH_SIZE, LEARNING_RATE, EPOCHS, NUM_CLASSES, CLASS_NAMES
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


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader):
    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    acc = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds, average="weighted", zero_division=0)
    report = classification_report(
        all_true,
        all_preds,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    return acc, f1, report


def main():
    X = np.load(os.path.join(PROCESSED_DATA_DIR, "combined", "X_combined.npy"))
    y = np.load(os.path.join(PROCESSED_DATA_DIR, "combined", "y_combined.npy"))

    print("Loaded combined dataset")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # 70:15:15 split
    # First split: 70% training, 30% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Second split: temporary 30% into 15% validation and 15% testing
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print("\nDataset split:")
    print(f"Training set   : {X_train.shape[0]} samples")
    print(f"Validation set : {X_val.shape[0]} samples")
    print(f"Testing set    : {X_test.shape[0]} samples")

    train_ds = EEGDataset(X_train, y_train)
    val_ds = EEGDataset(X_val, y_val)
    test_ds = EEGDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleSleepCNN().to(DEVICE)

    # Class-weighted loss to reduce class imbalance effect
    class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, foreach=False)

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, criterion, optimizer)

        val_acc, val_f1, _ = evaluate(model, val_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

    test_acc, test_f1, test_report = evaluate(model, test_loader)

    print("\nFinal Test Results")
    print("Final Test Accuracy:", test_acc)
    print("Final Weighted F1:", test_f1)
    print("\nClassification Report:\n", test_report)

    save_path = os.path.join(MODEL_DIR, "generic_sleep_cnn.pth")
    torch.save(model.state_dict(), save_path)

    print("Model saved.")
    print("Saved to:", save_path)


if __name__ == "__main__":
    main()