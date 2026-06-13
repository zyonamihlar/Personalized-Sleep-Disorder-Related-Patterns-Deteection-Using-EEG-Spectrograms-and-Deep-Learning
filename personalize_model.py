import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
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


def evaluate(model, loader):
    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y_batch.cpu().numpy())

    accuracy = accuracy_score(all_true, all_preds)
    precision = precision_score(all_true, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_true, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_true, all_preds, average="weighted", zero_division=0)

    report = classification_report(
        all_true,
        all_preds,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    return accuracy, precision, recall, f1, report


def main():
    # Load combined dataset
    X_path = os.path.join(PROCESSED_DATA_DIR, "combined", "X_combined.npy")
    y_path = os.path.join(PROCESSED_DATA_DIR, "combined", "y_combined.npy")

    X = np.load(X_path)
    y = np.load(y_path)

    print("Loaded combined dataset.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Create personalization subset
    # 10% of the combined dataset is reserved for personalization.
    # From this subset, 70% is used for fine-tuning and 30% is used for testing.
    # This corresponds to approximately 7% adaptation data and 3% personalization test data overall.
    X_remaining, X_personal, y_remaining, y_personal = train_test_split(
        X,
        y,
        test_size=0.10,
        random_state=42,
        stratify=y
    )

    X_personal_train, X_personal_test, y_personal_train, y_personal_test = train_test_split(
        X_personal,
        y_personal,
        test_size=0.30,
        random_state=42,
        stratify=y_personal
    )

    train_ds = EEGDataset(X_personal_train, y_personal_train)
    test_ds = EEGDataset(X_personal_test, y_personal_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Load generic combined model
    model_path = os.path.join(MODEL_DIR, "generic_sleep_cnn.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError("generic_sleep_cnn.pth not found. Please run train_model.py first.")

    model = SimpleSleepCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    print("\nGeneric model loaded.")

    # Evaluate before personalization
    before_acc, before_precision, before_recall, before_f1, before_report = evaluate(model, test_loader)

    print("\nBefore Personalization")
    print(f"Accuracy : {before_acc:.4f}")
    print(f"Precision: {before_precision:.4f}")
    print(f"Recall   : {before_recall:.4f}")
    print(f"F1-score : {before_f1:.4f}")
    print(before_report)

    # Freeze convolutional feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # Train only classifier layers
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE / 10,
        foreach=False
    )

    model.train()

    personalization_epochs = 5

    print("\nStarting personalization fine-tuning...")

    for epoch in range(personalization_epochs):
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Personalization Epoch {epoch + 1}/{personalization_epochs} | Loss: {avg_loss:.4f}")

    # Evaluate after personalization
    after_acc, after_precision, after_recall, after_f1, after_report = evaluate(model, test_loader)

    print("\nAfter Personalization")
    print(f"Accuracy : {after_acc:.4f}")
    print(f"Precision: {after_precision:.4f}")
    print(f"Recall   : {after_recall:.4f}")
    print(f"F1-score : {after_f1:.4f}")
    print(after_report)

    # Save personalized model
    save_path = os.path.join(MODEL_DIR, "personalized_sleep_cnn.pth")
    torch.save(model.state_dict(), save_path)

    print("\nPersonalized model saved.")
    print("Saved to:", save_path)


if __name__ == "__main__":
    main()