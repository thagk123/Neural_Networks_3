import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Classifier()

# 5. Συνάρτηση Αξιολόγησης με Accuracy και Confusion Matrix
def evaluate_accuracy(model, data, targets, set_name, py_name):
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
    accuracy = 100 * correct / len(targets)
    print(f"\n{set_name} Accuracy: {accuracy:.2f}%")
    cm = confusion_matrix(targets.numpy(), predicted.numpy())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap="Blues", xticks_rotation="vertical")
    plt.title(f"{py_name} {set_name} Confusion Matrix")
    plt.savefig(f"{py_name}_{set_name}_Confusion Matrix.png")
    return predicted

if __name__ == "__main__":
    program_start_time = time.time()

    # 1. Φόρτωση MNIST Dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True)

    # Τυποποίηση δεδομένων
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_dataset.data = torch.tensor(scaler.fit_transform(train_dataset.data.view(-1, 28 * 28))).float()
    test_dataset.data = torch.tensor(scaler.transform(test_dataset.data.view(-1, 28 * 28))).float()

    # Δημιουργία DataLoaders
    train_dataset_tensor = TensorDataset(train_dataset.data, train_dataset.targets)
    train_loader = DataLoader(train_dataset_tensor, batch_size=256, shuffle=True)



    # Εκπαίδευση
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Early Stopping parameters
    patience = 10
    best_loss = 10000
    counter = 0


    # Κώδικας εκπαίδευσης
    train_start = time.time()
    epochs = 1000
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            predictions = model(images)
            loss = criterion(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) <= 20 or (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.8f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    train_end = time.time() - train_start
    print(f"Χρόνος εκπαίδευσης: {train_end/60:.2f} λεπτά")

    # Στο Classifier.py, μετά την εκπαίδευση:
    torch.save(model.state_dict(), "classifier.pth")

    # Αξιολόγηση για Training και Test Set
    train_predictions = evaluate_accuracy(model, train_dataset.data, train_dataset.targets, "Training Set", "Classifier")
    test_predictions = evaluate_accuracy(model, test_dataset.data, test_dataset.targets, "Test Set", "Classifier")