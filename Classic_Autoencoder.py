import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
import matplotlib.pyplot as plt
from Classifier import evaluate_accuracy, model
model.load_state_dict(torch.load("classifier.pth", weights_only=True))


program_start_time = time.time()

# Φόρτωση MNIST Dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, download=True)

# Τυποποίηση δεδομένων
scaler = MinMaxScaler(feature_range=(0, 1))
train_dataset.data = torch.tensor(scaler.fit_transform(train_dataset.data.view(-1, 28*28))).float()
test_dataset.data = torch.tensor(scaler.transform(test_dataset.data.view(-1, 28*28))).float()

# Δημιουργία DataLoader
train_dataset_tensor = TensorDataset(train_dataset.data, train_dataset.targets)
train_loader = DataLoader(train_dataset_tensor, batch_size=256, shuffle=True)


# Ορισμός Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.flat = nn.Flatten()

        # Encoder
        self.enc = nn.Linear(784, 128)

        # Decoder
        self.dec = nn.Linear(128, 784)

    def forward(self, x):
        x = self.flat(x)
        x = F.relu(self.enc(x))
        recon = F.sigmoid(self.dec(x))
        return recon


# Εκπαίδευση
model_auto = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model_auto.parameters(), lr=0.001)

# Εκπαίδευση του Autoencoder
train_start = time.time()
epochs = 1000
patience = 10
best_loss = 10000
counter = 0

for epoch in range(epochs):
    model_auto.train()
    total_loss = 0.0
    for images, labels in train_loader:
        # Forward Pass
        reconstructed = model_auto(images)

        # Υπολογισμός Loss
        loss = criterion(reconstructed, images)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Εκτύπωση Loss
    if (epoch + 1) <= 20 or (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.8f}")

    # Early Stopping Check
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


# Ανακατασκευή των Εικόνων
model_auto.eval()
with torch.no_grad():
  train_reconstructed = model_auto(train_dataset.data)
  test_reconstructed = model_auto(test_dataset.data)


# Αξιολόγηση Ανακατασκευασμένων Εικόνων με τον Classifier
evaluate_accuracy(model, train_reconstructed, train_dataset.targets, "Training Set", "Classic_Autoencoder")
predicted = evaluate_accuracy(model, test_reconstructed, test_dataset.targets, "Test Set", "Classic_Autoencoder")

print("\n")

incorrect_indices = []
correct_indices = []
for i in range(len(test_dataset.data)):
    if test_dataset.targets[i] != predicted[i]:  # Αν δεν ταιριάζουν
        incorrect_indices.append(i)
    else:
        correct_indices.append(i)

# Εμφάνιση Πρωτότυπων και Ανακατασκευασμένων Εικόνων από το Test Set
plt.figure(figsize=(15, 12))  # Αυξήστε το ύψος για 4 σειρές

# Πρώτη γραμμή: Πρωτότυπες εικόνες
for i in range(10):
    plt.subplot(4, 10, i + 1)  # 4 σειρές, 10 στήλες
    plt.imshow(test_dataset.data[correct_indices[i]].reshape(28, 28), cmap='gray')
    plt.axis('off')

# Δεύτερη γραμμή: Ανακατασκευασμένες εικόνες
for i in range(10):
    plt.subplot(4, 10, i + 11)
    plt.imshow(test_reconstructed[correct_indices[i]].numpy().reshape(28, 28), cmap='gray')
    plt.axis('off')

# Τρίτη γραμμή: Πρωτότυπες εικόνες
for i in range(10):
    plt.subplot(4, 10, i + 21)
    plt.imshow(test_dataset.data[incorrect_indices[i]].numpy().reshape(28, 28), cmap='gray')
    plt.axis('off')

# Τέταρτη γραμμή: Ανακατασκευασμένες εικόνες
for i in range(10):
    plt.subplot(4, 10, i + 31)
    plt.imshow(test_reconstructed[incorrect_indices[i]].numpy().reshape(28, 28), cmap='gray')
    plt.axis('off')

# Προσθήκη τίτλων
plt.gcf().text(0.5, 0.92, "Original Test Images ", ha="center", fontsize=16)
plt.gcf().text(0.5, 0.68, "Correct Predicted Images ", ha="center", fontsize=16)
plt.gcf().text(0.5, 0.5, "Original Test Images ", ha="center", fontsize=16)
plt.gcf().text(0.5, 0.3, " False Predicted Images ", ha="center", fontsize=16)

plt.savefig("Classic_Autoencoder.png")
print("Το plot αποθηκεύτηκε στο 'Classic_Autoencoder.png'.")


# Υπολογισμός ακρίβειας ανά κατηγορία
def accuracy_per_category(test_labels, predicted):
    class_correct = []
    class_total = []

    for i in range(10):
        class_correct.append(0)
        class_total.append(0)

    for i in range(len(test_labels)):
        label = test_labels[i].item()
        class_total[label] += 1
        if predicted[i].item() == label:
            class_correct[label] += 1

    for i in range(10):
        if class_total[i] > 0:
          accuracy = 100 * class_correct[i] / class_total[i]
        else:
            accuracy = 0
        if i == 0:
            print(f"Σωστές Ανασκευασμένες Εικόνες 9 -> 0 | Σωστά: {class_correct[i]:<3} / {class_total[i]:<3} | Ακρίβεια: {accuracy:.2f}%")
        else:
            print(f"Σωστές Ανασκευασμένες Εικόνες {i - 1} -> {i} | Σωστά: {class_correct[i]:<3} / {class_total[i]:<3} | Ακρίβεια: {accuracy:.2f}%")

accuracy_per_category(test_dataset.targets, predicted)

print("\n")

program_end_time = time.time()
print(f"Χρόνος εκτέλεσης προγράμματος: {(program_end_time-program_start_time)/60:.2f} λεπτά")
