import time
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from torchvision import datasets
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

# Εφαρμογή PCA
n_components = 128  # Ίσος αριθμός διαστάσεων με Autoencoder
pca = PCA(n_components=n_components)
x_train_pca = pca.fit_transform(train_dataset.data.numpy())
x_test_pca = pca.transform(test_dataset.data.numpy())

# Ανακατασκευή των εικόνων
train_reconstructed = torch.tensor(pca.inverse_transform(x_train_pca)).float()
test_reconstructed = torch.tensor(pca.inverse_transform(x_test_pca)).float()

# Αξιολόγηση για Training και Test Set
train_predictions = evaluate_accuracy(model, train_reconstructed, train_dataset.targets, "Training Set", "PCA_Reconstruction")
predicted = test_predictions = evaluate_accuracy(model, test_reconstructed, test_dataset.targets, "Test Set", "PCA_Reconstruction")

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

plt.savefig("PCA_Reconstruction.png")
print("Το plot αποθηκεύτηκε στο 'PCA_Reconstruction.png'.")


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
print(f"Χρόνος εκτέλεσης προγράμματος: {program_end_time-program_start_time:.2f} δευτερόλεπτα")
