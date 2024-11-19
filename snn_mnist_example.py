
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from snntorch import spikegen
import snntorch as snn
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
num_epochs = 2
learning_rate = 0.001

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define a simple SNN model
class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # Input layer
        self.lif1 = snn.Leaky(beta=0.9)   # Spiking layer
        self.fc2 = nn.Linear(128, 10)    # Output layer
    
    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        # First layer
        x = self.fc1(x)
        x = self.lif1(x)
        # Output layer
        x = self.fc2(x)
        return x

# Initialize the model, loss, and optimizer
model = SNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print("Training the SNN...")
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

# Testing loop
print("Testing the SNN...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Plot a few test samples with predictions
examples = iter(test_loader)
example_data, example_labels = examples.next()
example_data = example_data.to(device)
outputs = model(example_data)
_, predictions = torch.max(outputs, 1)

# Plot images and predictions
fig, axes = plt.subplots(1, 6, figsize=(12, 4))
for i in range(6):
    axes[i].imshow(example_data[i].cpu().squeeze(), cmap="gray")
    axes[i].set_title(f"Predicted: {predictions[i].item()}")
    axes[i].axis('off')
plt.show()
