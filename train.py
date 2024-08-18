import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import nn
from torchvision.datasets import OxfordIIITPet
from efficientnet.model import EfficientNet
from efficientnet.utils import EarlyStopping
from data.transforms import train_transform, val_transform
from torchinfo import summary

model_name = 'b0'
num_classes = 37
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet(model_name, num_classes).to(device)

# Print model summary
print(summary(model, (1, 3, 224, 224)))

optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_dataset = OxfordIIITPet(root='./data', split='trainval', download=True, transform=train_transform)
val_dataset = OxfordIIITPet(root='./data', split='test', download=True, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, save_path='efficientnet_model.pth', patience=5):
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')

        early_stopping(val_loss, model, save_path=save_path)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(save_path))

# Train the model with early stopping
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, save_path='efficientnet_pet_model.pth', patience=5)
