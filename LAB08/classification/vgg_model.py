import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# VGG16 layout: 13 conv layers
VGG16_CONFIG = [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]
VGG_SMALL_CONFIG = [(32, 2), (64, 2), (128, 3), (256, 3)]


class VGGNet(nn.Module):
    def __init__(self, input_shape, num_classes, blocks=VGG16_CONFIG):
        super(VGGNet, self).__init__()
        
        in_channels = input_shape[2]  # (H, W, C) -> C
        layers = []
        
        for filters, n_conv in blocks:
            for _ in range(n_conv):
                layers.append(nn.Conv2d(in_channels, filters, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(filters))
                layers.append(nn.ReLU(inplace=True))
                in_channels = filters
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        out_features = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, out_features)
        )
        self.num_classes = num_classes

    def forward(self, x):
        # Scale 0-255 to 0-1 and reshape (B, H, W, C) -> (B, C, H, W)
        x = x.float() / 255.0
        x = x.permute(0, 3, 1, 2)
        
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train_model(X_train, y_train, X_val, y_val, num_classes,
                output_dir=None, epochs=30, batch_size=64, blocks=VGG16_CONFIG):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device} (GPU Working!)" if device.type == 'cuda' else "\nUsing CPU")

    # Convert to Tensors
    train_x = torch.tensor(X_train, dtype=torch.uint8)
    train_y = torch.tensor(y_train, dtype=torch.float32 if num_classes == 2 else torch.long)
    val_x = torch.tensor(X_val, dtype=torch.uint8)
    val_y = torch.tensor(y_val, dtype=torch.float32 if num_classes == 2 else torch.long)

    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = VGGNet(X_train.shape[1:], num_classes, blocks).to(device)
    
    criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            if num_classes == 2:
                loss = criterion(outputs.squeeze(), labels)
                preds = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
            else:
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Validation Phase
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                if num_classes == 2:
                    loss = criterion(outputs.squeeze(), labels)
                    preds = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
                else:
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)

                val_running_loss += loss.item() * inputs.size(0)
                val_correct += (preds == labels.long()).sum().item()
                val_total += labels.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        history["accuracy"].append(epoch_acc)
        history["val_accuracy"].append(val_acc)
        history["loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1:02d}/{epochs:02d} - "
              f"loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} - "
              f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, "vgg_model.pth"))
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump(history, f)

    class HistoryWrapper:
        def __init__(self, h):
            self.history = h

    return model, HistoryWrapper(history)


def predict_model(model, X_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    test_x = torch.tensor(X_test, dtype=torch.uint8)
    test_loader = DataLoader(TensorDataset(test_x), batch_size=64, shuffle=False)

    predictions = []
    with torch.no_grad():
        for (inputs,) in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if model.num_classes == 2:
                preds = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
            else:
                preds = outputs.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())

    return torch.tensor(predictions).numpy()