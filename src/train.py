import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os

def train_model(model, train_loader, val_loader, device, epochs=20, lr=0.001, checkpoint_dir='checkpoints'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Use AMP only if GPU
    scaler = GradScaler() if device.type == 'cuda' else None

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_acc = 0
    early_stop_counter = 0
    max_patience = 5

    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for data, targets in loop:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()

            if device.type == 'cuda':
                # Mixed precision for GPU
                with autocast():
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Normal precision for CPU
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        # Epoch stats
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(avg_train_loss)
        train_accs.append(train_accuracy)

        # Validation
        val_loss, val_accuracy = validate(model, val_loader, device, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={train_accuracy:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.2f}%")

        # Checkpoint & early stopping
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            early_stop_counter = 0
            print("Saved new best model!")
        else:
            early_stop_counter += 1
            if early_stop_counter >= max_patience:
                print("Early stopping triggered.")
                break

        scheduler.step()

    return train_losses, train_accs, val_losses, val_accs

def validate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    return total_loss / len(loader), 100 * correct / total
