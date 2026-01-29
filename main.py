import os
import torch
from src.data import get_cifar10_dataloaders
from src.model import ConvNet
from src.train import train_model
from src.evaluate import evaluate_model, visualize_misclassified
from src.utils import plot_training_curves

#select GPU if available, otherwise fallback to cpu 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results', exist_ok=True)

#hyperparameters
BATCH_SIZE = 64 # Number of images processed before the model updates its weights
LR = 0.001 #learning rate 
EPOCHS = 20 #number of times the model sees the entire dataset 


train_loader, val_loader, test_loader = get_cifar10_dataloaders(batch_size=BATCH_SIZE)

model = ConvNet().to(device)

#train
train_losses, train_accs, val_losses, val_accs = train_model(
    model, train_loader, val_loader, device,
    epochs=EPOCHS, lr=LR
)

#training curves 
plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path='results/training_curves.png')

#load best model and evaluate 
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
criterion = torch.nn.CrossEntropyLoss()
test_loss, test_acc, all_preds, all_targets = evaluate_model(model, test_loader, device, criterion)

# Save confusion matrix and misclassified samples
visualize_misclassified(model, test_loader, device, results_dir='results')
