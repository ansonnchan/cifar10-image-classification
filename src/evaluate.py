import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

#categories the images will be sorted into 
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def evaluate_model(model, loader, device, criterion, results_dir='results'):
    os.makedirs(results_dir, exist_ok=True)
    model.eval()
    all_preds, all_targets = [], []
    
    total_loss, correct, total = 0, 0, 0
    #disable gradient calculation to save memory
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets) #calculate loss to see how confident model is
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    val_loss = total_loss/len(loader)
    val_acc = 100*correct/total
    
    print(f"Evaluation: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_dir,'confusion_matrix.png'))
    plt.close()
    
    # Classification Report
    report = classification_report(all_targets, all_preds, target_names=class_names)
    with open(os.path.join(results_dir,'classification_report.txt'), 'w') as f:
        f.write(report)
    print("\nClassification Report saved to classification_report.txt")
    
    return val_loss, val_acc, all_preds, all_targets

def visualize_misclassified(model, loader, device, results_dir='results', num_images=8):
    import matplotlib.pyplot as plt
    os.makedirs(results_dir, exist_ok=True)
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            #identify misclassified samples 
            for img, pred, true in zip(images, preds, labels):
                if pred != true:
                    misclassified.append((img.cpu(), pred.cpu().item(), true.cpu().item()))
                if len(misclassified) >= num_images:
                    break
            if len(misclassified) >= num_images:
                break
    
    plt.figure(figsize=(15,3))
    for i, (img, pred, true) in enumerate(misclassified):
        plt.subplot(1,num_images,i+1)
        img = img.permute(1,2,0).numpy()*0.5 + 0.5
        plt.imshow(img)
        plt.title(f"T:{class_names[true]}\nP:{class_names[pred]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,'misclassified_samples.png'))
    plt.show()
