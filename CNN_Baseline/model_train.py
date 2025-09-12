import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import make_dataset
import logging
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import torch.nn.functional as F
import seaborn as sns
import argparse
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
import mynet_3
import time


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
file_handler = logging.FileHandler('./output/training.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)


def mobilenet(train_loader, validation_loader, resize=False):
    # Basic configurations
    num_epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on: ", str(device))
    # Define model, loss function and optimizer
    model = mynet_3.MyModel()
    model_name = 'mynet_3'
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-7)

    # Initialize tracking param. 
    best_val_loss = float("inf")
    best_model_weights = None
    early_stop_patience = 10  # opt out of training with 5 steps no better. 
    patience_counter = 0

    # Resource evaluation
    torch.cuda.reset_peak_memory_stats(device)
    total_start_time = time.time()
    train_time_per_epoch = []
    
    # Train model
    train_losses = []
    val_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        batch_10_loss = 0.0
        for batch_idx, (image0s, image1s, labels) in enumerate(train_loader):
            image0s = image0s.float().to(device)
            image1s = image1s.float().to(device)
            labels = torch.tensor([label.to(device) for label in labels], dtype=torch.int64)  # Labels should start from 0. \
            labels = labels.to(device)
            output = model(image0s, image1s)
            loss = criterion(output, labels)
            # update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * image0s.size(0)  # Multiplies batch size. 

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        epoch_end = time.time()
        train_time_per_epoch.append(epoch_end - epoch_start)

        # Evaluate model
        model.eval()
        validation_loss = 0.0
        validation_accuracy = 0.0
        sum_precision = 0
        num_precision = 0
        preds_list = []
        labels_list = []
        for image0s, image1s, labels in validation_loader:
            if resize:
                image0s = F.interpolate(image0s, size=(224, 224), mode='bilinear', align_corners=False)
                image1s = F.interpolate(image1s, size=(224, 224), mode='bilinear', align_corners=False)

            image0s = image0s.float().to(device)
            image1s = image1s.float().to(device)
            labels = torch.tensor([label.to(device) for label in labels], dtype=torch.int64)  # Labels should start from 0. \
            labels = labels.to(device)
            output = model(image0s, image1s)
            loss = criterion(output, labels)
            # calculate accuracy
            preds = torch.argmax(output, dim=1)
            epoch_accuracy = pred_accuracy(preds, labels)
            # Append preds and labels matrix. 
            preds_list.extend(preds.cpu())
            labels_list.extend(labels.cpu())
            # total_loss = sum(losses)
            validation_loss += loss.item() * image0s.size(0)
            validation_accuracy += epoch_accuracy

        val_loss = validation_loss / len(validation_loader.dataset)
        val_accuracy = validation_accuracy / len(validation_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Save improved model. 
        if val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.4f} --> {val_loss:.4f}). Saving model...")
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy() 
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss,
            }, f"./checkpoint/{model_name}.pth") 
            patience_counter = 0  
            save_cm(epoch, labels_list, preds_list)
        else:
            patience_counter += 1

        # early stop
        if patience_counter >= early_stop_patience or epoch==1:
            print("Early stopping triggered!")
            total_training_time = time.time() - total_start_time
            peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # 转MB
            print("\n=== Training Summary ===")
            print(f"Total training time: {total_training_time:.2f} sec")
            print(f"Average time per epoch: {np.mean(train_time_per_epoch):.2f} sec")
            print(f"Peak GPU memory usage: {peak_memory:.2f} MB")
            break
        
        # Save figure at the end of each epoch
        fig, ax1 = plt.subplots()
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax2 = ax1.twinx()
        ax2.plot(val_accuracies, 'g-', label='Val Accu')
        ax2.set_ylabel('Accu')
        fig.legend()
        plt.title('Training and Validation Loss')
        plt.savefig(f'./output/lossplot_julytest.png')
        plt.close()
        # Update logging.
        print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_accuracy:.4f}")

    # Save best model weights. 
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), f"./checkpoint/final_{model_name}_weights.pth")


def save_cm(epoch, labels_list, preds_list):
    sklearn_cm = confusion_matrix(labels_list, preds_list)
    sklearn_cm = sklearn_cm[0:3, 0:3]
    # normalization
    row_sums = sklearn_cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 
    sklearn_cm_norm = sklearn_cm.astype('float') / row_sums
    # Display labels. 
    sklearn_disp = ConfusionMatrixDisplay(
        confusion_matrix=sklearn_cm_norm,
        display_labels=['nue', 'numu', 'nc']  
    )
    # Generate fig. 
    fig, ax = plt.subplots(figsize=(8, 6))
    sklearn_disp.plot(
        cmap=plt.cm.Blues,
        ax=ax,
        values_format='.2f',
        colorbar=False  #
    )
    # Adjust color and layout. 
    plt.colorbar(sklearn_disp.im_, ax=ax, fraction=0.046, pad=0.04)
    plt.title(f'Normalized Confusion Matrix (epoch={epoch+1})')
    # Adjust labels. 
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    # Close and save. 
    plt.tight_layout()
    plt.savefig(f"./output/cm_apr23.png", dpi=300, bbox_inches='tight')
    plt.close()


def pred_accuracy(preds, label):
    total_score = 0
    for i, pred in enumerate(preds):
        if pred == label[i]:
            total_score += 1
        else: 
            total_score += 0

    return total_score / len(preds)


def save_image(image):
    # Print images and labels. 
    def denormalize(tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor
    singleImage = image
    singleImage = transforms.ToPILImage()(singleImage)
    print(type(singleImage))
    singleImage.save("./output/trainImage.png")


def confidence_score(train_loader, validation_loader):
    # Basic configurations
    num_epochs = 600
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on: ", str(device))
    # Define model, loss function and optimizer
    resnet_weights = ResNet18_Weights.DEFAULT
    model = DualImageResNet18Gray(num_classes=3)
    model = torch.load('./output/resnet_model_1204_cropped.pt')
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    probabilities = []
    true_labels = []

    with torch.no_grad():
        for image0s, image1s, labels in validation_loader:
            image0s = image0s.float().to(device)
            image1s = image1s.float().to(device)
            labels = labels.to(device)
            output = model(image0s, image1s)
            
            # Calculate the probability for each class
            probs = F.softmax(output, dim=1)
            probabilities.extend(probs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    probabilities = np.array(probabilities)
    print(probabilities.shape)
    true_labels = np.array(true_labels)
    
    # Plot and save confidence scores for each class
    class_name = ['nuecc', 'numucc', 'nc']
    for i in range(3):  # Assuming three classes
        plt.figure(figsize=(8, 6))
        # Plot KDE for each class's probability
        for j in range(3):  # Three classes
            sns.kdeplot(probabilities[true_labels == i, j], label=f'{class_name[j]} Probability', fill=True)
        
        plt.title(f'Confidence Score Distributions for True {class_name[i]}')
        plt.xlabel('Confidence Score')
        plt.ylabel('Density')
        plt.legend()
        # Save the plot as a file
        plt.savefig(f'confidence_scores_{class_name[i]}_kde.png')
        plt.close()


def separate_train(args):
    train_loader, validation_loader = make_dataset.classifier_dataloader_cropped2(1, True)
    mobilenet(train_loader, validation_loader, resize=args.resize) 

def main():
    parser = argparse.ArgumentParser(description="Execute different functions based on input parameters.")
    parser.add_argument('function', type=str, help='The function to execute: one, two, or three')
    parser.add_argument('--resize', type=bool, default=False, help='Resize images to 256x256 for MobileNet training')
    args = parser.parse_args()
    if args.function == 'separate_train':
        separate_train(args)
    else:
        print("Invalid function name.")


if __name__ == '__main__':
    main()