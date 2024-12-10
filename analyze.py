import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

from model import Net



def load_model(num_classes, weights_path, test_dir):

    my_model = Net(num_classes)

    model_param = torch.load(weights_path)
    my_model.load_state_dict(model_param['model_state_dict'])
    my_model.eval()  # Set the model to evaluation mode

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # Adjust to your model's input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example for ImageNet models
    ])

    # Load the test dataset
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Verify the dataset is loaded correctly
    try:
        test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
        print(f"Number of test samples: {len(test_dataset)}")
        print(f"Class labels: {test_dataset.classes}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        test_dataset = None

    return test_dataset, test_loader, my_model



def load_model_32(num_classes, weights_path, test_dir):

    my_model = Net(num_classes)

    model_param = torch.load(weights_path, map_location=torch.device('cpu'))
    my_model.load_state_dict(model_param['model_state_dict'])
    my_model.eval()  # Set the model to evaluation mode

    # Define the transformation
    transform = transforms.Compose([
            transforms.ToTensor(),                     # Convert image to tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with CIFAR-10 mean & std
        ])

    # Load the test dataset
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Verify the dataset is loaded correctly
    try:
        test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
        print(f"Number of test samples: {len(test_dataset)}")
        print(f"Class labels: {test_dataset.classes}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        test_dataset = None

    return test_dataset, test_loader, my_model




def plot_class_distributions(image_datasets):
    """
    Plots the class distributions for training and testing datasets.
    
    Args:
        image_datasets (dict): A dictionary containing training and testing datasets.
    """
    class_names = image_datasets['train'].classes  # Get class names from the training dataset
    
    # Get class counts for training and testing datasets
    train_class_counts = Counter([label for _, label in image_datasets['train'].samples])
    test_class_counts = Counter([label for _, label in image_datasets['test'].samples])
    
    # Map class indices to counts for train and test
    train_counts = [train_class_counts[i] for i in range(len(class_names))]
    test_counts = [test_class_counts[i] for i in range(len(class_names))]
    
    # Plotting the distributions
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Training set distribution
    axs[0].bar(class_names, train_counts, color='skyblue')
    axs[0].set_title('Training Set Class Distribution')
    axs[0].set_xlabel('Classes')
    axs[0].set_ylabel('Count')
    axs[0].set_xticklabels(class_names, rotation=45)
    
    # Testing set distribution
    axs[1].bar(class_names, test_counts, color='orange')
    axs[1].set_title('Testing Set Class Distribution')
    axs[1].set_xlabel('Classes')
    axs[1].set_ylabel('Count')
    axs[1].set_xticklabels(class_names, rotation=45)
    
    plt.tight_layout()
    plt.show()



def display_predictions(model, dataset, mean, std, num_images=20, grid_size=(5, 4)):
    """
    Display a grid of images with their true and predicted labels.

    Args:
        model: Trained PyTorch model for predictions.
        dataset: Dataset to sample images from.
        mean: Tensor of mean values for unnormalization.
        std: Tensor of standard deviation values for unnormalization.
        num_images: Number of images to display.
        grid_size: Tuple indicating grid dimensions (rows, columns).
    """
    # Randomly select images
    indices = np.random.choice(len(dataset), num_images, replace=False)

    # Set up the plot
    fig, axes = plt.subplots(*grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for idx, ax in zip(indices, axes):
        img, true_label = dataset[idx]
        img_batch = img.unsqueeze(0)  # Add batch dimension

        # Predict label
        with torch.no_grad():
            output = model(img_batch)
            pred_label = torch.argmax(output, dim=1).item()

        # Unnormalize the image
        img = img * std[:, None, None] + mean[:, None, None]
        img = torch.clamp(img, 0, 1)

        # Display image
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.axis('off')

        # Labeling
        true_label_name = dataset.classes[true_label]
        pred_label_name = dataset.classes[pred_label]
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f"True: {true_label_name}\nPred: {pred_label_name}", fontsize=10, color=color)

    plt.tight_layout()
    plt.show()


def analyze_class_performance(model, dataset, class_name, mean, std, num_images_to_display=20):
    """
    Analyze the performance of a classification model on a specific class in a dataset.
    
    Args:
        model (torch.nn.Module): The trained model.
        dataset (torch.utils.data.Dataset): The dataset to analyze.
        class_name (str): The name of the class to analyze.
        mean (torch.Tensor): Mean values for unnormalization.
        std (torch.Tensor): Standard deviation values for unnormalization.
        num_images_to_display (int): Number of images to display for visualization.
    """
    if class_name not in dataset.classes:
        print(f"Class '{class_name}' not found.")
        return

    class_idx = dataset.classes.index(class_name)
    selected_indices = [idx for idx, (_, label) in enumerate(dataset) if label == class_idx]

    if not selected_indices:
        print(f"No images found for class: {class_name}")
        return

    # Initialize metrics
    correct_predictions = 0
    misclassifications = {cls: 0 for cls in dataset.classes}
    confusion_matrix = np.zeros((len(dataset.classes), len(dataset.classes)), dtype=int)

    # Analyze predictions
    for idx in selected_indices:
        img, true_label = dataset[idx]
        img = img.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(img)
            _, pred_label = torch.max(output, 1)
        
        # Update metrics
        if true_label == pred_label:
            correct_predictions += 1
        else:
            misclassifications[dataset.classes[pred_label.item()]] += 1
        
        confusion_matrix[true_label, pred_label] += 1

    # Calculate accuracy
    total_images = len(selected_indices)
    accuracy = correct_predictions / total_images * 100
    print(f"Accuracy for class '{class_name}': {accuracy:.2f}%")

    # Visualization of predictions
    random_indices = np.random.choice(selected_indices, min(num_images_to_display, total_images), replace=False)
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()

    for idx, ax in zip(random_indices, axes):
        img, true_label = dataset[idx]
        img = img.unsqueeze(0)
        
        with torch.no_grad():
            output = model(img)
            _, pred_label = torch.max(output, 1)

        img = img.squeeze(0) * std[:, None, None] + mean[:, None, None]
        img = torch.clamp(img, 0, 1)
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.axis('off')

        true_label_name = dataset.classes[true_label]
        pred_label_name = dataset.classes[pred_label.item()]
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f"True: {true_label_name}\nPred: {pred_label_name}", fontsize=10, color=color)

    plt.tight_layout()
    plt.show()

    # Plot misclassification counts
    plt.figure(figsize=(10, 6))
    plt.bar(misclassifications.keys(), misclassifications.values(), color='orange')
    plt.bar(class_name, correct_predictions, color='green')
    plt.title(f"Misclassifications and Correct Predictions for '{class_name}'", fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=dataset.classes, yticklabels=dataset.classes)
    plt.title(f"Confusion Matrix for '{class_name}'", fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.tight_layout()
    plt.show()

    print(f"Correct predictions: {confusion_matrix[class_idx, class_idx]}")
    for i, count in enumerate(confusion_matrix[class_idx]):
        if i != class_idx and count > 0:
            print(f"Misclassified as '{dataset.classes[i]}': {count}")