import torch
from torch.utils.data import random_split, DataLoader, ConcatDataset
from torchvision import transforms, datasets

import os

'''
def augment_data(data_dir: str):
    
    # Define data transformations for data augmentation and normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Create data loaders
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}

    return image_datasets
'''


'''
def augment_data(data_dir: str):
    
    # Define data transformations for data augmentation and normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),         # Randomly flip images for augmentation
            transforms.RandomCrop(32, padding=4),     # Random crop for additional augmentation
            transforms.ToTensor(),                     # Convert image to tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with CIFAR-10 mean & std
        ]),

        'test': transforms.Compose([
            transforms.ToTensor(),                     # Convert image to tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with CIFAR-10 mean & std
        ])
    }

    # Create data loaders
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}

    return image_datasets
'''

def augment_data(data_dir: str):
    """
    Augments CIFAR-10 data with improved transformations for generalization.
    
    Args:
        data_dir (str): Path to the dataset directory containing 'train' and 'test' folders.
        
    Returns:
        dict: A dictionary with training and testing datasets.
    """

    # Define data transformations for training and testing
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),         # Random horizontal flip
            transforms.RandomCrop(32, padding=4),     # Random crop with padding
            transforms.RandomRotation(15),            # Random rotation up to 15 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
            #transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),  # Randomly erase patches
            transforms.ToTensor(),                     # Convert image to tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with CIFAR-10 mean & std
        ]),

        'test': transforms.Compose([
            transforms.ToTensor(),                     # Convert image to tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with CIFAR-10 mean & std
        ])
    }

    # Create datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}

    return image_datasets

def prepare_dataset(num_partitions: int,
                    num_partitions_pos: int = None, 
                    data_dir: str = None,
                    data_dir_pos: str = None,
                    batch_size: int=16,
                    val_ratio: float = 0.1):

    rng = torch.Generator().manual_seed(49)

    image_datasets = augment_data(data_dir)
    num_images = len(image_datasets['train'])
    base_partition_size = num_images // num_partitions

    partition_len = [base_partition_size] * num_partitions

    # Calculate remainder and distribute it across partitions to match the exact number of images
    remainder = num_images % num_partitions
    for i in range(remainder):
        partition_len[i] += 1

    trainsets = random_split(image_datasets['train'], partition_len, rng)


    #Create dataloaders with train+val support
    trainloaders = []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        #num_val = int(val_ratio * num_total)
        num_val = max(1, int(val_ratio * num_total))
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], rng)

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))



    if num_partitions_pos and data_dir_pos:
        image_datasets_pos = augment_data(data_dir_pos)
        num_images_pos = len(image_datasets_pos['train'])
        base_partition_size_pos = num_images_pos // num_partitions_pos

        partition_len_pos = [base_partition_size_pos] * num_partitions_pos

        # Calculate remainder and distribute it across partitions to match the exact number of images
        remainder_pos = num_images_pos % num_partitions_pos
        for i in range(remainder_pos):
            partition_len_pos[i] += 1
        
        trainsets_pos = random_split(image_datasets_pos['train'], partition_len_pos, rng)

        for trainset_ in trainsets_pos:
            num_total = len(trainset_)
            #num_val = int(val_ratio * num_total)
            num_val = max(1, int(val_ratio * num_total))
            num_train = num_total - num_val

            for_train, for_val = random_split(trainset_, [num_train, num_val], rng)

            trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
            valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

        combined_test_dataset = ConcatDataset([image_datasets['test'], image_datasets_pos['test']])
        testloader = DataLoader(combined_test_dataset, batch_size=batch_size)

    else:
        testloader = DataLoader(image_datasets['test'], batch_size=batch_size)


    return trainloaders, valloaders, testloader

'''
def prepare_dataset(num_partitions: int,
                         data_dir: str,
                         batch_size: int=16,
                         val_ratio: float = 0.1):

    image_datasets = augment_data(data_dir)
    num_images = len(image_datasets['train'])
    base_partition_size = num_images // num_partitions

    partition_len = [base_partition_size] * num_partitions

    # Calculate remainder and distribute it across partitions to match the exact number of images
    remainder = num_images % num_partitions
    for i in range(remainder):
        partition_len[i] += 1

    # Split the dataset using random_split with the specified partition lengths
    trainsets = random_split(image_datasets['train'], partition_len, torch.Generator().manual_seed(49))

    #Create dataloaders with train+val support
    trainloaders = []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(49))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

    testloader = DataLoader(image_datasets['test'], batch_size=batch_size)

    return trainloaders, valloaders, testloader
'''