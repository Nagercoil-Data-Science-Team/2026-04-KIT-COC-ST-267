import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CraftDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the craft workshop images.
    It recursively traverses the main categories (workshop, artifacts, process, textures)
    and fetches all downloaded images.
    """
    def __init__(self, root_dir="dataset", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        self.classes = []
        
        # Check if dataset exists
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory '{root_dir}' not found. Please run download_dataset.py first.")
            
        # Traverse the main directories
        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.classes.append(class_name)
                class_idx = len(self.classes) - 1
                
                # Recursively find all images in this class directory (handling bing-image-downloader nested structure)
                for root, _, files in os.walk(class_path):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                            self.image_paths.append(os.path.join(root, file))
                            self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image and convert to RGB (some downloaded images might be Grayscale or RGBA)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # If an image is corrupted, just return a black image or raise
            # For simplicity, returning a zero tensor of expected size if transform is present
            print(f"Warning: Failed to load image {img_path}: {e}")
            image = Image.new('RGB', (256, 256), color='black')
            
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloader(data_dir="dataset", batch_size=32, image_size=(256, 256), apply_augmentation=True):
    """
    Creates and returns a DataLoader ready for GAN training with data augmentation.
    """
    transform_list = []
    
    # 1. Image Resizing: Resize to consistent size and crop
    transform_list.extend([
        transforms.Resize((image_size[0], image_size[1])),
        transforms.CenterCrop(image_size),
    ])
    
    # 3. Data Augmentation: Expand dataset and simulate different conditions
    if apply_augmentation:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),                    # Random flips
            transforms.RandomRotation(degrees=15),                     # Random rotations
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translations
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1) # Brightness, contrast, colors
        ])

    # 2. Normalization: Convert to Tensor and scale to [-1, 1] range
    transform_list.extend([
        transforms.ToTensor(),                                 # Convert to [0.0, 1.0]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Scale to [-1.0, 1.0]
    ])

    transform = transforms.Compose(transform_list)

    dataset = CraftDataset(root_dir=data_dir, transform=transform)
    
    print(f"Loaded {len(dataset)} images from {len(dataset.classes)} categories.")
    print(f"Categories discovered: {dataset.classes}")
    
    # Create dataloader
    # Note: drop_last=True is useful for GANs to keep batch sizes consistent during training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    return dataloader, dataset

def plot_sample_batch(dataloader):
    """
    A helper utility to visualize a batch of images from the dataloader.
    """
    dataiter = iter(dataloader)
    try:
        images, labels = next(dataiter)
    except StopIteration:
        print("DataLoader is empty.")
        return
        
    # Unnormalize images back to [0, 1] range for plotting
    images = images / 2 + 0.5
    
    batch_size = images.size(0)
    cols = 4
    rows = int(np.ceil(batch_size / cols))
    
    fig, axes = plt.subplots(figsize=(12, 3 * rows), nrows=rows, ncols=cols)
    axes = axes.flatten()
    
    dataset_classes = dataloader.dataset.classes
    
    for i in range(batch_size):
        ax = axes[i]
        npimg = images[i].numpy()
        # PyTorch tensors are [C, H, W], matplotlib wants [H, W, C]
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        class_name = dataset_classes[labels[i].item()]
        ax.set_title(f"{class_name} ({labels[i].item()})")
        ax.axis('off')
        
    # Turn off unused subplots
    for j in range(batch_size, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.show()

def visualize_preprocessing_comparison(data_dir="dataset", num_samples=2, image_size=(256, 256)):
    """
    Shows a side-by-side comparison of original vs preprocessed images
    for a few samples in each category, and saves the comparison figures.
    """
    import random
    
    transform = transforms.Compose([
        transforms.Resize((image_size[0], image_size[1])),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),                    
        transforms.RandomRotation(degrees=15),                     
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),                                 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found!")
        return

    categories = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    for category in categories:
        category_dir = os.path.join(data_dir, category)
        all_imgs = []
        for root, _, files in os.walk(category_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    all_imgs.append(os.path.join(root, file))
                    
        if not all_imgs:
            continue
            
        # Select exactly exactly `num_samples` (or however many exist if less)
        samples = random.sample(all_imgs, min(num_samples, len(all_imgs)))
        
        fig, axes = plt.subplots(len(samples), 2, figsize=(10, 5 * len(samples)))
        fig.suptitle(f"Category: {category.upper()} - Original vs Preprocessed", fontsize=16)
        
        if len(samples) == 1:
            axes = np.array([axes])

        for idx, img_path in enumerate(samples):
            try:
                orig_img = Image.open(img_path).convert('RGB')
                
                # Plot Original
                axes[idx, 0].imshow(orig_img)
                axes[idx, 0].set_title(f"Original\nSize: {orig_img.size}")
                axes[idx, 0].axis('off')

                # Apply Transformation
                preprocessed_tensor = transform(orig_img)
                
                # Unnormalize [-1, 1] -> [0, 1] for displaying
                display_tensor = preprocessed_tensor / 2 + 0.5
                display_img = np.transpose(display_tensor.numpy(), (1, 2, 0))
                
                # Plot Preprocessed
                axes[idx, 1].imshow(display_img)
                axes[idx, 1].set_title(f"Preprocessed (Expanded + {image_size})\nSize: (256, 256)")
                axes[idx, 1].axis('off')
                
            except Exception as e:
                print(f"Error visualizing {img_path}: {e}")

        plt.tight_layout()
        save_path = f"preprocessing_comparison_{category}.png"
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved visualization for category '{category}' at {save_path}")
        plt.close(fig) # close to free memory

if __name__ == "__main__":
    print("-" * 50)
    print("Testing data loader for Craft Virtual Reality GAN")
    print("-" * 50)
    
    try:
        dataloader, dataset = get_dataloader(data_dir="dataset", batch_size=16)
        
        if len(dataset) > 0:
            for images, labels in dataloader:
                print(f"Batch Tensor shape: {images.shape} -> (Batch Size, Channels, Height, Width)")
                print(f"Batch Labels: {labels}")
                print(f"Label Mapping: {{idx: name for idx, name in enumerate(dataset.classes)}}")
                break
                
            print("\nGenerating sample plot...")
            # We wrap this in try-except in case there is no GUI environment available in the terminal
            plot_sample_batch(dataloader)
            print("Sample plot generated!")
            
            print("\nGenerating Original vs Preprocessing Comparisons...")
            visualize_preprocessing_comparison(data_dir="dataset", num_samples=2)
            print("Comparisons successfully saved to disk as images!")
        else:
            print("No images found. Ensure the dataset structure is correct.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
