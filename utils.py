import torch
import numpy as np
import tensorflow_datasets as tfds
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

class MalariaDataset(Dataset):
    def __init__(self, tfds_split, transform=None):
        self.data = list(tfds.as_numpy(tfds_split))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        return image, label

def get_dataloaders(config):
    print("Downloading subset of data via TFDS...")
    # Uzimamo samo 10% podataka da ne pukne RAM
    subset_data = tfds.load('malaria', split='train[:10%]', as_supervised=True)
   
    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = MalariaDataset(subset_data, transform=transform)
    val_size = int(len(full_dataset) * config['data']['val_split'])
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    print(f"OPTIMIZED MODE: Using {len(full_dataset)} images total.")
    train_loader = DataLoader(train_ds, batch_size=config['data']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['data']['batch_size'], shuffle=False)
    return train_loader, val_loader
