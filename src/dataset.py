import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

DATA_DIR = "../data/raw/Tobacco3482-jpg"

def get_all_filepaths_and_labels(data_dir=DATA_DIR):
    """
    Walks through the dataset folder and collects (filepath, label) pairs.
    """
    classes = sorted(os.listdir(data_dir))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    filepaths = []
    labels = []

    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepaths.append(os.path.join(cls_path, fname))
                labels.append(class_to_idx[cls])

    return filepaths, labels, classes, class_to_idx


def create_stratified_splits(filepaths, labels, test_size=0.15, val_size=0.15, random_state=42):
    """
    Splits data into train/val/test while preserving class proportions (stratified).
    """
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        filepaths, labels, test_size=test_size, stratify=labels, random_state=random_state
    )

    relative_val_size = val_size / (1 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=relative_val_size,
        stratify=train_val_labels, random_state=random_state
    )

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


class DocumentDataset(Dataset):
    """
    Custom PyTorch Dataset for the Tobacco3482 document classification task.
    """
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('L')  # ensure grayscale

        if self.transform:
            image = self.transform(image)

        return image, label
    
# ---- Transforms ----

IMG_SIZE = 224

# Training transforms: includes light augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ], p=0.3),
    transforms.RandomApply([
        transforms.RandomAffine(degrees=2, translate=(0.02, 0.02))
    ], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # single channel (grayscale)
])

# Val/Test transforms: NO augmentation, just resize + normalize
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])  