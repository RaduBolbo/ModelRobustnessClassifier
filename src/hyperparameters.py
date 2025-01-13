from torchvision import transforms as T

# paths
dataset_root = "dataset/raw-img"

epochs = 15
# lr = 1e-4 # this is good
lr = 1e-4
batch_size = 32
num_workers = 8
device = "cuda"

# mean, std and image dimensions
mean, std, size = (
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225],
    224,
)

# training and validation transformations
val_tfs = T.Compose(
    [
        T.ToTensor(),
        T.Resize(size=(size, size), antialias=False),
        T.Normalize(mean=mean, std=std),
    ]
)
train_tfs = T.Compose(
    [
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=45),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        T.ToTensor(),
        T.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.2)),
        T.Resize(size=(size, size), antialias=False),
        T.Normalize(mean=mean, std=std),
    ]
)
