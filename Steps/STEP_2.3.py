'''
STEP 2.3
Input: CoCo Dataset (5 classes)
Output: autoencoder.pth
'''


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from PIL import UnidentifiedImageError
import os

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        with torch.no_grad():
            return self.encoder(x)

    def decode(self, x):
        with torch.no_grad():
            return self.decoder(x)

autoencoder = ConvAutoencoder()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# CHECKPOINT & SAVE
checkpoint_path = 'autoencoder_checkpoint.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
else:
    start_epoch = 0

def train(model, dataloader, epochs):
    model.train()
    i = 0
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")
        for data in dataloader:
            i += 1
            print(f"Processing batch {i, + 1}/{len(dataloader)}")

            inputs = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, f'autoencoder_checkpoint_epoch_{epoch}.pth')

from pycocotools.coco import COCO
class COCODataset(Dataset):
    def __init__(self, annotation_path, image_dir, categories=['person'], transform=None):
        self.coco = COCO(annotation_path)
        self.image_dir = image_dir
        self.categories = categories
        self.catIds = self.coco.getCatIds(catNms=self.categories)

        imgIds = []
        for catId in self.catIds:
            imgIds.extend(self.coco.getImgIds(catIds=[catId]))
        self.imgIds = list(set(imgIds))
        self.transform = transform

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, idx):
        try:
            img_id = self.imgIds[idx]
            img_info = self.coco.loadImgs(img_id)[0]
            path = os.path.join(self.image_dir, img_info['file_name'])
            image = Image.open(path).convert('RGB')

            annIds = self.coco.getAnnIds(imgIds=img_id, catIds=self.catIds, iscrowd=None)
            anns = self.coco.loadAnns(annIds)

            if len(anns) > 0:
                ann = anns[0]
                bbox = ann['bbox']
                object_image = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

                if self.transform:
                    object_image = self.transform(object_image)

                return object_image
            else:
                return self.get_placeholder_image()
        except UnidentifiedImageError:
            print(f"Warning: Could not identify image file {path}. Returning placeholder.")
            return self.get_placeholder_image()

    def get_placeholder_image(self):
        placeholder = torch.zeros((3, 224, 224), dtype=torch.float32)
        return placeholder

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ——————————————————————— START TRAINING ———————————————————————————————————————————

# IMPORT DATASET
dataset = COCODataset(
    annotation_path='/Users/yunge/fiftyone/coco-2017/raw/instances_train2017.json',
    image_dir='/Users/yunge/fiftyone/coco-2017/train/data/',
    categories=['person', 'bicycle', 'car', 'train', 'boat', 'cake'],
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# TRAIN
train(autoencoder, dataloader, epochs=10)

# SAVE
torch.save(autoencoder.state_dict(), 'autoencoder.pth')
torch.save(autoencoder, 'autoencoder_complete.pth')
