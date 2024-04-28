'''
STEP 3
Input:
- autoencoder weights
- cropped images
Output:
- image_vector.csv: [vidId, frameNum, timestamp, detectedObjId, detectedObjClass, confidence, bbox info, vector]
'''

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import csv


# GET MODEL
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


model = ConvAutoencoder()
model.load_state_dict(torch.load('autoencoder.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# GET PATHS
cropped_images_path = 'cropped_images'
cropped_images_sorted_csv_path = 'cropped_images_sorted.csv'
image_vector_csv_path = 'image_vector.csv'

# START
with open(cropped_images_sorted_csv_path, mode='r') as file_read, \
        open(image_vector_csv_path, mode='w', newline='') as file_write:
    csv_reader = csv.reader(file_read)
    csv_writer = csv.writer(file_write)

    # NEW CSV HEADER
    header = next(csv_reader)
    header.append('vector')
    csv_writer.writerow(header)

    # PROCESS EACH IMAGE
    for row in csv_reader:
        image_file = f"{row[0]}_{row[1]}_{row[3]}.jpg"
        image_path = os.path.join(cropped_images_path, image_file)
        try:
            im = Image.open(image_path)
        except IOError:
            print(f"Could not load image from {image_path}")
            continue

        # GET IMAGE VECTOR
        image = transform(im).unsqueeze(0)
        with torch.no_grad():
            image_vector = model.encode(image)

        image_vector = image_vector.flatten().to("cpu")
        vector_str = ','.join([str(x) for x in image_vector.squeeze().tolist()])
        row.append(vector_str)
        csv_writer.writerow(row)

        print(image_file)
