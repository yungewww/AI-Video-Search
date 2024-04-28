from flask import Flask, request, render_template, send_from_directory
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


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


csv_file = "image_vector.csv"
df = pd.read_csv(csv_file)
all_image_vectors = np.array([np.fromstring(vec[1:-1], dtype=float, sep=',') for vec in df.iloc[:, -1]])

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    file = request.files['file']
    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Get Input Image Vectors
    input_image = Image.open(file_path)
    input_image = transform(input_image).unsqueeze(0)
    with torch.no_grad():
        input_image_vector = model.encode(input_image)

    # Get Top 10 Similar Image Paths
    num_top = 10
    similarities = cosine_similarity(input_image_vector.reshape(1, -1), all_image_vectors).flatten()
    top_indices = np.argsort(similarities)[-num_top:][::-1]
    similar_image_paths = [
        f"{df.iloc[idx]['vidId']}_{df.iloc[idx]['frameNum']}_{df.iloc[idx]['detectedObjId']}.jpg"
        for idx in top_indices
    ]
    app.logger.info(similar_image_paths)

    return render_template('display.html', filename=filename, similar_image_paths=similar_image_paths)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)
