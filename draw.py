import json
from PIL import Image
import os

imgdir = 'images/'
predfile = 'predictions.json'

with open(predfile, 'r') as f:
    predictions = json.load(f)

for fname in os.listdir(imgdir):
    img = Image.open(f'{imgdir}{fname}')
    bbox = predictions['fname']['bbox']

    fig, ax = plt.subplots()
    ax.imshow(im)
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='white', facecolor='none')
    ax.add_patch(rect)
    plt.show()

