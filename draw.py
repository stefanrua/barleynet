import json
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import patches

imgdir = 'images/'
predfile = 'predictions.json'

with open(predfile, 'r') as f:
    predictions = json.load(f)

def draw(fname):
    im = Image.open(f'{imgdir}{fname}')
    fig, ax = plt.subplots()
    ax.imshow(im)
    for pred in predictions[fname]:
        if pred['score'] >= 0.5:
            bbox = pred['bbox']
            rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2], bbox[3],
                    linewidth=1,
                    edgecolor='white',
                    facecolor='none')
            ax.add_patch(rect)
    plt.show()

if len(sys.argv) > 1:
    draw(sys.argv[1])
else:
    for fname in os.listdir(imgdir):
        draw(fname)
