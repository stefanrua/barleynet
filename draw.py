import json
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from image_utils import get_tiles, bbox_in_tile, xywh_to_x1y1x2y2

imgdir = '/home/axel-paivansalo/alien_barley/datasets/alien_barley/tiny_kultti/'
predfile = 'predictions.json'
fname = None
tile_w = 1200
tile_h = 800

helptext = f'''
Usage:
 {sys.argv[0]} [options]

Cuts an image into tiles, and for each tile with predictions,
draws the predictions and saves the tile to disk.
The output is saved in inference/<image_name>/tile_n.png

Options:
 -i <path>    image directory     default={imgdir}
 -p <path>    prediction json     default={predfile}
 -f <path>    input image         required
 -h, --help   get help

Example:
 python {sys.argv[0]} -f my_image.JPG
'''

for i in range(len(sys.argv)):
    arg = sys.argv[i]
    if arg == '-i': # images
        imgdir = sys.argv[i+1]
        i += 2
    if arg == '-p': # predictions json
        predfile = sys.argv[i+1]
        skip_empty = True
        i += 2
    if arg == '-f':
        fname = sys.argv[i+1]
        i += 2
    if arg == '-h' or arg == '--help':
        print(helptext)
        exit()

if fname is None:
    print('Input image required')
    print(helptext)
    exit()

output_dir = f"inference/{os.path.splitext(fname)[0]}"
os.makedirs(output_dir, exist_ok=True)

with open(predfile, 'r') as f:
    predictions = json.load(f)

img = Image.open(f'{imgdir}{fname}')
img = np.array(img)
img_h, img_w, _ = img.shape

score = 0.5
args = ['box', 'text']
img_predictions = predictions.get(fname, [])

# Create tiles from the image and check for predictions
for tile_coords, tile_img, tile_n in get_tiles(img, tile_w, tile_h):
    offset_x, offset_y, _, _ = tile_coords
    
    # Get all predictions for the current tile that are above the score threshold
    tile_predictions = [
        p for p in img_predictions
        if p['score'] >= score and bbox_in_tile(tile_coords, xywh_to_x1y1x2y2(p['bbox']))
    ]

    # If there are any predictions, draw them on the tile and save it
    if len(tile_predictions) > 0:
        fig, ax = plt.subplots()
        ax.imshow(tile_img)

        for pred in tile_predictions:
            bbox = pred['bbox']
            
            # transform bbox to tile coordinates
            new_bbox = [bbox[0] - offset_x, bbox[1] - offset_y, bbox[2], bbox[3]]

            if 'box' in args:
                rect = patches.Rectangle(
                        (new_bbox[0], new_bbox[1]), new_bbox[2], new_bbox[3],
                        linewidth=2,
                        edgecolor='white',
                        facecolor='none')
                ax.add_patch(rect)
            if 'dot' in args:
                ax.plot(new_bbox[0]+new_bbox[2]/2,
                        new_bbox[1]+new_bbox[3]/2,
                        marker='.',
                        color='white')
            if 'text' in args:
                ax.text(new_bbox[0],
                        new_bbox[1],
                        str(round(pred['score'], 2)),
                        color='white')
        
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        tilename = f"tile_{tile_n}.png"
        plt.savefig(f'{output_dir}/{tilename}', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
