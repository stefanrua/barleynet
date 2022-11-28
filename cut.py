from PIL import Image
import json
import numpy as np
import os

# input:  imgdir/image.jpg
# output: tiledir/tile.jpg, instances.json

verbose = True
imgdir = 'images/'
tiledir = 'tiles/'
cocofile = 'instances.json'
tile_w = 1000
tile_h = 1000

# instances.json looks like this:
'''
{
    "categories": [
        {
            "id": 0,
            "name": "barley",
            "supercategory": "none"
        }
    ],
    "annotations": [],
    "images": [
        {
            "id": 198,
            "file_name": "DJI_20210709135005_0049_98.JPG",
            "width": 820,
            "height": 510,
            "offset": [
                6560,
                4950
            ],
            "parent": "DJI_20210709135005_0049.JPG"
        },
        {
            "id": 199,
            "file_name": "DJI_20210709135005_0049_99.JPG",
            "width": 812,
            "height": 510,
            "offset": [
                7380,
                4950
            ],
            "parent": "DJI_20210709135005_0049.JPG"
        }
    ]
}
'''

tile_id = 0 # initial tile id, modified by cut_image()

def name_tile(fname, n):
    name, extension = fname.split('.')
    tilename = f'{name}_{n}.{extension}'
    return tilename

# reads fname, cuts it into tiles, and saves to disk
# returns: list of images in coco format
def cut_image(fname):
    global tile_id
    img = Image.open(f'{imgdir}{fname}')
    img = np.array(img)
    img_h, img_w, _ = img.shape
    if verbose: print(f'  shape: ({img_w}, {img_h})')
    offset_x = 0
    offset_y = 0
    tile_n = 0
    images_coco = []
    while offset_y < img_h:
        max_y = min(offset_y + tile_h, img_h)
        while offset_x < img_w:
            if verbose:
                print(f'  tile {tile_n}, offset: ({offset_x}, {offset_y})')
            max_x = min(offset_x + tile_w, img_w)
            tile = img[offset_y:max_y, offset_x:max_x, :]
            tile = Image.fromarray(tile)
            tilename = name_tile(fname, tile_n)
            tile.save(f'{tiledir}{tilename}')
            images_coco.append({
                    'id': tile_id,
                    'file_name': tilename,
                    'width': max_x - offset_x,
                    'height': max_y - offset_y,
                    'offset': [offset_x, offset_y],
                    'parent': fname,
                })
            tile_n += 1
            tile_id += 1
            offset_x += tile_w
        offset_x = 0
        offset_y += tile_h
    return images_coco

instances_coco = {
    "categories": [
        {
            "id": 0,
            "name": "barley",
            "supercategory": "none"
        }
    ],
    "annotations": [],
    "images": []
}

for fname in os.listdir(imgdir):
    if verbose: print(f'cutting {fname}')
    images_coco = cut_image(fname)
    instances_coco['images'] += images_coco

with open(cocofile, 'w') as f:
    json.dump(instances_coco, f)
