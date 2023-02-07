from PIL import Image
import json
import numpy as np
import os
import sys
from tqdm import tqdm

imgdir = 'images/'
labeldir = None
tiledir = 'tiles/'
cocofile = 'instances.json'

skip_empty = False
tile_w = 1200
tile_h = 800

helptext = f'''
Usage:
 {sys.argv[0]} [options]

Cut images and generate COCO annotations

Options:
 -i <path>    full size images              default={imgdir}
 -l <path>    labels in labelme format      default={labeldir}
 -t <path>    cut output images             default={tiledir}
 -j <path>    output labels in coco format  default={cocofile}
'''

for i in range(len(sys.argv)):
    arg = sys.argv[i]
    if arg == '-i': # full size images
        imgdir = sys.argv[i+1]
        i += 2
    if arg == '-l': # labels in labelme format
        labeldir = sys.argv[i+1]
        skip_empty = True
        i += 2
    if arg == '-t': # cut images
        tiledir = sys.argv[i+1]
        i += 2
    if arg == '-j': # labels in coco format + positional info
        cocodir = sys.argv[i+1]
        i += 2
    if arg == '-h' or arg == '--help':
        print(helptext)
        exit()

# {"image_id": 1, "category_id": 0, "bbox": [672.3211669921875, 216.00074768066406, 38.85247802734375, 84.76887512207031], "score": 0.990157425403595}

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
annotation_id = 0 # initial annotation id, modified by save_tile()

def name_tile(fname, n):
    name, extension = fname.split('.')
    tilename = f'{name}_{n}.{extension}'
    return tilename

# returns: [[xmin, ymin, xmax, ymax], ...]
def bboxes_from_labelme(fname):
    with open(fname, 'r') as f:
        labelme = json.load(f)
    # group_id 0: bbox is gps measurement with no visible barley
    bboxes = [s['points'] for s in labelme['shapes'] if s['group_id'] != 0]
    bboxes = [b[0] + b[1] for b in bboxes]
    # some gps measurements are missing the group id, remove large boxes
    bboxes = [b for b in bboxes if (b[2] - b[0]) * (b[3] - b[1]) < 20000]
    return bboxes

# tile: [x1, y1, x2, y2]
# bbox: [x1, y1, x2, y2]
# returns: None if no overlap, else the overlapping part
#                                   [x, y, width, height]
def bbox_in_tile(tile, bbox):
    xmin = max(tile[0], bbox[0])
    xmax = min(tile[2], bbox[2])
    ymin = max(tile[1], bbox[1])
    ymax = min(tile[3], bbox[3])
    dx = xmax - xmin
    dy = ymax - ymin
    if (dx > 0) and (dy > 0):
        return [xmin, ymin, dx, dy]
    return None

def save_tile(offset_x, offset_y, max_x, max_y,
        img, bboxes, tile_n, annotations_coco, images_coco):
    global annotation_id
    tilecoords = [offset_x, offset_y, max_x, max_y]
    empty = True
    for bbox in bboxes:
        bbox = bbox_in_tile(tilecoords, bbox)
        if bbox:
            empty = False
            bbox = [
                    bbox[0] - offset_x,
                    bbox[1] - offset_y,
                    bbox[2],
                    bbox[3],
                ]
            annotations_coco.append({
                    "image_id": tile_id,
                    "category_id": 0,
                    "bbox": bbox,
                    "id": annotation_id,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "supercategory": "none",
                    "ignore": 0,
                })
            annotation_id += 1

    if not empty or not skip_empty:
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

# reads fname, cuts it into tiles, and saves to disk
# returns: list of images in coco format
def cut_image(fname):
    global tile_id
    bboxes = []
    if labeldir:
        labelfile = fname.replace('.JPG', '.json')
        labelfile = labelfile.replace('.jpg', '.json')
        bboxes = bboxes_from_labelme(f'{labeldir}{labelfile}')
    img = Image.open(f'{imgdir}{fname}')
    img = np.array(img)
    img_h, img_w, _ = img.shape
    offset_x = 0
    offset_y = 0
    tile_n = 0
    annotations_coco = []
    images_coco = []
    while offset_y < img_h:
        max_y = min(offset_y + tile_h, img_h)
        while offset_x < img_w:
            max_x = min(offset_x + tile_w, img_w)
            save_tile(offset_x, offset_y, max_x, max_y,
                img, bboxes, tile_n, annotations_coco, images_coco)
            tile_n += 1
            tile_id += 1
            offset_x += tile_w
        offset_x = 0
        offset_y += tile_h
    return annotations_coco, images_coco

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

for fname in tqdm(os.listdir(imgdir)):
    annotations_coco, images_coco = cut_image(fname)
    instances_coco['images'] += images_coco
    instances_coco['annotations'] += annotations_coco

with open(cocofile, 'w') as f:
    json.dump(instances_coco, f)
