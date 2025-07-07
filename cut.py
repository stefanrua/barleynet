from PIL import Image
import json
import numpy as np
import os
import sys
from tqdm import tqdm
from image_utils import get_tiles, get_overlapping_part, x1y1x2y2_to_xywh

imgdir = '/home/axel-paivansalo/alien_barley/datasets/alien_barley/tiny_kultti/'
labeldir = None
tiledir = 'tiles/'
cocofile = 'instances.json'
resume = False

skip_empty = False
tile_w = 1200
tile_h = 800
existing_files = []

helptext = f'''
Usage:
 {sys.argv[0]} [options]

Cut images and generate COCO annotations

Options:
 -i <path>    full size images              default={imgdir}
 -l <path>    labels in labelme format      default={labeldir}
 -t <path>    cut output images             default={tiledir}
 -j <path>    output labels in coco format  default={cocofile}
 --resume     don't re-cut done files
 -h, --help   get help
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
        cocofile = sys.argv[i+1]
        i += 2
    if arg == '--resume':
        resume = True
        i += 1
    if arg == '-h' or arg == '--help':
        print(helptext)
        exit()

os.makedirs(tiledir, exist_ok=True)

if resume:
    existing_files = ['_'.join(x.split('_')[:-1]) + '.JPG' for x in os.listdir(tiledir)]
    counts = {}
    for f in existing_files:
        if f in counts:
            counts[f] += 1
        else:
            counts[f] = 1
    existing_files = set(existing_files)
    mean = sum(counts.values()) / len(counts)
    for f in counts:
        if counts[f] < mean:
            print(f'{f} only has {counts[f]} tiles. Marking as not done.')
            existing_files.remove(f)

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

# reads fname, cuts it into tiles, and saves to disk
# returns: list of images in coco format
def cut_image(fname):
    global tile_id, annotation_id
    bboxes = []
    if labeldir:
        labelfile = fname.replace('.JPG', '.json')
        labelfile = labelfile.replace('.jpg', '.json')
        bboxes = bboxes_from_labelme(f'{labeldir}{labelfile}')
    
    img = None
    try:
        if fname not in existing_files:
            img = Image.open(f'{imgdir}{fname}')
            img = np.array(img)
    except FileNotFoundError:
        print(f"Image not found: {imgdir}{fname}")
        return [], []

    annotations_coco = []
    images_coco = []
    
    if img is None: # if file exists but we are resuming and don't load it
        img = np.empty((5460, 8192, 3)) # placeholder

    for tile_coords, tile_img, tile_n in get_tiles(img, tile_w, tile_h):
        offset_x, offset_y, max_x, max_y = tile_coords
        
        tile_annotations = []
        for bbox in bboxes:
            overlapping_part = get_overlapping_part(tile_coords, bbox)
            if overlapping_part:
                # transform bbox to tile coordinates
                new_bbox = [
                    overlapping_part[0] - offset_x,
                    overlapping_part[1] - offset_y,
                    overlapping_part[2] - offset_x,
                    overlapping_part[3] - offset_y
                ]
                new_bbox_xywh = x1y1x2y2_to_xywh(new_bbox)
                
                tile_annotations.append({
                    "image_id": tile_id,
                    "category_id": 0,
                    "bbox": new_bbox_xywh,
                    "id": annotation_id,
                    "area": new_bbox_xywh[2] * new_bbox_xywh[3],
                    "iscrowd": 0,
                    "supercategory": "none",
                    "ignore": 0,
                })
                annotation_id += 1
        
        if not skip_empty or len(tile_annotations) > 0:
            annotations_coco.extend(tile_annotations)
            
            tilename = name_tile(fname, tile_n)
            if fname not in existing_files:
                tile_to_save = Image.fromarray(tile_img)
                tile_to_save.save(f'{tiledir}{tilename}')
            
            images_coco.append({
                'id': tile_id,
                'file_name': tilename,
                'width': max_x - offset_x,
                'height': max_y - offset_y,
                'offset': [offset_x, offset_y],
                'parent': fname,
            })
            tile_id += 1
            
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
