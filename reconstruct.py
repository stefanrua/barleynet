import json
import os

# input:  instances.json, coco_instances_results.json
# output: predictions.json

instancefile = 'instances.json'
predfile_in = 'output/inference/coco_instances_results.json'
predfile_out = 'predictions.json'

ANNOTATIONS = os.getenv('ANNOTATIONS')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')

if ANNOTATIONS:
    instancefile = ANNOTATIONS
if OUTPUT_DIR:
    predfile_in = f'{OUTPUT_DIR}/inference/coco_instances_results.json'
    predfile_out = f'{OUTPUT_DIR}/predictions.json'

# instancefile
'''
{
    "images": [
        {
            "id": 198,
            "file_name": "DJI_20210709135005_0049_98.JPG",
            "width": 820,
            "heights": 510,
            "offset": [
                6560,
                4950
            ],
            "parent": "DJI_20210709135005_0049.JPG"
        },
    ]
}
'''

# predfile_in
'''
[
    {
        "image_id": 3465,
        "category_id": 0,
        "bbox": [
            421.7579040527344,
            0.19289562106132507,
            25.198028564453125,
            10.352855682373047
        ],
        "score": 0.06408041715621948
    },
]
'''

# predfile_out
'''
{
  'image.jpg': [
    {
      'bbox': [x, y, width, height],
      'score': confidence
    },
    {
      'bbox': [x, y, width, height],
      'score': confidence
    },
  ],
}
'''

def load_json(fname):
    with open(fname, 'r') as f:
        j = json.load(f)
    return j

tiles = load_json(instancefile)['images']
preds_in = load_json(predfile_in)
preds_out = {}

# organize by id for faster reads
by_id = {}
for tile in tiles:
    by_id[tile['id']] = {
        'offset': tile['offset'],
        'parent': tile['parent']
    }

for pred in preds_in:
    tileid = pred['image_id']
    tile = by_id[tileid]
    image = tile['parent']
    bbox = pred['bbox']
    bbox[0] += tile['offset'][0]
    bbox[1] += tile['offset'][1]
    pred_out = {
            'bbox': bbox,
            'score': pred['score']
        }
    if image in preds_out:
        preds_out[image].append(pred_out)
    else:
        preds_out[image] = [pred_out]

with open(predfile_out, 'w') as f:
    json.dump(preds_out, f)
