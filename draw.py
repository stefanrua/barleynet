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

def draw(fname, score, args):
    im = Image.open(f'{imgdir}{fname}')
    fig, ax = plt.subplots()
    ax.imshow(im)
    print(fname)
    print(score)
    print(args)
    for pred in predictions[fname]:
        if pred['score'] >= score:
            bbox = pred['bbox']
            alpha = 1
            if 'alpha' in args:
                alpha = pred['score']
            if 'box' in args:
                rect = patches.Rectangle(
                        (bbox[0], bbox[1]), bbox[2], bbox[3],
                        linewidth=2,
                        edgecolor='white',
                        facecolor='none',
                        alpha=alpha)
                ax.add_patch(rect)
            if 'dot' in args:
                ax.plot(bbox[0]+bbox[2]/2,
                        bbox[1]+bbox[3]/2,
                        marker='.',
                        color='white',
                        alpha=alpha)
            if 'text' in args:
                ax.text(bbox[0],
                        bbox[1],
                        str(round(pred['score'], 2)),
                        color='white',
                        alpha=alpha)
    plt.show()

#if len(sys.argv) > 1:
#    draw(sys.argv[1])
#else:
#    for fname in os.listdir(imgdir):
#        draw(fname)

fname = ''
score = 0.5
args = ['box', 'alpha']
while True:
    try:
        f = input('Filename: ')
        if len(f) > 0: fname = f
        s = input('Score: ')
        if len(s) > 0: score = float(s)
        a = input('Args: ')
        if len(a) > 0: args = a.split(' ')
        draw(fname, score, args)
    except Exception as e: print(e)
