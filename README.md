# barleynet

![](barley-detection.png)

Detect barleys in an oat field!

## Prerequisites

- A GPU with CUDA support
- Drone images of barleys in a an oat field

## Installation

You'll need Python's venv module. If you don't have it, install it using
```
sudo apt install python3-venv
```

Then, clone this repository and run the setup script.
```
git clone https://github.com/stefanrua/barleynet.git
cd barleynet
./setup.sh
```

## Inference

1. Put your images in the `images/` directory
1. Run `./detect-barleys.sh`
1. The model's predictions are in `predictions.json`

The `predictions.json` file is structured like this:
```json
{                                                      
    "DJI_20210709134934_0032.JPG": [                                                                           
        {                                                                                                      
            "bbox": [                                                                                          
                693.0487670898438,                                                                             
                92.3312759399414,                                                                              
                29.84295654296875,                                                                             
                30.927993774414062                     
            ],                                         
            "score": 0.06010641157627106                                                                       
        },                                                                                                     
        {                                              
            "bbox": [                                                                                          
                6576.6680698394775,                                                                            
                551.454402089119,                                                                              
                107.68167114257812,                                                                            
                58.3837776184082                                                                               
            ],                                                                                                 
            "score": 0.23410336673259735                                                                       
        },
    ]
}
```

## Visualization

To visualize the predictions for an image:
```
python3 draw.py DJI_20210709134934_0032.JPG
```
