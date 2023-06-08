# Object Selection via Mask Extraction

This repository holds codes that aim to conduct the first step in ROME -  object selection .
More specifically, we recursively select the objects in the
original image to be removed, which constructs the image
pairs of ancestors and descendants. 
The extracted masks will be saved in .png format, with information
from 80 object categories according to COCO Caption Dataset annotation. 

This repository does not directly generate the inpainted pictures, but
it generates the images and masks in a format that can be directly used by 
LaMa (available in `../image_mutation/` folder)

## Environment Setup
- Please make sure Conda is installed as we will create a new conda environment
- Follow the scripts below to create a new conda environment named 'lama'. This environment is shared with the `image_mutation` part.

    conda create --name object_selection python=3.10
    conda activate object_selection
    pip install torch torchvision torchaudio
    pip install -r requirements.txt

    <!-- if the above failed, try:
    conda env create -f conda_env.yml
    conda activate lama
    pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
    pip install pytorch-lightning==1.2.9 termcolor pycocotools -->

## Running the Code
First, if you would like to alter the settings like the maximum number of synthesized images per seed, please configure the properties in `config_object_selection.yaml`. The current setting is suitable for demonstration.

Then, inside this `object_selection` directory, run:

    conda activate object_selection
    python object_selection.py

The results are placed in `out/` folder.
The output `out/name_img_id_dict.pkl` contains deleted object information to be passed to MR checking.
The input to LaMa are further placed in `out/lama_inputs/ancestor` and `out/lama_inputs/descendant` folders,
which could directly be LaMa's input folders.

## More Explanations about File Structures

### About `sample_images/`
This folder contains a subset of images that we used in our main experiment, for demonstration purpose.

### About `instances_annotation.json`
This is the annotation file of the COCO 2014 image captioning dataset, which can be also found on their website.

This json file contains useful image annotations including segmentations, category 
annotations, image id information in COCO Dataset, and so on.

