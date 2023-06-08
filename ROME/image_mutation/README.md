# Image Mutation via LaMa Image Inpainting
## Environment Setup
Please refer to the [object selection](../object_selection/README.md) part to prepare the required conda environment.
## Checkpoint Preparation
Please install the `big-lama` checkpoint with:

    cd lama
    pip3 install wldhx.yadisk-direct
    curl -L $(yadisk-direct https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) -o big-lama.zip
    # or manually download from https://disk.yandex.ru/d/ouP6l8VJ0HpMZg
    unzip big-lama.zip

## Inference

First, please ensure your current working directory is the `lama` folder.

    conda activate object_selection
    export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
    python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/../../object_selection/out/lama_inputs/ancestor outdir=$(pwd)/output/ancestor
    python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/../../object_selection/out/lama_inputs/descendant outdir=$(pwd)/output/descendant


## Examine the Outputs
The outputs of the inpainted ancestor and descendant images are saved in `lama/output/ancestor` and `lama/output/descendant` directory, respectively