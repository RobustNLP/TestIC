# Runs captioning inference on OFA
# This file should be copied to the root of the OFA repo

import torch
import numpy as np
from fairseq import utils,tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks.caption import CaptionTask
from models.ofa import OFAModel
from PIL import Image
import argparse
import glob
import os
import pdb
import json
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="checkpoints/caption_base_best.pt")
parser.add_argument('--in_dir', type=str)
parser.add_argument('--out_file', type=str)
args = parser.parse_args()

input_imgs_size = len(os.listdir(args.in_dir))
if 'origin_imgs' in os.listdir(args.in_dir):
    input_imgs_size -= 1
    # pdb.set_trace()


# Register caption task
tasks.register_task('caption',CaptionTask)

# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
# use fp16 only when GPU is available
use_fp16 = False
# use_fp16 = True

"""## **Build Model**
Below you can build your model and load the weights from the given checkpoint, and also build a generator. 
"""

# Load pretrained ckpt & config

# ofa_base
# overrides={"bpe_dir":"utils/BPE", "eval_cider":False, "beam":5, "max_len_b":16, "no_repeat_ngram_size":3, "seed":7}
# models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
#         utils.split_paths('ofa_base.pt'),
#         arg_overrides=overrides
#     )

# ofa finetuned caption large
overrides={"bpe_dir":"utils/BPE", "eval_cider":False, "beam":5, "max_len_b":16, "no_repeat_ngram_size":3, "seed":7}
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths(args.model),
        arg_overrides=overrides
    )


# Move models to GPU
for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

# Initialize generator
generator = task.build_generator(models, cfg.generation)

"""## **Preprocess**
We demonstrate the required transformation fucntions for preprocessing inputs.
"""

# Image transform
from torchvision import transforms
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Text preprocess
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()
def encode_text(text, length=None, append_bos=False, append_eos=False):
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s

# Construct input for caption task
def construct_sample(image: Image, id):
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    src_text = encode_text(" what does the image describe?", append_bos=True, append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask
        }
    }
    return sample
  
# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t

"""## **Run Inference**
Download an image and run the following scripts to generate the caption.
"""

# Download an image from COCO or you can use other images with wget
# ! wget http://farm4.staticflickr.com/3539/3836680545_2ccb331621_z.jpg 
# ! mv 3836680545_2ccb331621_z.jpg  test.jpg
# image = Image.open('test.png')

fout = open(args.out_file, 'w')

for i in range(1, input_imgs_size+1):
    image = Image.open(os.path.join(args.in_dir, f'{i}.png'))

    # Construct input sample & preprocess for GPU if cuda available
    sample = construct_sample(image, i)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

    # Run eval step for caption
    with torch.no_grad():
        result, scores = eval_step(task, generator, models, sample)
    # pdb.set_trace()
    str_out = f'{i}\t' + json.dumps(result) + '\n'
    # pdb.set_trace()
    fout.write(str_out)
    print(i, str_out)
    # pdb.set_trace()

    # Append result to tsv file

# display(image)
# print('Caption: {}'.format(result[0]['caption']))

fout.close()