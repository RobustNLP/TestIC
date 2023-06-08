# First, clone the show-attend-and-tell repo:
# git clone https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning.git
# Then use the following command to run the captioning script:
# replace the paths with the correct paths

python caption.py --img='path/to/image.jpeg' --model='path/to/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' --word_map='path/to/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' --beam_size=5