import json
import math
import numpy as np
from PIL import Image
import os
import torch
import pickle
import cv2
import random
import hydra
from typing import Dict, List
import glob
from tqdm import tqdm
from termcolor import colored
from maskUtils import polygons_to_bitmask, rle_to_bitmask
import logging

logger = logging.getLogger(__name__)

if torch.cuda.is_available(): device = torch.device('cuda')
else: device = torch.device('cpu')

# The pkl to be passed to MR checking step
# indexed by unique ids as in the output filename
# values are dicts of {'file_name', 'image_id', 'removed_object_ancestor', 'removed_object_children'}
# The latter two are unique strings of removed objects, e.g. 'cat_1'
pkl_object = dict()
overlap_filtering_pkl = dict()      # The pkl that contains filtering information

def extract_mask_and_save(instance_info_all: list, config, id, file_idx: int, src_img_path: str, is_last_one: bool) -> int:
    """
    Extracts mask from a list of object instance information in an image.

    @param instance_info_all: list of ([cat_name, mask_arr]), containing all annotated instances of this image in COCO Caption dataset
    @param config: the configuration object of hydra
    @param id: image id of COCO Caption Dataset
    @param file_idx: the file name to be saved, an integer
    @param src_img_path: the path to the source image of COCO Caption Dataset
    @param is_last_one: if this parameter is set true, after this image the pkl to be passed to MR checking will be saved and program terminates
    @return: the file index saved
    """

    is_valid = False
    _, file_name, _ = parse_id_filenamenoext(src_img_path)

    # Re-index each object with cat name + index
    for idx in range(len(instance_info_all)):
        instance_info_all[idx][0] += '_' + str(idx+1)
    ori_instance_info_all = instance_info_all.copy()

    # Filter out maximum space object. We won't remove them for saliency reasons
    instance_info_all = sorted(instance_info_all, key=lambda x: get_mask_space(x[1]))       # In ascending space order
    instance_info_all = instance_info_all[0:-1]
    random.shuffle(instance_info_all)

    if len(instance_info_all) == 0: is_valid = False
    else: is_valid = True
    available_set = set()
    solution_list = []
    state_results = dict()
    # max_num = config.max_num

    if is_valid:
        img_shape = instance_info_all[0][1].shape   # The shape of all masks are same as the image
        for [cat_name, mask_arr] in instance_info_all:
            available_set.add(cat_name)
        RECURSION(available_set, set(), solution_list, state_results, config.max_depth) # Limit the maximum depth into the recursion, 3 in paper
        test_cases = random.sample(solution_list, min(len(solution_list), config.max_num))  # Randomly select some cases according to configuration; 100 in the paper
        for case in test_cases:
            pkl_object[file_idx] = {
                'image_id': id,
                'file_name': file_name,
            }

            save_png(src_img_path, case, img_shape, file_idx, instance_info_all, config, ori_instance_info_all)
            file_idx += 1

    # Dump the pickle dictionary when reaching the end, after this the program terminates
    if is_last_one:
        print('Writing to PKL')
        write_pkl_output(pkl_object, config.pkl_dir, 'name_img_id_dict.pkl')
        exit(0)

    return file_idx


def save_png(src_img_path: str, case, img_shape, file_idx, instance_info_all, config, ori_instance_info_all):
    '''
    An auxiliary function to convert the bitmask tensor to png format and save

    @param src_img_path: the target path to save the bitmask png
    @param case: a list of two items containing the ancestor image removed object names and the descendant image removed object names
    
    Other params: see `extract_mask_and_save`
    '''
    # Initialize the info dict for this iteration
    ancestor_deleted_objects_set = case[0]
    pkl_object[file_idx]['removed_object_ancestor'] = ancestor_deleted_objects_set
    ancestor_mask = torch.zeros(img_shape).to(torch.uint8).to(device)
    descendant_deleted_objects_set = case[1]
    descendant_mask = torch.zeros(img_shape).to(torch.uint8).to(device)
    descendant_deleted_objects = []
    ancestor_deleted_objects = []
    pkl_object[file_idx]['removed_object_descendant'] = descendant_deleted_objects_set
    for object_name in ancestor_deleted_objects_set:
        mask = query_mask(object_name, instance_info_all)
        ancestor_mask = torch.logical_or(ancestor_mask, mask).to(torch.uint8)
        ancestor_deleted_objects.append([object_name, ancestor_mask])
    for object_name in descendant_deleted_objects_set:
        mask = query_mask(object_name, instance_info_all)
        descendant_mask = torch.logical_or(descendant_mask, mask).to(torch.uint8)
        descendant_deleted_objects.append([object_name, descendant_mask])
    save_origin_and_mask(f'{config.output_dir}/ancestor', file_idx, ancestor_mask, src_img_path)
    save_origin_and_mask(f'{config.output_dir}/descendant', file_idx, descendant_mask, src_img_path)
        
def query_mask(object_name: str, instance_info_all) -> torch.Tensor:
    '''
    Find bitmask (Tensor) according to object name string
    '''
    for [name, mask_arr] in instance_info_all:
        if name == object_name:
            return mask_arr

def RECURSION(available_set: set, deleted_set: set, solution_list: list, state_results: dict, max_depth: int):
    '''
    This function is the main recursive steps proposed in our paper, in which
    we recursively select the objects in the original image to be removed and
    form test pairs of ancestors and descendants. 


    @param available_set: a list of available object candidate names in a seed image
    @param max_depth: a maximum recursion depth
    @param deleted_set & state_result: used inside recursive steps to pass the current 

    @param solution_list:  a list containing pairs of object set
        (ances set, desc set), where ances set and desc set represent
        the objects selected to be removed in the ancestor image and
        descendant image, respectively
    '''
    current_hash_index = Hash_Index(deleted_set)

    if current_hash_index in state_results.keys():
        return
        
    state_results[current_hash_index] = []

    # Add deleted_set to state_results[current_hash_index]
    state_results[current_hash_index].append(deleted_set)

    if max_depth == len(deleted_set):
        return

    # for object in available_set if object not in current_deleted_objects do
    # waiting_to_deleted_objects = [i for i in available_set if i not in current_deleted_objects]
    # for object in waiting_to_deleted_objects:
    for object in available_set - deleted_set:

        # next_deleted_objects <-- current_deleted_objects.copy()
        next_deleted_set = deleted_set.copy()
        # Add object to next_deleted_objects
        next_deleted_set.add(object)

        RECURSION(available_set, next_deleted_set, solution_list, state_results, max_depth)
        next_hash_index = Hash_Index(next_deleted_set)

        for descendant_deleted_set in state_results[next_hash_index]:

            if descendant_deleted_set not in state_results[current_hash_index]:
                state_results[current_hash_index].append(descendant_deleted_set)
            if (deleted_set, descendant_deleted_set) not in solution_list:

                # Add deleted_set, descendant_deleted_set to solution_list
                solution_list.append((deleted_set, descendant_deleted_set))

def Hash_Index(deleted_set: set):
    return hash(tuple(deleted_set))

def write_pkl_output(pkl_out: dict, target_dir: str, filename: str):
    '''
    An auxilliary function to save the file through pickle to be passed to MR step.
    
    @param pkl_out: The pkl to be passed to MR checking step, indexed by unique ids as in the output filename, 
        whose values are dicts of {'file_name', 'image_id', 'removed_object_ancestor', 'removed_object_children'}
        The latter two are unique strings of removed objects, e.g. 'cat_1'
    '''
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, filename), 'wb') as f:
        pickle.dump(pkl_out, f)

def save_origin_and_mask(target_dir: str, file_idx: int, mask: torch.Tensor, src_img_path):
    # Save the mask
    
    mask_to_png_and_save(mask, os.path.join(target_dir, str(file_idx)+'_mask.png'), target_dir)
    # Save the original image
    img = Image.open(src_img_path)
    img.save(os.path.join(target_dir, str(file_idx)+'.png'))

def get_img_paths(config) -> List:
    img_paths = glob.glob(config.input_dir)
    return img_paths

def get_raw_ann_dict(config):
    '''Load json file, which is the annotation file in the COCO Caption Dataset as a dict'''
    with open(config.annotation_path) as f:
        tmpStr = f.readline()
    parent_dict = json.loads(tmpStr)   # The parsed dictionary object, containing annotations for all used categories
    return parent_dict

def mask_to_png_and_save(mask_arr, dest_path: str, dest_dir: str):
    '''Auxiliary function to save the bitmask'''
    if not os.path.isdir(dest_dir): os.makedirs(dest_dir)
    mask_arr = 255 * mask_arr
    rgb_arr = torch.stack([mask_arr, mask_arr, mask_arr], dim=2)
    img = Image.fromarray(rgb_arr.cpu().numpy(),mode='RGB')
    print('Saving to', dest_path)
    img.save(dest_path)

def query_img_height_width(img_list: List[Dict], target_id: int):
    '''Get the image height and width specified as in the COCO Dataset annotation'''
    for img_dict in img_list:
        if img_dict['id'] != target_id: continue
        else:
            return img_dict['height'], img_dict['width']

def query_segmentation_catid_list(ann_list: List[Dict], target_id: int) -> List:
    '''Get the segmentation and category id as in the COCO Dataset annotation'''
    returnList = []
    for ann_dict in ann_list:
        if ann_dict['image_id'] != target_id: continue
        else:
            returnList.append({"segmentation": ann_dict['segmentation'], "cat_id": ann_dict['category_id']})
    if len(returnList) == 0: colored(f"Warning: Unreached target id: {target_id}", "yellow")
    return returnList

def get_mask_space(mask):
    return float(torch.sum(mask)) / (mask.shape[0] * mask.shape[1])

def query_category_name(cat_list: List, target_cat_id: int):
    '''Get the category name from category id, as both specified in the COCO Dataset annotation'''
    for cat_dict in cat_list:
        if cat_dict['id'] != target_cat_id: continue
        else:
            return cat_dict['name']
    colored(f"Error: No such category id: {target_cat_id}", "red")

def parse_id_filenamenoext(path: str):
    '''An auxiliary function to get the COCO image id from the COCO image's filename'''
    # Because the ids are written in filename, we parse them
    # Located at the end 
    if not (os.path.isfile(path) and (path.endswith('.jpg') or path.endswith('.png'))):
        colored(f'Warning: {path} is not valid and will be skipped.', 'yellow')
    id_string = path[:-4].split('_')[-1]
    filenamenoext = path[:-4].split('/')[-1]
    _, ext = os.path.splitext(path)

    return int(id_string), filenamenoext, ext

@hydra.main(config_name='config_object_selection.yaml', config_path='./')
def main(config):
    # Change the working directory to the one of this script
    os.chdir(hydra.utils.get_original_cwd())

    os.makedirs(config.output_dir, exist_ok=True)
    img_paths = get_img_paths(config)
    raw_coco_ann_dict = get_raw_ann_dict(config)
    file_idx = 1
    is_last_one = False
    img_paths_len = len(img_paths)
    random.seed(0)

    for [idx, src_img_path] in tqdm(enumerate(img_paths)):
        if idx >= img_paths_len-1: 
            is_last_one = True

        id, filenamenoext, ext_name = parse_id_filenamenoext(src_img_path)
        # Annotation list, contains `segmentation`, `category_id`
        ann_list = raw_coco_ann_dict['annotations']
        # Images list, contains image `width` and `height`
        img_list = raw_coco_ann_dict['images']
        # Category list, contains `name` and `id`(category id)
        cat_list = raw_coco_ann_dict['categories']

        img_height, img_width = query_img_height_width(img_list, id)
        segmentation_list = query_segmentation_catid_list(ann_list, id) # A list containing all segmentation annotation
        target_dir = config.output_dir
        
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)

        # Initialize instance_info_all, which is a
        # list of ([cat_name, mask_arr]), containing all annotated instances of this image in COCO Caption dataset
        instance_info_all = [] 

        for i, seg_catid_dict in enumerate(segmentation_list):
            if isinstance(seg_catid_dict['segmentation'], List):
                mask_arr = polygons_to_bitmask(seg_catid_dict['segmentation'], img_height, img_width)   # This is the true-false mask
            else: mask_arr = rle_to_bitmask(seg_catid_dict['segmentation'])
            cat_name = query_category_name(cat_list, seg_catid_dict['cat_id'])
            cat_name = cat_name.split()
            cat_name = '_'.join(cat_name)       # Join phrases with underscores

            # Expand edges (dilation)
            kernel_size = int(math.sqrt(mask_arr.shape[0]*mask_arr.shape[1]) / 35) # Dilate the mask by some values
            # We perform dilation mainly to avoid incomplete object area annotations and to improve inpainting results
            # The extent of dilation is not a primary focus, here we just chose a dilation kernel size from our experience
            # However, different dilation kernel size would work and this value is expected to change to fit with different data
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            mask_arr = cv2.dilate(mask_arr, kernel, 1) 
            mask_arr = torch.tensor(mask_arr).to(device)

            instance_info_all.append([cat_name, mask_arr])

        file_idx = extract_mask_and_save(instance_info_all, config, id, file_idx, src_img_path, is_last_one)

if __name__ == '__main__':
    main()