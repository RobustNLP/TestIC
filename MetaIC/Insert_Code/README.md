## Environment Requirements 
- Python 3.8
- Json 
- Numpy 
- Pickle 
- Cv2 
- Re 
- Argparse 

## Preparatory Work
Run download.sh under `Insert_Code/` to download necessary files. 
```
sh download.sh 
```
After that we have `./coco_dataset_annotation/coco_annotations_trainval2014/annotations`, `./val2014`, and `./dataset_split/caption_datasets`, which can be downloaded from
```
http://images.cocodataset.org/annotations/annotations_trainval2014.zip

http://images.cocodataset.org/zips/val2014.zip

http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
```

When we do insertion task, we need to calculate the intersection ratio between the inserted object and the objects which were annotated in the images, so we run `get_annotations_dict.py`.
```
python get_annotations_dict.py
```

And we will get `image_dict.pkl`, which contains coordinates of the objects in the test set. 

According to the Karpathy split method, there are 5000 images in MSCOCO 2014 which were tagged as the test set.

Then we choose images that only contain objects classified as 60 categories that we have chosen previously, placing the chosen images in `./chosen_bg_images_all`.
Just run `image_filter.py`
```
python image_filter.py
```

## Insertion
Then we do the insertion, the objects which are going to be inserted into the images are placed in `./YOLACT_seg_results/image_pool_new`

For each object-image pair, we will do insertion for four times of four range of intersection, which are 0%, (0%, bar1], (bar1, bar2], (bar2, bar3], and separately placed in
```
./inserted_result/inserted_result_same0
./inserted_result/inserted_result_same_bar1
./inserted_result/inserted_result_same_bar2
./inserted_result/inserted_result_same_bar3
```
We also reserve the background images which are placed in `./inserted_result/bg_images`. 

bar1, bar2, and bar3 should be specified. 
For example
```
python insert.py --bar1 0.15 --bar2 0.30 --bar3 0.45
```
For the limitation of file size for supplement, we only reserve part of the new images in `./inserted_result`.

## Reorder
After finishing the insertion, we will rename and reorder the images we just produced.

We will place the reordered images in `./ordered_result`.
At the same time, we produce a `name_img_id_dict.pkl`
which records the image_id and object name of new images.
To do the reorder task, just run `reorder.py`
```
python reorder.py
```
