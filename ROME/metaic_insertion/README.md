# MetaIC insertion
To compare with ROME, we utilized MetaIC to synthesize images, generate captions with different captioning systems and then detect errors in the captions.
## Image synthesis
MetaIC insert objects from `YOLACT_seg_results/image_pool_80` into the background images from `chosen_bg_images_all/` with different overlap ratio threasholds.
To perform the insertion, execute the following script:
```shell
python metaic_insert.py --bar1 0.15 --bar2 0.3 --bar3 0.45
```
The variable `bar` represents different overlap ratio thresholds.
The synthesized images will be saved in the `inserted_result_80/` directory.
Note that we are providing only 30 background images for demonstration purposes.
To obtain more background image, please refer to [MS COCO dataset](http://images.cocodataset.org/zips/val2014.zip).
## Reorder images
Before captioning the images, we need to rename them and record the inserted objects in each image.
To accomplish this, run `reorder_result_general.py`.
The results will be saved in `ordered_results/`.
