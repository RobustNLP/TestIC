# Labeling Errors
In this module we will show you how to find labeling errors in MS COCO Caption dataset.
## Environment Setup
The environment here is exactly what it is in Error Detection module, please refer to [error_detection.readme](../error_detection/README.md)
## Finding suspicious issues
To find potential errors in MS COCO Caption dataset, just run `synonym_final.py`. 
The issues will be recorded to `gt_error.txt`.
Here we give explanation of the contents in `gt_err.txt`. 
Let's take an example:
```
393685	a person petting a dog laying on its back	a dog and a cat laying on a bed	a black and white cat laying on top of a bed
```
Each content is seperated by a tab. 
The first content denotes the current image id.
And the following contents denote the original caption, and the captions from ofa.
The issue reports that the original caption of image id 393685 may contain error.
The original caption is `a person petting a dog laying on its back`.
We use ofa to generate captions for the same image and the image after obejct melting, then obtained `a dog and a cat laying on a bed` and `a black and white cat laying on top of a bed`, which is quite different from the original caption from MS COCO Caption dataset.
We manually inspected the image and verified that this is an error occurs in MS COCO Caption dataset.

We manually inspected some of the suspicious issues and found 219 labeling errors in the dataset. More information is provided in `MSCOCO_Labeling_Errors.txt`.
