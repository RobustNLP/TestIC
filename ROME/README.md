# Artifact of ROME
This artifact is the package for ROME along with the video tutorials for reproducing the experiments, which includes the following content (due to file size limitation, we have uploaded the metadata to zenodo: https://zenodo.org/record/7980997):
## Content Description
- Video tutorials 
  - In Tutorial_1, we demonstrate how to perform object selection, image mutation, and fine-tuning. 
  - In Tutorial_2, we demonstrate how to do error detection with ROME.
  - In Tutorial_3, we demonstrate how to analyze our results of the user study, how to find labeling errors in the MS COCO Caption dataset, how to generate test cases, and how to detect captioning errors with MetaIC.
  - In Tutorial_4, we demonstrate how to use the images that triggered bugs in the IC systems under test for testing the real-world IC systems, i.e., Microsoft Powerpoint and Facebook Automatic Alternative Text.
- object_selection \
    This repository contains the codes used for the first step in ROME, which is object selection.
    For more details, please refer to [object_selection.readme](object_selection/README.md).
- image_mutation \
    This repository contains the codes used to melt objects in images and generate new images. 
    For more details, please refer to [image_mutation.readme](image_mutation/README.md).
- inference_scripts \
    This folder provides scripts for caption generation using different captioning systems.
    For more details, please refer to [inference_scripts.readme](inference_scripts/README.md).
- error_detection \
    After obtaining captions for synthetic images, this folder provides the codes and materials for error detection based on two metamorphic relations.
    For more details, please refer to [error_detection.readme](error_detection/README.md).
- finetuning \
    This folder contains the re-labeled annotations for 1,000 synthesized images as well as scripts for finetuning and checkpoint file. 
    For more details, please refer to [finetuning.readme](finetuning/README.md).
- naturalness \
    This folder contains codes and results of the user study on image naturalness.
    For more details, please refer to [naturlness.readme](naturalness/README.md).
- gt_err \
    This folder contains codes, suspicious issues, and results for finding labeling errors in the MS COCO Caption dataset. 
    For more details, please refer to [gt_err.readme](gt_err/README.md).
- metaic_insertion \
    This folder contains codes for synthesizing test cases with MetaIC.
    For more details, please refer to [metaic_insertion.readme](metaic_insertion/README.md).
- metaic_error_detection \
    This folder contains codes for detecting captioning errors with MetaIC.
    For more details, please refer to [metaic_error_detection.readme](metaic_error_detection/README.md).
- ms&face_Exp \
  This folder contains image test cases for testing the real-world IC systems [metaic_error_detection.readme](ms&face_Exp/README.md).

