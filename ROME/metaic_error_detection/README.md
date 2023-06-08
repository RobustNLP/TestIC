# Error Detection in MetaIC
To detect errors, MetaIC compares the captions of background images and the synthesized images.
## Files
Here we take captions from ofa_base as example.
In `ofa_base_meta/`, you will find captions for the background images in `ofa_base_1000_bg/`, captions for synthesized images without overlap in `ofa_base_1000_0/`, and captions for synthesized images with an overlap ratio between 15% and 30% in `ofa_base_1000_bar2/`.
## Environment
The same environment as described in [error_detection](../error_detection/README.md) should be used..
## Suspicious issues
To generate a list of suspicious issues, please execute `meta_relation_error_report_0.py` and `meta_relation_error_report_bar2.py`.
The results will be saved in `ofa_base_1000_0_report_error_list.txt` and `ofa_base_1000_bar2_report_error_list.txt`.
