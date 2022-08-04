# TestIC
A toolkit for testing image captioning [ISSTA'22]

If you use any of our tools or datasets in your research for publication, please kindly cite the following paper:
* Boxi Yu, Zhiqing Zhong, Xinran Qin, Jiayi Yao, Yuancheng Wang, and Pinjia He. 2022. Automated testing of image captioning systems. In Proceedings of the 31st ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA 2022). Association for Computing Machinery, New York, NY, USA, 467–479. https://doi.org/10.1145/3533767.3534389

## Environment requirements:
* Ubuntu 20.04.02 LTS
* Python 3.8
* Pytorch 1.7.1
* Stanza 1.2.3 (https://stanfordnlp.github.io/stanza)
* Pickle 
* Json
* CUDA 11.3
* CUDNN 8.05
  
## Structure of the code
The project includes code for our object inserting method, an implementation of MetaIC, data of the object source images we collect from [Flickr](https://www.flickr.com), the result of our manual review for erroneous issues, and a tool to report prospective errors in MS COCO Caption.

## Related code and links we use in this project:
* Yolact++ (https://github.com/dbolya/yolact).
* Microsoft Azure Cognitive Services (https://azure.microsoft.com/en-us/services/cognitive-services/).
* Oscar and VinVL (https://github.com/microsoft/Oscar).
* Show, Attend and Tell (https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning).
* Code to generate the region feature for Oscar (https://github.com/peteanderson80/bottom-up-attention).
* Code to generate the region feature for VinVL (https://github.com/microsoft/scene_graph_benchmark).

## Object inserting method
The implementation of our object inserting method is in Insert_code, please refer to [Insert_Code\README.md](Insert_code\README.md) to get the information of how to use the code.

## Object source images
We collect a pool of object source images on [flickr](https://www.flickr.com), containing 60 categories (60 out of 80) of MS COCO Caption. Users can use Yolact++ to perform the segmentation on our collected [dataset](Object_Source_Images) and obtain high-quality standalone object images. Users can also download their own dataset on Flickr. 

## Report erroneous issues
This directory corresponds to the implementation of MetaIC to report suspicious issues of the 1 paid IC API and 5 IC models. Each directory contains:
* An implementation of MetaIC to check the MRs we propose in the paper.
* The captions produced by the IC we tested on the background image of the test dataset of MS COCO Caption, and our synthetic images with various overlapping ratios.
* The reported suspicious issues produced by MetaIC.
* Example(report suspicious issues for Microsoft Azure Cognitive Servicses with $ratio_0$):
  ```bash
    # Start to report the suspicious issues of Microsoft Azure Cognitive Services:
    cd Report_erroneous_issues/azure_meta_relation_error_report_dir
    python meta_relation_error_report_0.py 
  ```
