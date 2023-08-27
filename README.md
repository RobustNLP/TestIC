# TestIC
A toolkit for testing image captioning [ISSTA'22] [ISSTA'23]

### Structure of the repository
The project includes the implementation of two IC Testing tools, i.e., the additive testing tool (MetaIC), and subtractive testing tool (ROME). For the usage of these tools, you can refer to the corresponding README file.

### Environment requirements:
* Ubuntu 20.04.02 LTS
* Python 3.8
* Pytorch 1.7.1
* Stanza 1.2.3 (https://stanfordnlp.github.io/stanza)
* Pickle 
* Json
* CUDA 11.3
* CUDNN 8.05

### Image Captioning Errors detected by MetaIC:

<img src=assets/MetaICError.png style="width:80%;height:auto;">


### Image Captioning Errors detected by ROME:

<img src=assets/ROMEError.png style="width:80%;height:auto;">

### Labeling errors found in the MS COCO Caption dataset:
You can search for the images with the corresponding Image-ID in the Website of MS COCO (https://cocodataset.org/#explore).
We hope that MS COCO team can fix the labeling errors.

<img src=assets/ROMELBError.png style="width:65%;height:auto;">

### Citation
If you use any of our tools or datasets in your research for publication, please kindly cite the following paper:
* Boxi Yu, Zhiqing Zhong, Xinran Qin, Jiayi Yao, Yuancheng Wang, and Pinjia He. 2022. Automated testing of image captioning systems. In Proceedings of the 31st ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA 2022). Association for Computing Machinery, New York, NY, USA, 467–479. https://doi.org/10.1145/3533767.3534389

* Boxi Yu, Zhiqing Zhong, Jiaqi Li, Yixing Yang, Shilin He, and Pinjia He. 2023. ROME: Testing Image Captioning Systems via Recursive Object Melting . In Proceedings of the 32nd ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA ’23), July 17–21, 2023, Seattle, WA, United States. ACM, New York, NY, USA, 13 pages. https://doi.org/10.1145/3597926.3598094 

### Feedback
Should you have any question, please post to [the issue page](https://github.com/RobustNLP/TestIC/issues), or email Boxi Yu via boxiyu@link.cuhk.edu.cn.
