# User Study
We recruited 24 participants on Prolific, a platform for posting tasks and hiring workers, to rate the naturalness score of images. 
The images were divided into four groups: raw images from the MS COCO Caption dataset, synthesized images from ROME, MetaIC-NO, and MetaIC-PO. 
Each group consists of 1,000 images, resulting in a total of 4,000 images.
Then we divided 4,000 images into 8 sessions, and assigned three participants to each session.
Detailed instructions can be found in the provided Statement.pdf.
## Label
To determine which group a specific image belongs to, please refer to the `session_fpath_2_ori_fpath.pkl` file.
Specifically, if we want to find the group of `session_1/1.png` belongs to, we can follow the Python scripts below:
```python
>>> import pickle as pk
>>> with open("session_fpath_2_ori_fpath.pkl","rb") as f:
...     dic = pk.load(f)
>>> dic["res/session_1/1.png"]
"ori/final_result_test_0/180.png"
```
The result indicates that the image is from `final_result_test_0/`, which corresponds to MetaIC-NO.
If the image is from `final_result_test_bar2/`, it corresponds to MetaIC-PO.
If the image is from `ori_1k/`, it corresponds to the MS COCO Caption dataset.
If the image is from `azure_1k/`, it corresponds to ROME.
## Results
The results are recorded in the `results/` folder. To see the overall results, use the following script:
```shell
python process_session_results.py
```
To calculate icc score, you will need pingouin==0.5.3 and pandas==1.3.0. 
Once the environment is set up, execute the script below:
```shell
python icc.py
```

