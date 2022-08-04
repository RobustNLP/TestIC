cd ./coco_dataset_annotation/coco_annotations_trainval2014
sh download.sh
cd ../../dataset_split/caption_datasets
sh download.sh
cd ../../
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip