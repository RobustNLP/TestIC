import json
import os

def get_area(x):
    return (x[3]-x[1])*(x[2]-x[0])

with open('./coco_dataset_annotation/coco_annotations_trainval2014/annotations/instances_val2014.json') as fs:
    fstr = fs.readline()
    f = json.loads(fstr)

# print('0')

# os.system('rm ./chosen_bg_images_all/*')

# rect = list()
# for i in range(5):
#     rect.append(0)

# get categories from annotations(80)
cat = list()
for i in range(0,92):
    cat.append('')
for i in range(len(f['categories'])):
    cat[f['categories'][i]['id']] = f['categories'][i]['name']
    # print(f['categories'][i]['name'])

with open('./dataset_split/caption_datasets/dataset_coco.json') as fs2:
    fstr2 = fs2.readline()
    f2 = json.loads(fstr2)

# mis_log = open('./missing_log', mode = 'a', encoding = 'utf-8')
# mis_log.seek(0)
# mis_log.truncate()
single_list = ['cat', 'sheep', 'truck', 'bowl', 'airplane', 'giraffe', 'scissors', 'backpack', 'couch', 'cup', 'broccoli', 'person', 'kite', 'banana', 'bus', 'umbrella', 'chair', 'keyboard', 'bear', 'vase', 'handbag', 'microwave', 'snowboard', 'remote', 'cake', 'elephant', 'cow', 'motorcycle', 'sandwich', 'bottle', 'oven', 'boat', 'apple', 'car', 'laptop', 'zebra', 'bicycle', 'carrot', 'pizza', 'toilet', 'sink', 'bed', 'tie', 'book', 'horse', 'orange', 'bird', 'surfboard', 'suitcase', 'bench', 'dog', 'frisbee', 'refrigerator', 'skateboard', 'clock', 'train', 'spoon', 'fork', 'toaster', 'toothbrush']

# test_set stores target image_id and test_id stores rect
test_set = list()
test_id = dict() 
for i in range(len(f2['images'])):
    if f2['images'][i]['split'] == 'test':
        cid = f2['images'][i]['cocoid']
        test_set.append(cid)
        test_id[cid] = list()

# rect =list()  
# print(test_id[test_set[0]])
for i in range(len(f['annotations'])):
    if f['annotations'][i]['image_id'] in test_id.keys():
        rect = list()
        rect.append(f['annotations'][i]['bbox'][0]) 
        rect.append(f['annotations'][i]['bbox'][1]) 
        rect.append(f['annotations'][i]['bbox'][0]+f['annotations'][i]['bbox'][2])
        rect.append(f['annotations'][i]['bbox'][1]+f['annotations'][i]['bbox'][3]) 
        rect.append(f['annotations'][i]['category_id'])
        # rect[4] = i
        test_id[f['annotations'][i]['image_id']].append(rect)
        # if f['annotations'][i]['image_id'] == 391895:
        #     print(rect,test_id[f['annotations'][i]['image_id']])
# mis = 0

print(len(test_set))
cnt2 = 0
for i in range(len(test_set)):
    # iid refers to image_id
    iid = test_set[i]
    # cnt means the number of class that involve in single_list (56)
    cnt = 0
    for j in range(len(test_id[iid])):
        # cid refers to category_id
        cid = test_id[iid][j][4]
        cstr = cat[cid]
        if cstr in single_list:
            cnt += 1
            # if cstr == 'skis':
            #     print('iid',cstr)
            #     import pdb
            #     pdb.set_trace()
            continue
        else:
            cnt = 0
            break
    if cnt != 0:
        cnt2 += 1
        # for j in range(len(test_id[iid])):
        #     cstr = cat[cid]
        #     if iid =:
        #         print(iid,cstr)
        # if iid == 496575:
        #     print(iid)
        os.system('cp ./val2014/COCO_val2014_'+str(iid).zfill(12)+'.jpg ./chosen_bg_images_all')
        # if cnt2<= 100:
        #     os.system('cp ./val2014/COCO_val2014_'+str(iid).zfill(12)+'.jpg ./chosen_bg_images2')
        # test_set[i] = -1

# cnt = 0
# for i in range(len(test_set)):
#     if test_set[i]!= -1 :
#         cnt += 1
# print(cnt)
# import pickle 
# pickle.dump(test_id, open("./insert_objects/image_dict2.pkl", 'wb'))
# print('total_miss: ', mis, file=mis_log)