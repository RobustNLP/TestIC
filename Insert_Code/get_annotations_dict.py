import json

def get_area(x):
    return (x[3]-x[1])*(x[2]-x[0])

with open('./coco_dataset_annotation/coco_annotations_trainval2014/annotations/instances_val2014.json') as fs:
    fstr = fs.readline()
    f = json.loads(fstr)

print('0')

# rect = list()
# for i in range(5):
#     rect.append(0)

cat = list()
for i in range(0,92):
    cat.append('')
for i in range(len(f['categories'])):
    cat[f['categories'][i]['id']] = f['categories'][i]['name']

with open('./dataset_split/caption_datasets/dataset_coco.json') as fs2:
    fstr2 = fs2.readline()
    f2 = json.loads(fstr2)

mis_log = open('./missing_log', mode = 'a', encoding = 'utf-8')
mis_log.seek(0)
mis_log.truncate()

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
        # rect[4] = i
        test_id[f['annotations'][i]['image_id']].append(rect)
        # if f['annotations'][i]['image_id'] == 391895:
        #     print(rect,test_id[f['annotations'][i]['image_id']])
mis = 0
for i in range(len(test_set)):
    test_id[test_set[i]] = sorted(test_id[test_set[i]], key= lambda x:((x[2]-x[0])*(x[3]-x[1])), reverse = True)
    count = 1
    if(len(test_id[test_set[i]])) == 0:
        # print('hahahhahah', test_set[i])
        print('image_id: ', test_set[i], file=mis_log)
        mis += 1
        continue
    # main_area = get_area(test_id[test_set[i]][0])
    # for j in range(1,len(test_id[test_set[i]])):
    #     # you can add constraints on count, which mean total items of label
    #     if get_area(test_id[test_set[i]][j])< main_area*0.1 :
    #         del test_id[test_set[i]][j:]
    #         break
    #     count += 1
    # if i< 5:
    #     print(test_set[i],' : ')
    #     for j in range(len(test_id[test_set[i]])):
    #         # print((test_id[test_set[i]][j][3]-test_id[test_set[i]][j][1])*(test_id[test_set[i]][j][2]-test_id[test_set[i]][j][0]) , end=', ')
    #         print(get_area(test_id[test_set[i]][j]), end=', ')
    #     print('\n')
        # print(test_set[i],':',test_id[test_set[i]])

import pickle 
pickle.dump(test_id, open("./image_dict.pkl", 'wb'))

# print('total_miss: ', mis, file=mis_log)