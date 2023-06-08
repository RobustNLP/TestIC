import json 
import pickle as pk
from tsv_file import TSVFile
import stanza
from tqdm import tqdm

# initialize function of stanza to analyze the sentece
nlp = stanza.Pipeline('en')

# 60 classes
single_list = ['cat', 'sheep', 'truck', 'bowl', 'airplane', 'giraffe', 'scissor', 'backpack', 'couch', 'cup', 'broccoli', 'person', 'kite', 'banana', 'bus', 'umbrella', 'chair', 'keyboard', 'bear', 'vase', 'handbag', 'microwave', 'snowboard', 'remote', 'cake', 'elephant', 'cow', 'motorcycle', 'sandwich', 'bottle', 'oven', 'boat', 'apple', 'car', 'laptop', 'zebra', 'bicycle', 'carrot', 'pizza', 'toilet', 'sink', 'bed', 'tie', 'book', 'horse', 'orange', 'bird', 'surfboard', 'suitcase', 'bench', 'dog', 'frisbee', 'refrigerator', 'skateboard', 'clock', 'train', 'spoon', 'fork', 'toothbrush', 'toaster', 'potted plant', 'donut', 'dining table', 'sports ball', 'mouse', 'tennis racket', 'fire hydrant', 'baseball glove', 'baseball_bat', 'cell phone', 'knife', 'traffic light', 'parking meter', 'wine glass', 'hair drier', 'teddy bear', 'ski', 'tv', 'stop sign', 'hot dog']
plural_list = ['cats', 'sheep', 'trucks', 'bowls', 'airplanes', 'giraffes', 'scissors', 'backpacks', 'couches', 'cups', 'broccolis', 'people', 'kites', 'bananas', 'buses', 'umbrellas', 'chairs', 'keyboards', 'bears', 'vases', 'handbags', 'microwaves', 'snowboards', 'remotes', 'cakes', 'elephants', 'cows', 'motorcycles', 'sandwiches', 'bottles', 'ovens', 'boats', 'apples', 'cars', 'laptops', 'zebras', 'bicycles', 'carrots', 'pizzas','toilets', 'sinks', 'beds', 'ties', 'books', 'horses', 'oranges', 'birds', 'surfboards', 'suitcases', 'benches', 'dogs', 'frisbees', 'refrigerators', 'skateboards', 'clocks', 'trains', 'spoons', 'forks', 'toothbrushes', 'toasters', 'potted plants', 'donuts', 'dining tables', 'sports balls', 'mice', 'tennis rackets', 'fire hydrants', 'baseball gloves', 'baseball bats', 'cell phones', 'knives', 'traffic lights', 'parking meters', 'wine glasses', 'hair driers', 'teddy bears', 'skis', 'tvs', 'stop signs', 'hot dogs']

single_set = set(single_list)
plural_set = set(plural_list)

sin2plu, plu2sin = dict(), dict()

# create the dict for sin2plu, plu2sin
for idx in range( 0, len(single_list) ):
    # print( single_list[idx], plural_list[idx] )
    sin2plu[single_list[idx]] = plural_list[idx]
    plu2sin[plural_list[idx]] = single_list[idx]

#  load the groundTruth for coco_caption_test
# with open('./test_caption.json', 'r') as f:
#     data = f.readline()

#load the meta captions 
test_dir = 'ofa_base_1000_bar2'
bg_dir = 'ofa_base_1000_bg'
import os
for fname in os.listdir(test_dir):
    if '.tsv' in fname:
        test_file_name = os.path.join(test_dir, fname)

for fname in os.listdir(bg_dir):
    if '.tsv' in fname:
        bg_file_name = os.path.join(bg_dir, fname)
with open(test_file_name, 'r') as preds_file:
    preds = preds_file.readlines()

# load the prediction for meta_ic_cap with inserted obj and original cap
meta_tsv = TSVFile(test_file_name)
bg_tsv = TSVFile(bg_file_name)
# meta_tsv = TSVFile('ana_oscar_3280/pred.oscar_3280.test.beam5.max20.odlabels.tsv')
# ori_tsv = TSVFile('ori_ofa_cap/pred.val_test_5000.tsv')
# set of id which we choose as background image from coco_caption_test
pred_id_set = set()

with open('ofa_base_1000_bar2/name_img_id_dict.pkl', 'rb') as f:
    image_id_dict = pk.load(f)

# construct the image_id set we need to compare 
cnt_id_set = 1
for line in preds:
    pred_id_set.add(str( image_id_dict[cnt_id_set]['image_id'] ))
    # print(cnt)
    cnt_id_set+=1

image_keys = list(pred_id_set)
# image_keys = ['25989', '572055']

key2captions = {key: [] for key in image_keys} 
# captions = json.loads(data)

# for cap in captions:
#     if cap['image_id'] in set(image_keys):
#         key2captions[cap['image_id']].append(cap['caption'])

# list of error example against the meta_ic_rule
error_report_list = []
first_error_report_list = []
second_error_report_list = []

# ori_cap_dict = dict()
# for idx in range( ori_tsv.num_rows() ):
#     ori_cap_dict[ ori_tsv.seek(idx)[0] ] =  json.loads( ori_tsv.seek(idx)[1] )[0]['caption']

# cnt is the number to control the row id
cnt = 1
for line in tqdm(preds):
    # row id of seek starts with 0, but image_id_dict starts from 1,
    # thus we use cnt-1
    current_meta = json.loads( meta_tsv.seek(cnt-1)[1] )[0]['caption']
    # current_gt = key2captions[ str(image_id_dict[cnt]['image_id']) ]
    # current_ori_cap = ori_cap_dict[ str(image_id_dict[cnt]['image_id']) ]
    current_ori_cap = json.loads( bg_tsv.seek(cnt-1)[1])[0]['caption']
    current_insert_obj = image_id_dict[cnt]['object']
    current_insert_obj = current_insert_obj.replace('_', ' ')
    
    ana_current_meta = nlp(current_meta)
    ana_current_ori_cap = nlp(current_ori_cap)

    tmp_sin_list_meta = []
    tmp_plu_list_meta = []

    tmp_sin_list_ori = []
    tmp_plu_list_ori = []

    # preperty of words in two captions
    tmp_word_property_meta = dict()
    tmp_word_property_ori = dict()
    # frequency of word in meta and ori sentences 
    tmp_word_frequency_meta = dict()
    tmp_word_frequency_ori = dict()

    # if cnt==24:
        # pdb.set_trace()
    for sent in ana_current_meta.sentences:
        for word in sent.words:
            if word.xpos == 'NN' or word.xpos == 'NNS':
                # judge the property of word
                if word.text.lower() in single_list:
                    # if word.text.lower() =='orange':
                    #     pdb.set_trace()
                    tmp_sin_list_meta.append(word.text.lower())
                    # asssign the property of word in set of 60 classes
                    if word.text.lower() in tmp_word_property_meta.keys():
                        tmp_word_property_meta[word.text.lower()] = 'NNS'
                    else:
                        tmp_word_property_meta[word.text.lower()] = word.xpos
                    if word.text.lower() not in tmp_word_frequency_meta:
                        tmp_word_frequency_meta[word.text.lower()] = 1
                    else:
                        tmp_word_frequency_meta[word.text.lower()] += 1                      
                elif word.text.lower() in plural_list:
                    tmp_plu_list_meta.append(word.text.lower())
                    # asssign the property of word in set of 60 classes
                    tmp_word_property_meta[ plu2sin[word.text.lower()] ] = word.xpos
            else:
                continue

    for sent in ana_current_ori_cap.sentences:
        for word in sent.words:
            if word.xpos == 'NN' or word.xpos == 'NNS':
                if word.text.lower() in single_list:
                    tmp_sin_list_ori.append(word.text.lower())
                    # asssign the property of word in set of 60 classes
                    if word.text.lower() in tmp_word_property_ori.keys():
                        tmp_word_property_ori[word.text.lower()] = 'NNS'
                    else:
                        tmp_word_property_ori[word.text.lower()] = word.xpos
                    if word.text.lower() not in tmp_word_frequency_ori:
                        tmp_word_frequency_ori[word.text.lower()] = 1
                    else:
                        tmp_word_frequency_ori[word.text.lower()] += 1                      
                elif word.text.lower() in plural_list:
                    tmp_plu_list_ori.append(word.text.lower())
                    # asssign the property of word in set of 60 classes
                    tmp_word_property_ori[ plu2sin[word.text.lower()] ] = word.xpos
            else:
                continue
    
    # When we assign the properties of word, we use the single form of word to
    # assign them
    ori_class_set = set()
    meta_class_set = set()
    for itm in tmp_sin_list_ori:
        ori_class_set.add(itm)
    for item in tmp_plu_list_ori:
        ori_class_set.add( plu2sin[item] )
    for itm in tmp_sin_list_meta:
        meta_class_set.add(itm)
    for item in tmp_plu_list_meta:
        meta_class_set.add( plu2sin[item] )
    # we insert an obj so the meat_class_set should contain the new obj cls
    
    # metamorphic relation one: the rule of class
    # if image_id_dict[cnt][1] != meta_class_set or not( ori_class_set.issubset(meta_class_set) ):
    #     error_report_list.append(cnt)
    ori_plus_isrtobj = ori_class_set.copy()
    ori_plus_isrtobj.add( image_id_dict[cnt]['object'] )
    
    if not( ori_plus_isrtobj == meta_class_set ):
        # pdb.set_trace()
        error_report_list.append(cnt)
        first_error_report_list.append(cnt)
    else:
        # The condition = False means that bad condition doesn't happen
        condition1 = False
        condition2 = False
        condition3 = False 
        condition4 = False
        if current_insert_obj in ori_class_set:
            if tmp_word_property_meta[ current_insert_obj ] == 'NN':
                condition1 = True # if the inserted obj in the ori class set and its form is singular, report errors
            for key in tmp_word_property_meta:
                if key != current_insert_obj and (tmp_word_property_meta[key] != tmp_word_property_ori[key]):
                    condition2 = True # besides the inserted obj, any changes of singular-plural form of the ori class set, report errors
        else:
            if tmp_word_property_meta[ current_insert_obj ] == 'NNS':
                condition3 = True # if the obj not in the ori class set, and its form is plural, report error
            for key in tmp_word_property_meta:
                if key != current_insert_obj and (tmp_word_property_meta[ key ] != tmp_word_property_ori[key]):
                    condition4 = True #  besides the inserted obj, any changes of singular-plural form of the ori class set, report errors
        
        if condition1 or condition2 or condition3 or condition4:
            # pdb.set_trace()
            error_report_list.append(cnt)
            second_error_report_list.append(cnt)

    cnt += 1

print(0)

with open( test_file_name.split('/')[0] + '_report_error_list.txt', 'w') as writefile:
    writefile.write(str(error_report_list) + '\n')
    writefile.write( str( len(error_report_list) / (cnt-1) ) + '\n'  )