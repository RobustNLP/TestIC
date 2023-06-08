import ast
import argparse
import os
import re
from tqdm import tqdm
import pdb
import json
import stanza
import nltk
import copy
import pickle as pk
from tsv_file import TSVFile, generate_lineidx_file
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from ofa_template_gen import Anno

# pdb.set_trace()

def check(ori_word, syn_list, cur_obj_set, keyword_in_sentence, hhyper):
    # syn_list consists of the words to be checked
    # hhyper determines whether to build hyper_list or not
    # print(syn_list)
    hyper_list = []
    flag = False # if flag==True:matched in the wordlist
    for syn in syn_list: # syn_list[0] = Synset('dog.n.01')
        if hhyper != 3:
            hyper_list.extend(wn.synset(syn.name()).hypernyms())
        if flag == True:
            continue
        for word in [lemma.name() for lemma in wn.synset(syn.name()).lemmas()]: #['dog', 'domestic_dog', 'Canis_familiaris']
            # pdb.set_trace()
            if  ( (word in single_list) or (word in plural_list) )  and (word in keyword_in_sentence):
                cur_obj_set.append((ori_word,word,hhyper))
                flag = True
                break
    return (flag,hyper_list)

def add_phrase(grammar, l, word_list):
    for t in l:
        compound_word = t[0][0] + '_' + t[1][0]
        if (compound_word in single_list) or (compound_word in plural_list):
            word_list.append(compound_word)
            if grammar == "NP: {<NN><NN>}" or grammar=="NP: {<NN><NNS>}":
                if t[0][0] in word_list: # this is to avoid <NN><NN><NN> phrase like: hot dog bowl
                    word_list.remove(t[0][0])
                if t[1][0] in word_list:
                    word_list.remove(t[1][0])
            else:
                if t[1][0] in word_list:
                    word_list.remove(t[1][0])
        else:
            if wn.synsets(compound_word) != []:
                word_list.append(compound_word)
            # 7_15/result-80-cats-all/group3 194 train track, train
            # for <NN><NN>  we only focus on the phrase and later noun
            if grammar == "NP: {<NN><NN>}" or grammar=="NP: {<NN><NNS>}":
                # pdb.set_trace()
                if t[0][0] in word_list:
                    word_list.remove(t[0][0]) # remove only remove 1 element
    return word_list

def parse(grammar, word_pos_list):
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(word_pos_list)
    l = []
    for tree in result.subtrees():
        if tree.label() == 'NP':
            l.append(tuple(tree))
    return l

def create_wordlist(str):
    doc = nlp(str)
    word_pos_list = [(word.text,word.xpos) for sent in doc.sentences for word in sent.words]
    word_list = []
    # putting all the noun and noun phrase into the word_list
    # noun
    # pdb.set_trace()
    for word in word_pos_list:
        if word[1] in ['NN','NNS']:
            word_list.append(word[0])          

    # noun phrase      
    for grammar in ["NP: {<JJ><NN>}","NP: {<JJ><NNS>}","NP: {<NN><NN>}","NP: {<NN><NNS>}"]:
        l = parse(grammar, word_pos_list)
        # pdb.set_trace()
        add_phrase(grammar, l, word_list)
    
    # print(word_list)
    return word_list


def check_word(word_list, keyword_in_sentence):
    # pdb.set_trace()
    cur_obj_set = []
    for word in word_list: #dog
        # in key_words lists
        if (word in single_list) or (word in plural_list):
            cur_obj_set.append((word,word,0))
        # synonyms & hypernyms & hyperhypernyms
        else:
            syn_list = wn.synsets(word, pos=wn.NOUN) #[Synset('dog.n.01'), Synset('frump.n.01'), Synset('dog.n.03'), Synset('cad.n.01'), Synset('frank.n.02'), Synset('pawl.n.01'), Synset('andiron.n.01')]
            # pdb.set_trace()
            # synonyms
            result = check(word, syn_list, cur_obj_set, keyword_in_sentence, 1)
            # cur_obj_set = result[0]
            if result[0] == True:
                continue
            # hypernyms
            # pdb.set_trace()
            result = check(word, result[1], cur_obj_set, keyword_in_sentence, 2)   
            # cur_obj_set = result[0]
            if result[0] == True:
                continue
            # hyperhypernyms
            # pdb.set_trace()
            result = check(word, result[1], cur_obj_set, keyword_in_sentence, 3)   
            # cur_obj_set = result[0]


    return cur_obj_set


def print_into_txt(tsv_name, cur_obj_set, obj_totally_deleted, flag):
    if flag:
        file_name = os.path.join('out', tsv_name,'ancestor.txt')
    else:
        file_name = os.path.join('out', tsv_name,'descendant.txt')
    with open(file_name, 'a') as f:
        for obj in cur_obj_set:
            f.write(str(obj)+' ')
        f.write('obj_td: '+ str(obj_totally_deleted))
        f.write('\n')


def read_obj_from_pkl(pkl_ls):
    out_set = set()
    for obj in pkl_ls:
        out_set.add('_'.join(obj.split('_')[:-1]))
    return out_set


def ls_to_standard_form_set(ls):
    out_set = []
    for obj in ls:
        if obj in keyword_list:
            out_set.append(obj)
        elif obj in single_list:
            idx = single_list.index(obj)
            out_set.append(keyword_list[idx])
        else:
            assert obj in plural_list
            idx = plural_list.index(obj)
            out_set.append(keyword_list[idx])
    return set(out_set)


def get_gt_obj_set(caption, keyword_in_sentence):
    obj_list = []
    word_list = create_wordlist(caption)
    # keyword_in_sentence = set()
    cur_obj_set = check_word(word_list, keyword_in_sentence)
    doc = nlp(caption)
    word_pos_list = [(word.text,word.xpos) for sent in doc.sentences for word in sent.words]
    word_text = [word.text for sent in doc.sentences for word in sent.words]
    word_pos = [word.xpos for sent in doc.sentences for word in sent.words]
    l = []
    for grammar in ["NP: {<JJ><NN>}","NP: {<JJ><NNS>}","NP: {<NN><NN>}","NP: {<NN><NNS>}"]:
        l.extend(parse(grammar, word_pos_list))
        #[(('remote', 'JJ'), ('control', 'NN'))]

   
    cap_set = ls_to_standard_form_set([tup[1] for tup in cur_obj_set])
    obj_list.append(cur_obj_set)
    return cap_set





    
def major_loop_body(tsv_name, cur_caption0, cur_caption1, deleted_classes, Ann, image_id):
    obj0_list = []
    obj1_list = []
    (ss_error, mr1_error, mr2_error) = (False, False, False)
    # flag = True

    # Rule12
    word_list_0 = create_wordlist(cur_caption0)
    word_list_1 = create_wordlist(cur_caption1)

    

    # preprocessing for word_list:
    # record the keyword in both list, only when the keyword are recorded, 
    # the search through synonyms or hypernyms is valid.
    # additional notice when doing preprocessing1: for object not in set, check its singular/plural form
    # e.g. group1 62 (person, people)
    # thought: 
    # in keyword_list:just match to singular form
    # not in keyword_list: nltk's tool change into singular form
    keyword_in_sentence = set()
    # pdb.set_trace()
    # if len( [obj for obj in obj_totally_deleted if obj in word_list_1] ) == 0:
    # remove all the same noun/noun phrase in the wordlist
    
    # add both the word's singular and plural form into keyword_in_sentence
    for word in copy.deepcopy(word_list_0): # won't delete the word in keywordList
        if word in single_list:
            keyword_in_sentence.add(word)
            idx = single_list.index(word)
            keyword_in_sentence.add(plural_list[idx])
        if word in plural_list:
            keyword_in_sentence.add(word)
            idx = plural_list.index(word)
            keyword_in_sentence.add(single_list[idx])

    for word in copy.deepcopy(word_list_1):
        if word in single_list:
            keyword_in_sentence.add(word)
            idx = single_list.index(word)
            keyword_in_sentence.add(plural_list[idx])
        if word in plural_list:
            keyword_in_sentence.add(word)
            idx = plural_list.index(word)
            keyword_in_sentence.add(single_list[idx])


    # Find the object list Rule3
    cur_obj_set0 = check_word(word_list_0, keyword_in_sentence)
    # cur_obj_set0 = list(cur_obj_set0)
    cur_obj_set1 = check_word(word_list_1, keyword_in_sentence)
    # cur_obj_set1 = list(cur_obj_set1)
    # cur_obj = cur_obj_set0+cur_obj_set1


    # find totally deleted obj       
    # parent_obj_set = check_word(word_list_0, keyword_in_sentence)
    # two list to store pos of nouns
    doc = nlp(cur_caption0)
    word_pos_list = [(word.text,word.xpos) for sent in doc.sentences for word in sent.words]
    word_text = [word.text for sent in doc.sentences for word in sent.words]
    word_pos = [word.xpos for sent in doc.sentences for word in sent.words]
    l = []
    for grammar in ["NP: {<JJ><NN>}","NP: {<JJ><NNS>}","NP: {<NN><NN>}","NP: {<NN><NNS>}"]:
        l.extend(parse(grammar, word_pos_list))
        #[(('remote', 'JJ'), ('control', 'NN'))]

    # two list to store pos of phrase
    # in stanza: "remote_controls" will be recognized as singular, so we just focus on the later noun
    t_word = []
    t_pos = []
    for t in l:
        t_word.append(t[0][0] + '_' + t[1][0])
        t_pos.append(t[1][1])
    obj_deleted_count = dict()
    obj_totally_deleted = []


    for obj in cur_obj_set0:
        # We change all the keywords into standard form to match the deleted_classes        
        word = ''
        if obj[1] in plural_list:
            idx = plural_list.index(obj[1])
            word = keyword_list[idx]
        elif obj[1] in single_list:
            idx = single_list.index(obj[1])
            word = keyword_list[idx]

        # find the Part-of-Speech
        # for noun phrase
        if (obj[0].find('_') != -1):
            idx = t_word.index(obj[0])
            pos = t_pos[idx]
        # for noun
        else:
            idx = word_text.index(obj[0])
            pos = word_pos[idx]


        # count the total number of one keyword
        if word in deleted_classes:
            if pos == 'NN':
                if word in obj_deleted_count.keys():
                    obj_deleted_count[word] += 1
                else:
                    obj_deleted_count[word] = 1
            elif pos == 'NNS':
                if word in obj_deleted_count.keys():
                    obj_deleted_count[word] += 2
                else:
                    obj_deleted_count[word] = 2
        
    for key in obj_deleted_count.keys():
        if obj_deleted_count[key] == 1:
            obj_totally_deleted.append(key)

    # For words in ancestor
    # pdb.set_trace()
    cap0_set = ls_to_standard_form_set([tup[1] for tup in cur_obj_set0])
    obj0_list.append(cur_obj_set0)
    print_into_txt(tsv_name, cur_obj_set0, obj_totally_deleted, True)
    # pdb.set_trace()

    # # For words in descendant
    cap1_set = ls_to_standard_form_set([tup[1] for tup in cur_obj_set1])
    obj1_list.append(cur_obj_set1)
    print_into_txt(tsv_name, cur_obj_set1, obj_totally_deleted, False)
    
    cap_set = cap0_set|cap1_set
    with open('gt_err.txt', 'a') as f :
        for gt_cap in Ann.gt_cap[image_id]:
            # pdb.set_trace()
            gt_set = get_gt_obj_set(gt_cap, keyword_in_sentence)
            if not cap_set.issubset(gt_set):
                print(image_id, gt_cap, cur_caption0, cur_caption1, sep='\t', file = f)

    
    # if flag:
    # test mrs
    if not mr1(cap0_set,cap1_set, deleted_classes):
        ss_error = True
        mr1_error = True
    if not mr2(cap0_set,cap1_set,obj_totally_deleted, deleted_classes):
        ss_error = True
        mr2_error = True
    # pdb.set_trace()
    return (ss_error, mr1_error, mr2_error)

def open_file(tsv_name):
    test_file_name = os.path.join(tsv_name + '.tsv')
    test_file_name_lineidx = os.path.join(tsv_name + '.lineidx')
    #generate_lineidx_file
    generate_lineidx_file(test_file_name, test_file_name_lineidx)
    # read tsv files
    p_f = TSVFile(test_file_name)
    return p_f


def iterating(tsv_name, p0_f, p1_f, dic):
    # initialize lists
    suspicious_list = []
    mr1_error_list = []
    mr2_error_list = []
    Ann = Anno()
    num_row = p0_f.num_rows()
    for idx in tqdm(range(num_row)):
        cur_caption0 = ast.literal_eval( p0_f.seek(idx)[1]) [0]['caption'].strip('.')
        cur_caption1 = ast.literal_eval( p1_f.seek(idx)[1]) [0]['caption'].strip('.')
        image_id = dic[int( p1_f.seek(idx)[0])]['image_id']

        ancestor_deleted = dic[idx+1]['removed_object_ancestor']
        descendant_deleted = dic[idx+1]['removed_object_descendant']
        # all in singular form
        deleted_classes = set()
        for classes in [obj_class for obj_class in descendant_deleted if obj_class not in ancestor_deleted]:
            obj_class = '_'.join(classes.split('_')[:-1])
            deleted_classes.add(obj_class)
        # pdb.set_trace()
        (ss_error, mr1_error, mr2_error) = major_loop_body(tsv_name, cur_caption0, cur_caption1, deleted_classes, Ann, image_id)
        if ss_error:
            suspicious_list.append(idx+1)
        if mr1_error:
            mr1_error_list.append(idx+1)
        if mr2_error:
            mr2_error_list.append(idx+1)

    out_file = open(os.path.join('out', tsv_name, 'report_issues'), 'w')
    print(suspicious_list, len(suspicious_list)/num_row, len(suspicious_list), num_row, len([obj for obj in suspicious_list if obj in dic_overlap[0.0]]),mr1_error_list, mr2_error_list, sep='\n', file = out_file)
    out_file.close()


def test():
    ancestor = input("please enter the ancestor caption\n").strip('.')
    descendant = input("please enter the descendant caption\n").strip('.')
    # obj_totally_deleted = input("please enter the obj_totally_deleted in list form, e.g. ['dog']\n")
    deleted_classes = input("please enter deleted_classes in list form, e.g. ['dog'], enter 's' to stop\n")
    while deleted_classes != 's':
        (ss_error, mr1_error, mr2_error) = major_loop_body('test', ancestor, descendant, deleted_classes)
        print('suspicious: ', ss_error) 
        print('mr1: ', mr1_error) 
        print('mr2: ', mr2_error) 
        ancestor = input("please enter the ancestor caption\n")
        descendant = input("please enter the descendant caption\n")
        # obj_totally_deleted = input("please enter the obj_totally_deleted in list form, e.g. ['dog']\n")
        deleted_classes = input("please enter deleted_classes in list form, e.g. ['dog'], enter 's' to stop\n")


def mr1(ancestor_cap, descendant_cap, deleted_classes):
    # pdb.set_trace()
    if not descendant_cap.issubset(ancestor_cap):
        return False
    return (ancestor_cap-descendant_cap).issubset(deleted_classes)


def mr2(ancestor_cap, descendant_cap, totally_deleted_obj, deleted_classes):
    if len(totally_deleted_obj)==0:
        return True

    expected_descendant_cap = copy.deepcopy(ancestor_cap)
    for obj in totally_deleted_obj:
        if obj in expected_descendant_cap:
            expected_descendant_cap.remove(obj)
    if not descendant_cap.issubset(expected_descendant_cap):
        return False
    return (ancestor_cap-descendant_cap).issubset(deleted_classes)
    

if __name__ == "__main__":
    nlp = stanza.Pipeline('en')
    wnl = WordNetLemmatizer()

    # tsv_names = ['oscar_base','oscar_large','showattend','vinvl_base','vinvl_large','ibm_captions', 'ofa_large', 'ofa_base']
    # tsv_names = ['ofa_base','showattend','vinvl_base']
    tsv_names = ['ofa_base']
    # tsv_names = ['pred.bottom-up-results.test.beam5.max20.odlabels']

    for tsv_name in tsv_names:
        # open files
        p0_f = open_file(tsv_name+'_ancestor')
        p1_f = open_file(tsv_name+'_descendant')

        # read removal information
        with open('name_img_id_dict.pkl','rb') as f:
            dic = pk.load(f)

        # read overlapping information
        with open('overlap_filtering_res_all.pkl','rb') as f:
            dic_overlap = pk.load(f)

        # key words
        # skis in both list because 7_13/group3_new20_2 20 skis!!
        single_list = ['cat', 'sheep', 'truck', 'bowl', 'airplane', 'giraffe', 'scissor', 'backpack', 'couch', 'cup', 'broccoli', 'person', 'kite', 'banana', 'bus', 'umbrella', 'chair', 'keyboard', 'bear', 'vase', 'handbag', 'microwave', 'snowboard', 'remote', 'cake', 'elephant', 'cow', 'motorcycle', 'sandwich', 'bottle', 'oven', 'boat', 'apple', 'car', 'laptop', 'zebra', 'bicycle', 'carrot', 'pizza', 'toilet', 'sink', 'bed', 'tie', 'book', 'horse', 'orange', 'bird', 'surfboard', 'suitcase', 'bench', 'dog', 'frisbee', 'refrigerator', 'skateboard', 'clock', 'train', 'spoon', 'fork', 'toothbrush', 'toaster', 'potted_plant', 'donut', 'dining_table', 'sports_ball', 'mouse', 'tennis_racket', 'fire_hydrant', 'baseball_glove', 'baseball_bat', 'cell_phone', 'knife', 'traffic_light', 'parking_meter', 'wine_glass', 'hair_drier', 'teddy_bear', 'ski', 'tv', 'stop_sign', 'hot_dog']
        plural_list = ['cats', 'sheep', 'trucks', 'bowls', 'airplanes', 'giraffes', 'scissors', 'backpacks', 'couches', 'cups', 'broccolis', 'people', 'kites', 'bananas', 'buses', 'umbrellas', 'chairs', 'keyboards', 'bears', 'vases', 'handbags', 'microwaves', 'snowboards', 'remotes', 'cakes', 'elephants', 'cows', 'motorcycles', 'sandwiches', 'bottles', 'ovens', 'boats', 'apples', 'cars', 'laptops', 'zebras', 'bicycles', 'carrots', 'pizzas','toilets', 'sinks', 'beds', 'ties', 'books', 'horses', 'oranges', 'birds', 'surfboards', 'suitcases', 'benches', 'dogs', 'frisbees', 'refrigerators', 'skateboards', 'clocks', 'trains', 'spoons', 'forks', 'toothbrushes', 'toasters', 'potted_plants', 'donuts', 'dining_tables', 'sports_balls', 'mice', 'tennis_rackets', 'fire_hydrants', 'baseball_gloves', 'baseball_bats', 'cell_phones', 'knives', 'traffic_lights', 'parking_meters', 'wine_glasses', 'hair_driers', 'teddy_bears', 'skis', 'tvs', 'stop_signs', 'hot_dogs']
        keyword_list = ['cat', 'sheep', 'truck', 'bowl', 'airplane', 'giraffe', 'scissors', 'backpack', 'couch', 'cup', 'broccoli', 'person', 'kite', 'banana', 'bus', 'umbrella', 'chair', 'keyboard', 'bear', 'vase', 'handbag', 'microwave', 'snowboard', 'remote', 'cake', 'elephant', 'cow', 'motorcycle', 'sandwich', 'bottle', 'oven', 'boat', 'apple', 'car', 'laptop', 'zebra', 'bicycle', 'carrot', 'pizza', 'toilet', 'sink', 'bed', 'tie', 'book', 'horse', 'orange', 'bird', 'surfboard', 'suitcase', 'bench', 'dog', 'frisbee', 'refrigerator', 'skateboard', 'clock', 'train', 'spoon', 'fork', 'toothbrush', 'toaster', 'potted_plant', 'donut', 'dining_table', 'sports_ball', 'mouse', 'tennis_racket', 'fire_hydrant', 'baseball_glove', 'baseball_bat', 'cell_phone', 'knife', 'traffic_light', 'parking_meter', 'wine_glass', 'hair_drier', 'teddy_bear', 'skis', 'tv', 'stop_sign', 'hot_dog']


        folder = os.getcwd()[:-4] + 'out//' + tsv_name + '//'
        if not os.path.exists(os.path.join('out', tsv_name)):
            os.makedirs(os.path.join('out', tsv_name))

        # iterating through the tsv files & print
        iterating(tsv_name, p0_f, p1_f, dic)

    # test()
        



    