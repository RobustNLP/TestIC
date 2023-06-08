import ast
import os
from tqdm import tqdm
import pdb
import stanza # stanza=='1.4.0'
import nltk   # nltk=='3.6.1'
import copy
import pickle as pk
from tsv_file import TSVFile, generate_lineidx_file
from nltk.corpus import wordnet as wn


def check(ori_word, syn_list, cur_obj_set, keyword_in_sentence, hhyper):
    '''
    This function checks whether the denotions of synset in synsets contains keyword
    @param syn_list: consists of the words to be checked
    @param hyper_list: consists of hypernyms of the syn_list
    @param hhyper: determines whether to build hyper_list or not
    '''
    hyper_list = []
    flag = False # if flag==True: matched in the wordlist
    for syn in syn_list: # syn_list[0] = Synset('dog.n.01')
        if hhyper != 3:
            hyper_list.extend(wn.synset(syn.name()).hypernyms())
        if flag == True:
            continue
        for word in [lemma.name() for lemma in wn.synset(syn.name()).lemmas()]: #['dog', 'domestic_dog', 'Canis_familiaris']
            if  ( (word in single_list) or (word in plural_list) )  and (word in keyword_in_sentence):
                cur_obj_set.append((ori_word,word,hhyper))
                flag = True
                break
    return (flag,hyper_list)

def add_phrase(grammar, l, word_list):
    '''
    This function add noun phrases into word_list
    @param word_list: a list contains nouns and noun phrases appeared in caption
    '''
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
            # for <NN><NN>  we only focus on the phrase and later noun
            if grammar == "NP: {<NN><NN>}" or grammar=="NP: {<NN><NNS>}":
                if t[0][0] in word_list:
                    word_list.remove(t[0][0])
    return word_list

def parse(grammar, word_pos_list):
    '''This function generates noun phrase according to certain grammar'''
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(word_pos_list)
    l = []
    for tree in result.subtrees():
        if tree.label() == 'NP':
            l.append(tuple(tree))
    return l

def create_wordlist(str):
    '''
    This function create a word_list to store nouns and noun phrases
    @param word_list: a list contains nouns and noun phrases appeared in caption
    '''
    doc = nlp(str)
    word_pos_list = [(word.text,word.xpos) for sent in doc.sentences for word in sent.words]
    word_list = []
    # putting all the noun and noun phrase into the word_list
    # noun
    for word in word_pos_list:
        if word[1] in ['NN','NNS']:
            word_list.append(word[0])          

    # noun phrase      
    for grammar in ["NP: {<JJ><NN>}","NP: {<JJ><NNS>}","NP: {<NN><NN>}","NP: {<NN><NNS>}"]:
        l = parse(grammar, word_pos_list)
        add_phrase(grammar, l, word_list)
    
    return word_list


def check_word(word_list, keyword_in_sentence):
    '''
    This function check the word meaning level by level
    check its original word, synonyms, hypernyms, hypernyms' hypernyms
    '''
    cur_obj_set = []
    for word in word_list: 
        # in key_words lists
        if (word in single_list) or (word in plural_list):
            cur_obj_set.append((word,word,0))
        # synonyms & hypernyms & hyperhypernyms
        else:
            syn_list = wn.synsets(word, pos=wn.NOUN) #[Synset('dog.n.01'), Synset('frump.n.01'), Synset('dog.n.03'), Synset('cad.n.01'), Synset('frank.n.02'), Synset('pawl.n.01'), Synset('andiron.n.01')]
            # synonyms
            result = check(word, syn_list, cur_obj_set, keyword_in_sentence, 1)
            if result[0] == True:
                continue
            # hypernyms
            result = check(word, result[1], cur_obj_set, keyword_in_sentence, 2)   
            if result[0] == True:
                continue
            # hyperhypernyms
            result = check(word, result[1], cur_obj_set, keyword_in_sentence, 3)   

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


def major_loop_body(tsv_name, cur_caption0, cur_caption1, deleted_classes):
    '''
    This function is the main processing step of every caption pairs
    '''
    obj0_list = []
    obj1_list = []
    (ss_error, mr1_error, mr2_error) = (False, False, False)

    # Rule1&2 implemented in create_wordlist
    word_list_0 = create_wordlist(cur_caption0)
    word_list_1 = create_wordlist(cur_caption1)


    # preprocessing for word_list:
    # record the keyword in both list,(add both the word's singular and plural form into keyword_in_sentence)
    # only when the keyword are recorded, the search through synonyms or hypernyms is valid.
    keyword_in_sentence = set()
    
    for word in copy.deepcopy(word_list_0): 
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


    # Rule3 implemented in check_word
    cur_obj_set0 = check_word(word_list_0, keyword_in_sentence)
    cur_obj_set1 = check_word(word_list_1, keyword_in_sentence)


    # find totally deleted obj       
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

    # words in ancestor
    cap0_set = ls_to_standard_form_set([tup[1] for tup in cur_obj_set0])
    obj0_list.append(cur_obj_set0)
    print_into_txt(tsv_name, cur_obj_set0, obj_totally_deleted, True)

    # words in descendant
    cap1_set = ls_to_standard_form_set([tup[1] for tup in cur_obj_set1])
    obj1_list.append(cur_obj_set1)
    print_into_txt(tsv_name, cur_obj_set1, obj_totally_deleted, False)
    

    if not mr1(cap0_set,cap1_set, deleted_classes):
        ss_error = True
        mr1_error = True
    if not mr2(cap0_set,cap1_set,obj_totally_deleted, deleted_classes):
        ss_error = True
        mr2_error = True
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
    '''
    This function iterates through the caption pairs
    '''
    # initialize lists
    suspicious_list = []
    mr1_error_list = []
    mr2_error_list = []

    num_row = p0_f.num_rows()
    for idx in tqdm(range(num_row)):
        cur_caption0 = ast.literal_eval( p0_f.seek(idx)[1]) [0]['caption'].strip('.')
        cur_caption1 = ast.literal_eval( p1_f.seek(idx)[1]) [0]['caption'].strip('.')

        ancestor_deleted = dic[idx+1]['removed_object_ancestor']
        descendant_deleted = dic[idx+1]['removed_object_descendant']
        # find the deleted object classes
        deleted_classes = set()
        for classes in [obj_class for obj_class in descendant_deleted if obj_class not in ancestor_deleted]:
            obj_class = '_'.join(classes.split('_')[:-1])
            deleted_classes.add(obj_class)

        (ss_error, mr1_error, mr2_error) = major_loop_body(tsv_name, cur_caption0, cur_caption1, deleted_classes)
        if ss_error:
            suspicious_list.append(idx+1)
        if mr1_error:
            mr1_error_list.append(idx+1)
        if mr2_error:
            mr2_error_list.append(idx+1)

    out_file = open(os.path.join('out', tsv_name, 'report_issues'), 'w')
    print(suspicious_list, len(suspicious_list)/num_row, len(suspicious_list), num_row,mr1_error_list, mr2_error_list, sep='\n', file = out_file)
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
    return True
    

if __name__ == "__main__":
    nlp = stanza.Pipeline('en')

    tsv_names = ['ofa_base','oscar_base','showattend','vinvl_base','azure']

    for tsv_name in tsv_names:
        # open files
        p0_f = open_file(tsv_name+'_ancestor')
        p1_f = open_file(tsv_name+'_descendant')

        # read removal information
        with open('name_img_id_dict.pkl','rb') as f:
            dic = pk.load(f)

        # key words
        single_list = ['cat', 'sheep', 'truck', 'bowl', 'airplane', 'giraffe', 'scissor', 'backpack', 'couch', 'cup', 'broccoli', 'person', 'kite', 'banana', 'bus', 'umbrella', 'chair', 'keyboard', 'bear', 'vase', 'handbag', 'microwave', 'snowboard', 'remote', 'cake', 'elephant', 'cow', 'motorcycle', 'sandwich', 'bottle', 'oven', 'boat', 'apple', 'car', 'laptop', 'zebra', 'bicycle', 'carrot', 'pizza', 'toilet', 'sink', 'bed', 'tie', 'book', 'horse', 'orange', 'bird', 'surfboard', 'suitcase', 'bench', 'dog', 'frisbee', 'refrigerator', 'skateboard', 'clock', 'train', 'spoon', 'fork', 'toothbrush', 'toaster', 'potted_plant', 'donut', 'dining_table', 'sports_ball', 'mouse', 'tennis_racket', 'fire_hydrant', 'baseball_glove', 'baseball_bat', 'cell_phone', 'knife', 'traffic_light', 'parking_meter', 'wine_glass', 'hair_drier', 'teddy_bear', 'ski', 'tv', 'stop_sign', 'hot_dog']
        plural_list = ['cats', 'sheep', 'trucks', 'bowls', 'airplanes', 'giraffes', 'scissors', 'backpacks', 'couches', 'cups', 'broccolis', 'people', 'kites', 'bananas', 'buses', 'umbrellas', 'chairs', 'keyboards', 'bears', 'vases', 'handbags', 'microwaves', 'snowboards', 'remotes', 'cakes', 'elephants', 'cows', 'motorcycles', 'sandwiches', 'bottles', 'ovens', 'boats', 'apples', 'cars', 'laptops', 'zebras', 'bicycles', 'carrots', 'pizzas','toilets', 'sinks', 'beds', 'ties', 'books', 'horses', 'oranges', 'birds', 'surfboards', 'suitcases', 'benches', 'dogs', 'frisbees', 'refrigerators', 'skateboards', 'clocks', 'trains', 'spoons', 'forks', 'toothbrushes', 'toasters', 'potted_plants', 'donuts', 'dining_tables', 'sports_balls', 'mice', 'tennis_rackets', 'fire_hydrants', 'baseball_gloves', 'baseball_bats', 'cell_phones', 'knives', 'traffic_lights', 'parking_meters', 'wine_glasses', 'hair_driers', 'teddy_bears', 'skis', 'tvs', 'stop_signs', 'hot_dogs']
        keyword_list = ['cat', 'sheep', 'truck', 'bowl', 'airplane', 'giraffe', 'scissors', 'backpack', 'couch', 'cup', 'broccoli', 'person', 'kite', 'banana', 'bus', 'umbrella', 'chair', 'keyboard', 'bear', 'vase', 'handbag', 'microwave', 'snowboard', 'remote', 'cake', 'elephant', 'cow', 'motorcycle', 'sandwich', 'bottle', 'oven', 'boat', 'apple', 'car', 'laptop', 'zebra', 'bicycle', 'carrot', 'pizza', 'toilet', 'sink', 'bed', 'tie', 'book', 'horse', 'orange', 'bird', 'surfboard', 'suitcase', 'bench', 'dog', 'frisbee', 'refrigerator', 'skateboard', 'clock', 'train', 'spoon', 'fork', 'toothbrush', 'toaster', 'potted_plant', 'donut', 'dining_table', 'sports_ball', 'mouse', 'tennis_racket', 'fire_hydrant', 'baseball_glove', 'baseball_bat', 'cell_phone', 'knife', 'traffic_light', 'parking_meter', 'wine_glass', 'hair_drier', 'teddy_bear', 'skis', 'tv', 'stop_sign', 'hot_dog']


        folder = os.getcwd()[:-4] + 'out//' + tsv_name + '//'
        if not os.path.exists(os.path.join('out', tsv_name)):
            os.makedirs(os.path.join('out', tsv_name))

        # iterating through the tsv files
        iterating(tsv_name, p0_f, p1_f, dic)

    # test()
        