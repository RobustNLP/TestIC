from collections import defaultdict
import json
import pickle
import os
import pdb

class Anno:
    def __init__(self) -> None:
        
        self.all_included_tokens = []

        with open ('name_img_id_dict.pkl', 'rb') as f:
            self.name_img_id_dict = pickle.load(f)
        with open ('./captions_train2014.json', 'r') as f:
            self.captions_train2014_ann = json.load(f)['annotations']
        # fout = open(fout_name, 'w')
        self.gt_cap= defaultdict(list)
        self.init_load_gt_cap()

    def query_captions(self, image_id: int) -> list:
        # image_id = 322141
        captions_list = []
        for entry in self.captions_train2014_ann:
            # pdb.set_trace()
            if entry['image_id'] == image_id:
                captions_list.append(entry['caption'])
        assert(captions_list != [])
        return captions_list
    def init_load_gt_cap(self):
        for entry in self.captions_train2014_ann:
            image_id = entry['image_id'] 
            self.gt_cap[image_id].append(entry['caption'])

if __name__ == '__main__':
    A = Anno()

