import pickle as pk
import os
import pandas as pd
from collections import defaultdict, Counter
import traceback
import json
import pdb

class session_results:
    def __init__(self, slice=None) -> None:
        if slice == None:
            self.session_size = 8
        else:
            self.session_size = len(slice)
        self.participants_num = 3
        # self.participants_name = ['p'+str(i) for i in range(1, self.participants_num+1)]
        self.result_path = 'results'
        self.s_size = 500
        self.label = set()
        if slice == None:
            self.session_name_ls = ['session_'+str(i) for i in range(1,self.session_size+1)]
        else:
            self.session_name_ls = ['session_'+str(i) for i in slice]
        with open('./session_fpath_2_ori_fpath.pkl', 'rb') as f: 
            self.dic = pk.load(f)
        
        # result_dict['session_2'][498]['p1']=4
        self.result_dict = {ses:defaultdict(dict) for ses in self.session_name_ls}
        for session_name in self.session_name_ls:
            self.load_xlsx(session_name)
        
        return
        
    def p_join(self, session_name:str, fname:str)->str:
        return os.path.join(self.result_path, session_name, fname)

    def load_xlsx(self, session_name:str)->None:
        fname_ls = [session_name+'_score_'+str(i)+'.xlsx' for i in range(1,self.participants_num+1)]
        # fpath = self.p_join(session_name, fname)
        for fname in fname_ls:
            df = pd.read_excel(self.p_join(session_name, fname))
            participant_num = fname.split('score_')[1].split('.xlsx')[0]
            for _,row in df.iterrows():
                try:
                    cur_id = int(row['id'])
                    cur_score = int(row['score'])
                    assert cur_score > 0 and cur_score < 5, 'cur_score out of range'
                    assert cur_id == _+1, 'row_num error'
                    self.result_dict[session_name][cur_id]['p'+participant_num] = cur_score
                    self.result_dict[session_name][cur_id]['label'] = self.get_label(session_name, cur_id)
                except Exception as e:
                    # print(traceback)
                    traceback.print_exc()
                    print(e)
                    pdb.set_trace()
                
    def get_label(self, session_name:str, cur_id:int)->str :
        key = os.path.join('res',session_name, str(cur_id)+'.png')
        ori_fpath = self.dic[key]
        label = ori_fpath.split('/')[1]
        self.label.add(label)
        return label
        
    def get_label_average_score(self, label_brief:str) :
        total_score = 0
        # total_size = self.session_name_ls.__len__()*self.s_size*self.participants_num/len(self.label)
        total_size = 0
        for ses, ses_dic in self.result_dict.items():
            for cid,cid_dic in ses_dic.items():
                label = cid_dic['label']
                if label_brief in label:
                    for pid in range(1, self.participants_num+1):
                        total_size += 1
                        total_score += cid_dic['p'+str(pid)]

        for ele in self.label:
            if label_brief in ele:
                print(ele, 'score: ', total_score/total_size, total_score, '/', total_size)
                break

    def get_score_distribution(self,):
        score_list = [[] for i in range(5)]
        for session_name in s_res.session_name_ls:
            for cid, cid_dic in s_res.result_dict[session_name].items():
                for i in range(1,self.participants_num+1):
                    score = cid_dic['p'+str(i)]
                    label = cid_dic['label']
                    # if score == 1 and label == 'final_result_test_bar2':
                    if score == 4 and label == 'azure_1k' and cid >460:
                        print(session_name, cid)
                        pdb.set_trace()
                    score_list[score].append(label)

        for i in range(1,5):
            ci = Counter(score_list[i])
            print(ci)
        
    
if __name__ == '__main__':
    s_res = session_results()
    for ele in s_res.label:
        s_res.get_label_average_score(ele)
    for session_name in s_res.session_name_ls:
        out_ls = []
        for cid, cid_dic in s_res.result_dict[session_name].items():
            s1 = cid_dic['p1']
            s2 = cid_dic['p2']
            s3 = cid_dic['p3']
            # pdb.set_trace()
            assert s1>0 and s1<5
            assert s2>0 and s2<5
            assert s3>0 and s3<5
            cur_tup = (s1, s2, s3)
            out_ls.append(cur_tup)
        assert out_ls.__len__() == 500
        # pk.dump(out_ls, open(os.path.join('list_for_icc',session_name+'_icc.pkl'), 'wb'))
    print('finished')
    
