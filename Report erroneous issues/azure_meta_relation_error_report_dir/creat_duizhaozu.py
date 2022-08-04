import pickle as pk
from tsv_file import TSVFile
import json

ori_tsv = TSVFile('ori_azure_cap/pred.val_test_5000.tsv')
ori_cap_dict = dict()
for idx in range( ori_tsv.num_rows() ):
    ori_cap_dict[ ori_tsv.seek(idx)[0] ] =  json.loads( ori_tsv.seek(idx)[1] )[0]['caption']

with open('azure_1000_0/name_img_id_dict.pkl', 'rb') as f:
	d = pk.load(f)

wfile = open('./duizhaozu.txt', 'w')

for i in range(1,1001):
    idx = d[i]['image_id']
    sentence = ori_cap_dict[str(idx)]
    wfile.write(sentence + '\n')

wfile.close()
