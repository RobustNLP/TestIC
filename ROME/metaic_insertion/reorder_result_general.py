import os
import re
import pickle
import cv2
import pdb 
last_image_id = 0
cnt = 0
name_img_id_dict = {}
img_id_name_dict = {}
bg_name_img_id_dict = {}
img_id = set()

os.system('rm ./ordered_results/bg_images/*')
os.system('rm ./ordered_results/final_result_test_0/*')
os.system('rm ./ordered_results/final_result_test_bar1/*')
os.system('rm ./ordered_results/final_result_test_bar2/*')
os.system('rm ./ordered_results/final_result_test_bar3/*')
bg_cnt = 0
for fname in os.listdir('./inserted_result_80/inserted_result_same0'):
    if fname.split('_')[2] =='log':
        continue
    image_id = int(fname.split('_')[2])
    # image_id = int(re.findall( r"COCO_val2014_(.+?).png", fname)[0])
    if image_id == last_image_id:
        continue
    cnt+=1
    # if image_id not in img_id:
    #     img_id.add(image_id)
    #     bg_cnt +=1
    #     bg_name_img_id_dict[bg_cnt]=image_id
    cur_dic = {}
    cur_dic['image_id'] = image_id
    cnt_ = 0
    for char in fname:
        if char == "_":
            cnt_ += 1
    if cnt_ == 6:
        cur_dic['object'] = fname.split('_')[3]
    elif cnt_ == 8:
        cur_dic["object"] = fname.split("_")[3]+"_"+fname.split("_")[4]
    cur_dic['file_name'] = fname
    name_img_id_dict[cnt] = cur_dic
    img_id_name_dict[image_id] = cnt
    # if cnt == 579 :
    #     pdb.set_trace()
    os.system('cp -i ./inserted_result_80/inserted_result_same0/'+fname+' ./ordered_results/final_result_test_0/'+str(cnt)+'.png')
    os.system('cp -i ./inserted_result_80/inserted_result_same_bar1/'+fname+' ./ordered_results/final_result_test_bar1/'+str(cnt)+'.png')
    os.system('cp -i ./inserted_result_80/inserted_result_same_bar2/'+fname+' ./ordered_results/final_result_test_bar2/'+str(cnt)+'.png')
    os.system('cp -i ./inserted_result_80/inserted_result_same_bar3/'+fname+' ./ordered_results/final_result_test_bar3/'+str(cnt)+'.png')
    # os.system('cp ./val2014/COCO_val2014_'+str(iid).zfill(12)+'.jpg ./insert_objects/chosen_bg_images_all')
    img = cv2.imread(r'./inserted_result_80/bg_images/COCO_val2014_'+str(image_id).zfill(12)+'.jpg')
    cv2.imwrite(r'./ordered_results/bg_images/'+str(cnt)+'.png',img)
    print('successfully reorder',cnt,'<---->',image_id)
pickle.dump(name_img_id_dict, open('./ordered_results/name_img_id_dict.pkl','wb'))
# pickle.dump(img_id_name_dict, open('./img_id_name_dict.pkl','wb'))
# pickle.dump(bg_name_img_id_dict, open('./bg_name_imgS_id_dict.pkl', 'wb'))