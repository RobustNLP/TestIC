# add constrain to other labels
# patse_all4 with new insection and bigger k2
import numpy as np
import sys
import os
import re
import pickle as pk
import random
import math
import argparse
import pdb 

random.seed(a=None)

global mis 
mis = 0

def init(bg, ob):
    save_base_path(bg)
    save_object_path(ob)

def mutate(x, y, idx):
    # paste ob over bg and synthesize a new image 
    return insert_image(x, y, idx)

def save_base_path(bg):
    with open("./background_path.txt", 'w') as f:
        f.write(bg)

def save_object_path(ob):
    with open("./object_path.txt", 'w') as f:
        f.write(ob)

from PIL import Image, ImageOps

def object_image_path():
    with open("./object_path.txt", 'r') as f:
        l = f.readlines()[0]
    return l

def background_image_path():
    with open("./background_path.txt", 'r') as f:
        l = f.readlines()[0]
    return l

def synthesized_imagepath(idx):
    base = "./new_"
    path = base + str(idx) + ".png"
    return path

def insert_image(x, y, idx):
    obj_img = Image.open(object_image_path(), 'r')
    # obj_img.thumbnail(size, Image.ANTIALIAS)
    bg_img = Image.open(background_image_path())

    bg_img.paste(obj_img.convert('RGBA'), (x, y), mask=obj_img.convert('RGBA'))
    new_path = synthesized_imagepath(idx)
    bg_img.save(new_path)
    return new_path

def get_diagonal(wid, hei):
    return (wid**2 + hei**2)**0.5

def get_insection(l1, l2, r1, r2, u1, u2, d1, d2):
    return max(0, min(r1, r2)-max(l1, l2))*max(0, min(d1, d2)-max(u1, u2))

# distance from object to label
def get_dis(orx, ory,r):
    dis = 10000000000000
    dis = min(dis,(orx-r[0])**2+(ory-r[1])**2)
    dis = min(dis,(orx-r[0])**2+(ory-r[3])**2)
    dis = min(dis,(orx-r[2])**2+(ory-r[1])**2)
    dis = min(dis,(orx-r[2])**2+(ory-r[3])**2)
    return dis

def softmax_area(rect, bg_area):
    maxlen = len(rect)
    ans_area = 0
    area_list = []
    soft_sum = 0
    for i in range(maxlen):
        area_list.append((rect[i][2]-rect[i][0])*(rect[i][3]-rect[i][1]))
        soft_sum+=np.exp(area_list[i]/bg_area)
    for i in range(maxlen):
        ans_area +=area_list[i]*(np.exp(area_list[i]/bg_area)/soft_sum)
    return ans_area

def check_out_of_boundary(xx, yy, obj_size, bg_size):
    if xx < 0 or yy < 0:
        return 0
    if xx+obj_size[0] >bg_size[0] or yy+obj_size[1] >bg_size[1]:
        return 0
    return 1


def get_mid_cord(lab_h, cord_h, obj_size, cx, cy, tan_obj):
    mid_h = (lab_h+cord_h)/2
    al_x = cx + (mid_h - cy)/tan_obj 
    al_x = al_x -obj_size[0]/2
    al_y = mid_h - obj_size[1]/2
    return (al_x, al_y)

def insert_single_image(bg_path, obj_path, image_id, bar1, bar2, bar3):
    bg_img = Image.open(bg_path)
    obj_img = Image.open(obj_path)
    obj2_img = Image.open(obj_path)
    rect = image_dict[image_id]
    max_size = [rect[0][2]-rect[0][0], rect[0][3]-rect[0][1]]

    biglabel = 0
    avg_area = 0
    cnt_arr = 0
    # for i in range(min(len(rect),5)):
    #     cnt_arr+=1
    #     avg_area += (rect[i][2]-rect[i][0])*(rect[i][3]-rect[i][1])
    bg_area = bg_img.size[0]*bg_img.size[1]
    avg_area = softmax_area(rect, bg_area)
    label_ratio = avg_area / bg_area
    if label_ratio > 0.7:
        biglabel = 1
        return 0


    original_size = (obj_img.size[0], obj_img.size[1])
    max_dig = get_diagonal(rect[0][2] - rect[0][0], rect[0][3] - rect[0][1])*0.27
    multiply_ratio = 1
    insection = 0
    # x0 = 0
    # y0 = 0
    # x1 = 0
    # y1 = 0
    x2 = 0
    y2 = 0
    x3 = 0
    y3 = 0
    count = 0 
    # the biggest and smallest ratio of the new object versus original object
    if label_ratio <0.4:
        k1 = avg_area * 0.8
        k2 = avg_area * 1.3
    else:
        k1 = avg_area * 0.03
        k2 = avg_area * 0.37
    global mis

    tan_obj = 0
    cx = rect[0][0]+(rect[0][2]-rect[0][0])/2
    cy = rect[0][1]+(rect[0][3]-rect[0][1])/2

    # first we get the obj_size with the 10%-20% and 20%-30% mission (part.1), then use the size to guess the starting point in 0% and 0-10% mission (part.2)
    while 1:
        count += 1

        k = k1 + random.randint(0,100)*(k2-k1)/100
        newobj_versus_obj_ratio = k / (original_size[0]*original_size[1])
        new_size = (int( multiply_ratio * original_size[0] * math.sqrt(newobj_versus_obj_ratio)),int(multiply_ratio * original_size[1]* math.sqrt(newobj_versus_obj_ratio)))
        # newobj_versus_obj_ratio = k / (obj_img.size[0]*obj_img.size[1])
        # new_size =  (int( multiply_ratio * obj_img.size[0] * math.sqrt(newobj_versus_obj_ratio)),int(multiply_ratio * obj_img.size[1]* math.sqrt(newobj_versus_obj_ratio)))
        if new_size[0]<1 or new_size[1]<1:
            count = count-1
            continue
        obj_img = obj_img.resize(new_size)
        if obj_img.size[0]>bg_img.size[0] or obj_img.size[1]>bg_img.size[1]:
            if count >=3000:
                break
            else:
                continue
        # insection0 = list()
        # insection1 = list()
        # insection2 = list()
        insection3 = list()
        x_min = 0
        x_max = bg_img.size[0] - obj_img.size[0]
        y_min = 0
        y_max = bg_img.size[1] - obj_img.size[1]



        # x2 = x_min + random.randint(0,1000) * (x_max - x_min)/1000
        # x2= int(x2)
        # y2 = y_min + random.randint(0,1000) * (y_max - y_min)/1000
        # y2= int(y2)

        x3 = x_min + random.randint(0,1000) * (x_max - x_min)/1000
        x3= int(x3)
        y3 = y_min + random.randint(0,1000) * (y_max - y_min)/1000
        y3= int(y3)

        flag = 1
        if count >= 3000:
            break

        # insection0.append ( get_insection(x0,rect[0][0],x0+obj_img.size[0], rect[0][2], y0, rect[0][1], y0+obj_img.size[1], rect[0][3])/((rect[0][2]-rect[0][0])*(rect[0][3]-rect[0][1])))
        # if insection0[0]>0:
        #     flag=0
        # for i in range(1,len(rect)):
        #     insection0.append ( get_insection(x0,rect[i][0],x0+obj_img.size[0], rect[i][2], y0, rect[i][1], y0+obj_img.size[1], rect[i][3])/((rect[i][2]-rect[i][0])*(rect[i][3]-rect[i][1])))
        #     if insection0[i]>0.5:
        #         flag = 0
        #         break

        # insection1.append ( get_insection(x1,rect[0][0],x1+obj_img.size[0], rect[0][2], y1, rect[0][1], y1+obj_img.size[1], rect[0][3])/((rect[0][2]-rect[0][0])*(rect[0][3]-rect[0][1])))
        # if insection1[0]>0.1 :
        #     flag=0
        # for i in range(1,len(rect)):
        #     insection1.append ( get_insection(x1,rect[i][0],x1+obj_img.size[0], rect[i][2], y1, rect[i][1], y1+obj_img.size[1], rect[i][3])/((rect[i][2]-rect[i][0])*(rect[i][3]-rect[i][1])))
        #     if insection1[i]>0.5:
        #         flag = 0
        #         break
        # in2 = get_insection(x2,rect[0][0],x2+obj_img.size[0], rect[0][2], y2, rect[0][1], y2+obj_img.size[1], rect[0][3])
        # insection2.append(in2/((rect[0][2]-rect[0][0])*(rect[0][3]-rect[0][1])))
        # # insection2.append ( in2 /((obj_img.size[0]*obj_img.size[1]-in2)+(rect[0][2]-rect[0][0])*(rect[0][3]-rect[0][1])))
        # if insection2[0]<bar1 or insection2[0]>bar2:
        #     flag=0
        # for i in range(1,len(rect)):
        #     ini = get_insection(x2,rect[i][0],x2+obj_img.size[0], rect[i][2], y2, rect[i][1], y2+obj_img.size[1], rect[i][3])
        #     insection2.append(ini/((rect[i][2]-rect[i][0])*(rect[i][3]-rect[i][1])))
        #     # insection2.append ( ini/((obj_img.size[0]*obj_img.size[1]-ini)+(rect[i][2]-rect[i][0])*(rect[i][3]-rect[i][1])))
        #     if insection2[i]>bar2:
        #         flag = 0
        #         break

        in3 = get_insection(x3,rect[0][0],x3+obj_img.size[0], rect[0][2], y3, rect[0][1], y3+obj_img.size[1], rect[0][3])
        insection3.append(in3/((rect[0][2]-rect[0][0])*(rect[0][3]-rect[0][1])))
        # insection3.append ( in3 /((obj_img.size[0]*obj_img.size[1]-in3)+(rect[0][2]-rect[0][0])*(rect[0][3]-rect[0][1])))
        if insection3[0]<bar2 or insection3[0]>bar3:
            flag=0
        for i in range(1,len(rect)):
            ini = get_insection(x3,rect[i][0],x3+obj_img.size[0], rect[i][2], y3, rect[i][1], y3+obj_img.size[1], rect[i][3])
            insection3.append(ini/((rect[i][2]-rect[i][0])*(rect[i][3]-rect[i][1])))
            # insection3.append ( ini/((obj_img.size[0]*obj_img.size[1]-ini)+(rect[i][2]-rect[i][0])*(rect[i][3]-rect[i][1])))
            if insection3[i]>bar3:
                flag = 0
                break
        if x3 == cx :
            continue
        if y3 == cy :
            continue
        if flag == 1:
            tan_obj = (y3-cy)/(x3-cx)
            break
        # count += 1
        # insection = get_insection(x,rect[0][0],x+obj_img.size[0], rect[0][2], y, rect[0][1], y+obj_img.size[1], rect[0][3])/((rect[0][2]-rect[0][0])*(rect[0][3]-rect[0][1]))
        # count += 1
        # if insection <0.2:
        #     break
    if count >= 3000:
        mis += 1
        print('skip ',mis)
        # return insection
        return 0
    # newobj_versus_obj_ratio = (obj_img.size[0]*obj_img.size[1]) / (obj2_img.size[0]*obj2_img.size[1])
    # new_size =  (int( multiply_ratio * obj2_img.size[0] * math.sqrt(newobj_versus_obj_ratio)),int(multiply_ratio * obj2_img.size[1]* math.sqrt(newobj_versus_obj_ratio)))
    new_size = obj_img.size
    obj2_img = obj2_img.resize(new_size)

# part.2
    # we run part.2 for 10000 times to get the closest new object to the main label
    min_dis = 10000000000000
    # x00 y00 x11 y11 are the final starting point in 0% and 0%-10% insection missions
    x00 = 0
    y00 = 0
    x11 = 0
    y11 = 0
    alter_ins0 = list()
    alter_ins1 = list()
    # count2 = 0
    if x3-cx < 0:
        alter_h1 = (obj2_img.size[0]/2-cx)*tan_obj
    else:
        alter_h1 = (bg_img.size[0]-obj2_img.size[0]/2-cx)*tan_obj
    if y3-cy < 0:
        alter_h2 = obj2_img.size[1]/2-cy
    else:
        alter_h2 = bg_img.size[1]-obj2_img.size[1]/2-cy
    
    if abs(alter_h1) < abs(alter_h2):
        alter_h = alter_h1
    else:
        alter_h = alter_h2

    count2 = 0
    lab_h = cy
    bord_h = alter_h
    alter_x = 0
    alter_y = 0
    insection2 = list()
    while count2 < 100:
        count2 += 1
        alter_x , alter_y = get_mid_cord(lab_h, bord_h, obj2_img.size, cx, cy, tan_obj)
        alter_x = int(alter_x)
        alter_y = int(alter_y)
        in0 = get_insection(alter_x, rect[0][0], alter_x+obj2_img.size[0], rect[0][2], alter_y, rect[0][1], alter_y+obj2_img.size[1], rect[0][3])
        # pdb.set_trace()
        print(bar1,'--',bar2,in0/((rect[0][2]-rect[0][0])*(rect[0][3]-rect[0][1])))
        if in0/((rect[0][2]-rect[0][0])*(rect[0][3]-rect[0][1])) > bar2:
            lab_h= (lab_h+bord_h)/2
        elif in0/((rect[0][2]-rect[0][0])*(rect[0][3]-rect[0][1])) < bar1:
            bord_h =(lab_h+bord_h)/2
        else:
            # flag = do_check(alter_x, alter_y, rect)
            # if flag == 0:
            #     mis+=1
            #     print('skip', mis)
            #     return
            # else:
            insection2.append(in0/((rect[0][2]-rect[0][0])*(rect[0][3]-rect[0][1])))
            for i in range(1,len(rect)):
                ini = get_insection(alter_x, rect[i][0], alter_x+obj2_img.size[0], rect[i][2], alter_y, rect[i][1], alter_y+obj2_img.size[1], rect[i][3])
                insection2.append(ini/((rect[i][2]-rect[i][0])*(rect[i][3]-rect[i][1])))
                if insection2[i] > bar2 :
                    mis+=1
                    print('skip bar2', mis)
                    # pdb.set_trace()
                    return 0
                
            x2 = alter_x
            y2 = alter_y
            break
    if count2 >= 100:
        mis+=1
        print('skip bar2', mis,'count2',count2)
        # pdb.set_trace()
        return 0

    count2 = 0
    lab_h = cy
    bord_h = alter_h
    insection1 = list()
    while count2 < 100:
        count2 += 1
        alter_x , alter_y = get_mid_cord(lab_h, bord_h, obj2_img.size, cx, cy, tan_obj)
        alter_x = int(alter_x)
        alter_y = int(alter_y)
        in0 = get_insection(alter_x, rect[0][0], alter_x+obj2_img.size[0], rect[0][2], alter_y, rect[0][1], alter_y+obj2_img.size[1], rect[0][3])
        if in0/((rect[0][2]-rect[0][0])*(rect[0][3]-rect[0][1])) > bar1:
            lab_h= (lab_h+bord_h)/2
        elif in0/((rect[0][2]-rect[0][0])*(rect[0][3]-rect[0][1])) == 0.0 :
            bord_h =(lab_h+bord_h)/2
        else:
            # flag = do_check(alter_x, alter_y, rect)
            # if flag == 0:
            #     mis+=1
            #     print('skip', mis)
            #     return
            # else:
            insection1.append(in0/((rect[0][2]-rect[0][0])*(rect[0][3]-rect[0][1])))
            for i in range(1,len(rect)):
                ini = get_insection(alter_x, rect[i][0], alter_x+obj2_img.size[0], rect[i][2], alter_y, rect[i][1], alter_y+obj2_img.size[1], rect[i][3])
                insection1.append(ini/((rect[i][2]-rect[i][0])*(rect[i][3]-rect[i][1])))
                if insection1[i] > bar1 :
                    mis+=1
                    print('skip bar1', mis)
                    return 0
                
            x1 = alter_x
            y1 = alter_y
            break
    if count2 >= 100:
        mis+=1
        print('skip bar1', mis)    
        return 0

    count2 = 0
    lab_h = cy
    bord_h = alter_h
    insection0 = list()
    while count2 < 10:
        count2 += 1
        alter_x , alter_y = get_mid_cord(lab_h, bord_h, obj2_img.size, cx, cy, tan_obj)
        alter_x = int(alter_x)
        alter_y = int(alter_y)
        in0 = get_insection(alter_x, rect[0][0], alter_x+obj2_img.size[0], rect[0][2], alter_y, rect[0][1], alter_y+obj2_img.size[1], rect[0][3])
        if in0/((rect[0][2]-rect[0][0])*(rect[0][3]-rect[0][1])) > 0.0:
            lab_h= (lab_h+bord_h)/2
        # elif in0 == 0.0 :
        #     bord_h =(lab_h+bord_h)/2
        else:
            bord_h = (lab_h+bord_h)/2
        # else:
            # flag = do_check(alter_x, alter_y, rect)
            # if flag == 0:
            #     mis+=1
            #     print('skip', mis)
            #     return
            # else:
    for i in range(len(rect)):
        ini = get_insection(alter_x, rect[i][0], alter_x+obj2_img.size[0], rect[i][2], alter_y, rect[i][1], alter_y+obj2_img.size[1], rect[i][3])
        insection0.append(ini/((rect[i][2]-rect[i][0])*(rect[i][3]-rect[i][1])))
        if insection0[i] > 0:
            mis+=1
            print('skip 0', mis)
            return 0
        
        x00 , y00 = get_mid_cord(lab_h, bord_h, obj2_img.size, cx, cy, tan_obj)
        x00 = int(x00)
        y00 = int(y00)
        # break

    if check_out_of_boundary(x00, y00, obj2_img.size, bg_img.size)==0:
        mis += 1
        print('skip. out of boundary', mis)
        return 0
    
    if check_out_of_boundary(x1, y1, obj2_img.size, bg_img.size) ==0 :
        mis += 1
        print('skip. out of boundary', mis)
        return 0
    if check_out_of_boundary(x2, y2, obj2_img.size, bg_img.size) ==0 :
        mis += 1
        print('skip. out of boundary', mis)
        return 0
    if check_out_of_boundary(x3, y3, obj2_img.size, bg_img.size) ==0 :
        mis += 1
        print('skip. out of boundary', mis)
        return 0




    print(bg_path.split('/')[-1][:-4] + '_' + obj_path.split('/')[-1],count,count2)
    print('file_name: ', bg_path.split('/')[-1][:-4] + '_' + obj_path.split('/')[-1], 'insection: ',insection0, 'cycle1: ',count, 'cycle2: ', count2, file = mylog0)
    print('file_name: ', bg_path.split('/')[-1][:-4] + '_' + obj_path.split('/')[-1], 'insection: ',insection1, 'cycle1: ',count, 'cycle2: ', count2, file = mylog1)
    print('file_name: ', bg_path.split('/')[-1][:-4] + '_' + obj_path.split('/')[-1], 'insection: ',insection2, 'cycle1: ',count, 'cycle2: ', count2, file = mylog2)
    print('file_name: ', bg_path.split('/')[-1][:-4] + '_' + obj_path.split('/')[-1], 'insection: ',insection3, 'cycle1: ',count, 'cycle2: ', count2, file = mylog3)
    current_bg_img0 = Image.fromarray(np.array(bg_img)) 
    current_bg_img1 = Image.fromarray(np.array(bg_img)) 
    current_bg_img2 = Image.fromarray(np.array(bg_img)) 
    current_bg_img3 = Image.fromarray(np.array(bg_img)) 
    # current_bg_img = bg_img
    #constrain the value of (x, y)
    # x0 = np.clip(x0, 0, bg_img.size[0]-obj_img.size[0])
    # y0 = np.clip(y0, 0, bg_img.size[1]-obj_img.size[1])

    x1 = np.clip(x1, 0, bg_img.size[0]-obj_img.size[0])
    y1 = np.clip(y1, 0, bg_img.size[1]-obj_img.size[1])

    x00 = np.clip(x00, 0, bg_img.size[0]-obj_img.size[0])
    y00 = np.clip(y00, 0, bg_img.size[1]-obj_img.size[1])

    # x11 = np.clip(x11, 0, bg_img.size[0]-obj_img.size[0])
    # y11 = np.clip(y11, 0, bg_img.size[1]-obj_img.size[1])

    x2 = np.clip(x2, 0, bg_img.size[0]-obj_img.size[0])
    y2 = np.clip(y2, 0, bg_img.size[1]-obj_img.size[1])

    x3 = np.clip(x3, 0, bg_img.size[0]-obj_img.size[0])
    y3 = np.clip(y3, 0, bg_img.size[1]-obj_img.size[1])

    current_bg_img0.paste(obj2_img.convert('RGBA'), (x00, y00), mask=obj2_img.convert('RGBA'))
        # remember to use split('\\') in windows but split('/') in ubuntu
    target_fname = os.path.join('./inserted_result_80/inserted_result_same0', bg_path.split('/')[-1][:-4] + '_' + obj_path.split('/')[-1])
        # target_fname = os.path.join('./inserted_result', bg_path.split('\\')[-1][:-4] + '_' + obj_path.split('\\')[-1][:-4] + '_random_' + str(i) + '.png')
    current_bg_img0.save( target_fname )


    current_bg_img1.paste(obj2_img.convert('RGBA'), (x1, y1), mask=obj2_img.convert('RGBA'))
        # remember to use split('\\') in windows but split('/') in ubuntu
    target_fname = os.path.join('./inserted_result_80/inserted_result_same_bar1', bg_path.split('/')[-1][:-4] + '_' + obj_path.split('/')[-1])
        # target_fname = os.path.join('./inserted_result', bg_path.split('\\')[-1][:-4] + '_' + obj_path.split('\\')[-1][:-4] + '_random_' + str(i) + '.png')
    current_bg_img1.save( target_fname )


    current_bg_img2.paste(obj2_img.convert('RGBA'), (x2, y2), mask=obj2_img.convert('RGBA'))
        # remember to use split('\\') in windows but split('/') in ubuntu
    target_fname = os.path.join('./inserted_result_80/inserted_result_same_bar2', bg_path.split('/')[-1][:-4] + '_' + obj_path.split('/')[-1])
        # target_fname = os.path.join('./inserted_result', bg_path.split('\\')[-1][:-4] + '_' + obj_path.split('\\')[-1][:-4] + '_random_' + str(i) + '.png')
    current_bg_img2.save( target_fname )


    current_bg_img3.paste(obj2_img.convert('RGBA'), (x3, y3), mask=obj2_img.convert('RGBA'))
        # remember to use split('\\') in windows but split('/') in ubuntu
    target_fname = os.path.join('./inserted_result_80/inserted_result_same_bar3', bg_path.split('/')[-1][:-4] + '_' + obj_path.split('/')[-1])
        # target_fname = os.path.join('./inserted_result', bg_path.split('\\')[-1][:-4] + '_' + obj_path.split('\\')[-1][:-4] + '_random_' + str(i) + '.png')
    current_bg_img3.save( target_fname )

    return 1
    # return insection1, insection2, insection3
# obj = Image.open('./obj.png')
# bg = Image.open('./bg.jpg')
# insert_size = (bg.size[0]//3, bg.size[1]//3)
# bg.paste(obj.resize(insert_size).convert('RGBA'), (bg.size[0]-100, bg.size[1]-100), mask=obj.resize(insert_size).convert('RGBA'))
# bg.show()

os.system('rm ./inserted_result_80/inserted_result_same0/*')
os.system('rm ./inserted_result_80/inserted_result_same_bar1/*')
os.system('rm ./inserted_result_80/inserted_result_same_bar2/*')
os.system('rm ./inserted_result_80/inserted_result_same_bar3/*')
os.system('rm ./inserted_result_80/bg_images/*')

with open('./image_dict2.pkl', 'rb') as f:
    image_dict = pk.load(f)

mylog0 = open('./inserted_result_80/inserted_result_same0/an_insert_log', mode = 'a' , encoding = 'utf-8')
mylog0.seek(0)
mylog0.truncate()

mylog1 = open('./inserted_result_80/inserted_result_same_bar1/an_insert_log', mode = 'a' , encoding = 'utf-8')
mylog1.seek(0)
mylog1.truncate()

mylog2 = open('./inserted_result_80/inserted_result_same_bar2/an_insert_log', mode = 'a' , encoding = 'utf-8')
mylog2.seek(0)
mylog2.truncate()

mylog3 = open('./inserted_result_80/inserted_result_same_bar3/an_insert_log', mode = 'a' , encoding = 'utf-8')
mylog3.seek(0)
mylog3.truncate()

# mis = 0
l = list()
max_ins1 = 0
max_ins2 = 0
max_ins3 = 0
min_ins= 10

parser = argparse.ArgumentParser()
parser.add_argument('--bar1', type=float, required=1)
parser.add_argument('--bar2', type=float, required=1)
parser.add_argument('--bar3', type=float, required=1)
args = parser.parse_args()

bg_cnt = 0
for fname in os.listdir('./chosen_bg_images_all'):
    # if f
    # if bg_cnt >= 1000:
    #     break
    # if random.randint(1, 2)%2:
    #     continue
    if fname[-4:] != '.jpg':
        continue
    image_id =  int(re.findall( r"COCO_val2014_(.+?).jpg", fname)[0])
    if image_id not in image_dict:
        continue #since some iamges don't have rect[0]angle labels
    largest_ob_rec= image_dict[image_id]
    count_obj = 0
    image_pool_dir='./YOLACT_seg_results/image_pool_80'
    image_pool_list = os.listdir(image_pool_dir)
    random.shuffle(image_pool_list)
    for obj_dir in image_pool_list:
        for obj_fname in os.listdir( os.path.join(
            image_pool_dir, obj_dir) ):
            # pick an object randomly

            # obj_lst=['bird', 'apple', 'airplane', 'car', 'cat', 'dog', 'elephant', 'giraffe', 'horse', 'sheep', 'zebra', 'cow']
            # if obj_dir not in obj_lst:
            #     continue
            # # modified at 2022/1/15

            # if random.randint(1,20) != 17:
            #     continue
            # im = Image.open(os.path.join(image_pool_dir, obj_dir, obj_fname))
            bg_path = os.path.join('./chosen_bg_images_all', fname)
            obj_path = os.path.join(image_pool_dir, obj_dir, obj_fname)
            if insert_single_image(bg_path, obj_path, image_id, args.bar1, args.bar2, args.bar3):
                count_obj+=1
            if count_obj >=5:
                break
        if count_obj >=5:
            break
    if count_obj>0:
        # os.system('rm ./final_bg_test/*')
        os.system('cp ./chosen_bg_images_all/'+fname+' ./inserted_result_80/bg_images')
        bg_cnt +=1
        continue
            # ans1, ans2, ans3 = insert_single_image( bg_path, obj_path, image_id )
            # if len(ans1)>0:
            #     max_ins1 = max(max_ins1, ans1[0])
            # if len(ans2)>0:
            #     max_ins2 = max(max_ins2, ans2[0])
            # if len(ans3)>0:
            #     max_ins3 = max(max_ins3, ans3[0])
            # min_ins = min(min_ins, insert_single_image( bg_path, obj_path, image_id )[0])
# print('max_insection: ',max_ins1,'min_insection: ', min_ins ,'missing_pic: ', mis, file= mylog1)
# print('max_insection: ',max_ins2,'min_insection: ', min_ins ,'missing_pic: ', mis, file= mylog2)
# print('max_insection: ',max_ins3,'min_insection: ', min_ins ,'missing_pic: ', mis, file= mylog3)

print('missing_pic: ', mis, file=mylog0)
print('missing_pic: ', mis, file=mylog1)
print('missing_pic: ', mis, file=mylog2)
print('missing_pic: ', mis, file=mylog3)