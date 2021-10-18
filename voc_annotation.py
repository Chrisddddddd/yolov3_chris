import os
import random
import xml.etree.ElementTree as ET

from utils.utils import get_classes

#--------------------------------------------------------------------------------------------------------------------------------#
#   annotation_mode鐢ㄤ簬鎸囧畾璇ユ枃浠惰繍琛屾椂璁＄畻鐨勫唴瀹�
#   annotation_mode涓�0浠ｈ〃鏁翠釜鏍囩澶勭悊杩囩▼锛屽寘鎷幏寰梀OCdevkit/VOC2007/ImageSets閲岄潰鐨則xt浠ュ強璁粌鐢ㄧ殑2007_train.txt銆�2007_val.txt
#   annotation_mode涓�1浠ｈ〃鑾峰緱VOCdevkit/VOC2007/ImageSets閲岄潰鐨則xt
#   annotation_mode涓�2浠ｈ〃鑾峰緱璁粌鐢ㄧ殑2007_train.txt銆�2007_val.txt
#--------------------------------------------------------------------------------------------------------------------------------#
annotation_mode     = 0
#-------------------------------------------------------------------#
#   蹇呴』瑕佷慨鏀癸紝鐢ㄤ簬鐢熸垚2007_train.txt銆�2007_val.txt鐨勭洰鏍囦俊鎭�
#   涓庤缁冨拰棰勬祴鎵�鐢ㄧ殑classes_path涓�鑷村嵆鍙�
#   濡傛灉鐢熸垚鐨�2007_train.txt閲岄潰娌℃湁鐩爣淇℃伅
#   閭ｄ箞灏辨槸鍥犱负classes娌℃湁璁惧畾姝ｇ‘
#   浠呭湪annotation_mode涓�0鍜�2鐨勬椂鍊欐湁鏁�
#-------------------------------------------------------------------#
classes_path        = 'model_data/voc_classes.txt'
#--------------------------------------------------------------------------------------------------------------------------------#
#   trainval_percent鐢ㄤ簬鎸囧畾(璁粌闆�+楠岃瘉闆�)涓庢祴璇曢泦鐨勬瘮渚嬶紝榛樿鎯呭喌涓� (璁粌闆�+楠岃瘉闆�):娴嬭瘯闆� = 9:1
#   train_percent鐢ㄤ簬鎸囧畾(璁粌闆�+楠岃瘉闆�)涓缁冮泦涓庨獙璇侀泦鐨勬瘮渚嬶紝榛樿鎯呭喌涓� 璁粌闆�:楠岃瘉闆� = 9:1  
#   浠呭湪annotation_mode涓�0鍜�1鐨勬椂鍊欐湁鏁�
#--------------------------------------------------------------------------------------------------------------------------------#
trainval_percent    = 0.9
train_percent       = 0.9
#-------------------------------------------------------#
#   鎸囧悜VOC鏁版嵁闆嗘墍鍦ㄧ殑鏂囦欢澶�
#   榛樿鎸囧悜鏍圭洰褰曚笅鐨刅OC鏁版嵁闆�
#-------------------------------------------------------#
VOCdevkit_path  = 'E:/DATASET/VOCdevkit'

VOCdevkit_sets  = [('2007', 'train'), ('2007', 'val')]
classes, _      = get_classes(classes_path)

def convert_annotation(year, image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml'%(year, image_id)), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        
if __name__ == "__main__":
    random.seed(0)
    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")
        xmlfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
        saveBasePath    = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')
        temp_xml        = os.listdir(xmlfilepath)
        total_xml       = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num     = len(total_xml)  
        list    = range(num)  
        tv      = int(num*trainval_percent)  
        tr      = int(tv*train_percent)  
        trainval= random.sample(list,tv)  
        train   = random.sample(trainval,tr)  
        
        print("train and val size",tv)
        print("train size",tr)
        ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
        ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
        ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
        fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
        
        for i in list:  
            name=total_xml[i][:-4]+'\n'  
            if i in trainval:  
                ftrainval.write(name)  
                if i in train:  
                    ftrain.write(name)  
                else:  
                    fval.write(name)  
            else:  
                ftest.write(name)  
        
        ftrainval.close()  
        ftrain.close()  
        fval.close()  
        ftest.close()
        print("Generate txt in ImageSets done.")

    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate 2007_train.txt and 2007_val.txt for train.")
        for year, image_set in VOCdevkit_sets:
            image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt'%(year, image_set)), encoding='utf-8').read().strip().split()
            list_file = open('%s_%s.txt'%(year, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('%s/VOC%s/images/%s.jpg'%(os.path.abspath(VOCdevkit_path), year, image_id))

                convert_annotation(year, image_id, list_file)
                list_file.write('\n')
            list_file.close()
        print("Generate 2007_train.txt and 2007_val.txt for train done.")
