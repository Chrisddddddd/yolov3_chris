import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image)
from utils.utils_bbox import DecodeBox

'''
璁粌鑷繁鐨勬暟鎹泦蹇呯湅娉ㄩ噴锛�
'''
class YOLO(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   浣跨敤鑷繁璁粌濂界殑妯″瀷杩涜棰勬祴涓�瀹氳淇敼model_path鍜宑lasses_path锛�
        #   model_path鎸囧悜logs鏂囦欢澶逛笅鐨勬潈鍊兼枃浠讹紝classes_path鎸囧悜model_data涓嬬殑txt
        #   濡傛灉鍑虹幇shape涓嶅尮閰嶏紝鍚屾椂瑕佹敞鎰忚缁冩椂鐨刴odel_path鍜宑lasses_path鍙傛暟鐨勪慨鏀�
        #--------------------------------------------------------------------------#
        "model_path"        : 'logs/ep001-loss46.911-val_loss8.004.pth',
        "classes_path"      : 'model_data/voc_classes.txt',
        #---------------------------------------------------------------------#
        #   anchors_path浠ｈ〃鍏堥獙妗嗗搴旂殑txt鏂囦欢锛屼竴鑸笉淇敼銆�
        #   anchors_mask鐢ㄤ簬甯姪浠ｇ爜鎵惧埌瀵瑰簲鐨勫厛楠屾锛屼竴鑸笉淇敼銆�
        #---------------------------------------------------------------------#
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        #---------------------------------------------------------------------#
        #   杈撳叆鍥剧墖鐨勫ぇ灏忥紝蹇呴』涓�32鐨勫�嶆暟銆�
        #---------------------------------------------------------------------#
        "input_shape"       : [416, 416],
        #---------------------------------------------------------------------#
        #   鍙湁寰楀垎澶т簬缃俊搴︾殑棰勬祴妗嗕細琚繚鐣欎笅鏉�
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   闈炴瀬澶ф姂鍒舵墍鐢ㄥ埌鐨刵ms_iou澶у皬
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   璇ュ彉閲忕敤浜庢帶鍒舵槸鍚︿娇鐢╨etterbox_image瀵硅緭鍏ュ浘鍍忚繘琛屼笉澶辩湡鐨剅esize锛�
        #   鍦ㄥ娆℃祴璇曞悗锛屽彂鐜板叧闂璴etterbox_image鐩存帴resize鐨勬晥鏋滄洿濂�
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
        #-------------------------------#
        #   鏄惁浣跨敤Cuda
        #   娌℃湁GPU鍙互璁剧疆鎴怓alse
        #-------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   鍒濆鍖朰OLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        #---------------------------------------------------#
        #   鑾峰緱绉嶇被鍜屽厛楠屾鐨勬暟閲�
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        #---------------------------------------------------#
        #   鐢绘璁剧疆涓嶅悓鐨勯鑹�
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    #---------------------------------------------------#
    #   鐢熸垚妯″瀷
    #---------------------------------------------------#
    def generate(self):
        #---------------------------------------------------#
        #   寤虹珛yolov3妯″瀷锛岃浇鍏olov3妯″瀷鐨勬潈閲�
        #---------------------------------------------------#
        self.net    = YoloBody(self.anchors_mask, self.num_classes)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    #---------------------------------------------------#
    #   妫�娴嬪浘鐗�
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   鍦ㄨ繖閲屽皢鍥惧儚杞崲鎴怰GB鍥惧儚锛岄槻姝㈢伆搴﹀浘鍦ㄩ娴嬫椂鎶ラ敊銆�
        #   浠ｇ爜浠呬粎鏀寔RGB鍥惧儚鐨勯娴嬶紝鎵�鏈夊叾瀹冪被鍨嬬殑鍥惧儚閮戒細杞寲鎴怰GB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   缁欏浘鍍忓鍔犵伆鏉★紝瀹炵幇涓嶅け鐪熺殑resize
        #   涔熷彲浠ョ洿鎺esize杩涜璇嗗埆
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   娣诲姞涓奲atch_size缁村害
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   灏嗗浘鍍忚緭鍏ョ綉缁滃綋涓繘琛岄娴嬶紒
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   灏嗛娴嬫杩涜鍫嗗彔锛岀劧鍚庤繘琛岄潪鏋佸ぇ鎶戝埗
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return image

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]
        #---------------------------------------------------------#
        #   璁剧疆瀛椾綋涓庤竟妗嗗帤搴�
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        
        #---------------------------------------------------------#
        #   鍥惧儚缁樺埗
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   鍦ㄨ繖閲屽皢鍥惧儚杞崲鎴怰GB鍥惧儚锛岄槻姝㈢伆搴﹀浘鍦ㄩ娴嬫椂鎶ラ敊銆�
        #   浠ｇ爜浠呬粎鏀寔RGB鍥惧儚鐨勯娴嬶紝鎵�鏈夊叾瀹冪被鍨嬬殑鍥惧儚閮戒細杞寲鎴怰GB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   缁欏浘鍍忓鍔犵伆鏉★紝瀹炵幇涓嶅け鐪熺殑resize
        #   涔熷彲浠ョ洿鎺esize杩涜璇嗗埆
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   娣诲姞涓奲atch_size缁村害
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   灏嗗浘鍍忚緭鍏ョ綉缁滃綋涓繘琛岄娴嬶紒
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   灏嗛娴嬫杩涜鍫嗗彔锛岀劧鍚庤繘琛岄潪鏋佸ぇ鎶戝埗
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                                                    
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------------#
                #   灏嗗浘鍍忚緭鍏ョ綉缁滃綋涓繘琛岄娴嬶紒
                #---------------------------------------------------------#
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                #---------------------------------------------------------#
                #   灏嗛娴嬫杩涜鍫嗗彔锛岀劧鍚庤繘琛岄潪鏋佸ぇ鎶戝埗
                #---------------------------------------------------------#
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                            
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   鍦ㄨ繖閲屽皢鍥惧儚杞崲鎴怰GB鍥惧儚锛岄槻姝㈢伆搴﹀浘鍦ㄩ娴嬫椂鎶ラ敊銆�
        #   浠ｇ爜浠呬粎鏀寔RGB鍥惧儚鐨勯娴嬶紝鎵�鏈夊叾瀹冪被鍨嬬殑鍥惧儚閮戒細杞寲鎴怰GB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   缁欏浘鍍忓鍔犵伆鏉★紝瀹炵幇涓嶅け鐪熺殑resize
        #   涔熷彲浠ョ洿鎺esize杩涜璇嗗埆
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   娣诲姞涓奲atch_size缁村害
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   灏嗗浘鍍忚緭鍏ョ綉缁滃綋涓繘琛岄娴嬶紒
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   灏嗛娴嬫杩涜鍫嗗彔锛岀劧鍚庤繘琛岄潪鏋佸ぇ鎶戝埗
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
