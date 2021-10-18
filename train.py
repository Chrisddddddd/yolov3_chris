#coding=gbk
#-------------------------------------#
#       �����ݼ�����ѵ��
#-------------------------------------#
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import YOLOLoss, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    #-------------------------------#
    #   �Ƿ�ʹ��Cuda
    #   û��GPU�������ó�False
    #-------------------------------#
    Cuda            = True
    #--------------------------------------------------------#
    #   ѵ��ǰһ��Ҫ�޸�classes_path��ʹ���Ӧ�Լ������ݼ�
    #--------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    #---------------------------------------------------------------------#
    #   anchors_path����������Ӧ��txt�ļ���һ�㲻�޸ġ�
    #   anchors_mask���ڰ��������ҵ���Ӧ�������һ�㲻�޸ġ�
    #---------------------------------------------------------------------#
    anchors_path    = 'model_data/yolo_anchors.txt'
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Ȩֵ�ļ��뿴README���ٶ��������ء����ݵ�Ԥѵ��Ȩ�ضԲ�ͬ���ݼ���ͨ�õģ���Ϊ������ͨ�õġ�
    #   Ԥѵ��Ȩ�ض���99%�����������Ҫ�ã����õĻ�Ȩֵ̫�������������ȡЧ�������ԣ�����ѵ���Ľ��Ҳ����á�
    #
    #   �����Ҫ�ϵ������ͽ�model_path���ó�logs�ļ������Ѿ�ѵ����Ȩֵ�ļ��� 
    #   ��model_path = ''��ʱ�򲻼�������ģ�͵�Ȩֵ��
    #
    #   �˴�ʹ�õ�������ģ�͵�Ȩ�أ��������train.py���м��صġ�
    #   �����Ҫ��ģ�ʹ�0��ʼѵ����������model_path = ''�������Freeze_Train = Fasle����ʱ��0��ʼѵ������û�ж������ɵĹ��̡�
    #   һ����������0��ʼѵ��Ч����ܲ��ΪȨֵ̫�������������ȡЧ�������ԡ�
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = 'model_data/yolo_weights.pth'
    #------------------------------------------------------#
    #   �����shape��С��һ��Ҫ��32�ı���
    #------------------------------------------------------#
    input_shape     = [416, 416]
    
    #----------------------------------------------------#
    #   ѵ����Ϊ�����׶Σ��ֱ��Ƕ���׶κͽⶳ�׶Ρ�
    #   �Դ治�������ݼ���С�޹أ���ʾ�Դ治�����Сbatch_size��
    #   �ܵ�BatchNorm��Ӱ�죬batch_size��СΪ2������Ϊ1��
    #----------------------------------------------------#
    #----------------------------------------------------#
    #   ����׶�ѵ������
    #   ��ʱģ�͵����ɱ������ˣ�������ȡ���粻�����ı�
    #   ռ�õ��Դ��С�������������΢��
    #----------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 8
    Freeze_lr           = 1e-3
    #----------------------------------------------------#
    #   �ⶳ�׶�ѵ������
    #   ��ʱģ�͵����ɲ��������ˣ�������ȡ����ᷢ���ı�
    #   ռ�õ��Դ�ϴ��������еĲ������ᷢ���ı�
    #----------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 4
    Unfreeze_lr         = 1e-4
    #------------------------------------------------------#
    #   �Ƿ���ж���ѵ����Ĭ���ȶ�������ѵ����ⶳѵ����
    #------------------------------------------------------#
    Freeze_Train        = True
    #------------------------------------------------------#
    #   ���������Ƿ�ʹ�ö��̶߳�ȡ����
    #   �������ӿ����ݶ�ȡ�ٶȣ����ǻ�ռ�ø����ڴ�
    #   �ڴ��С�ĵ��Կ�������Ϊ2����0  
    #------------------------------------------------------#
    num_workers         = 4
    #----------------------------------------------------#
    #   ���ͼƬ·���ͱ�ǩ
    #----------------------------------------------------#
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'

    #----------------------------------------------------#
    #   ��ȡclasses��anchor
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)

    #------------------------------------------------------#
    #   ����yoloģ��
    #------------------------------------------------------#
    model = YoloBody(anchors_mask, num_classes)
    weights_init(model)
    if model_path != '':
        #------------------------------------------------------#
        #   Ȩֵ�ļ��뿴README���ٶ���������
        #------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    yolo_loss    = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask)
    loss_history = LossHistory("logs/")

    #---------------------------#
    #   ��ȡ���ݼ���Ӧ��txt
    #---------------------------#
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    #------------------------------------------------------#
    #   ����������ȡ��������ͨ�ã�����ѵ�����Լӿ�ѵ���ٶ�
    #   Ҳ������ѵ�����ڷ�ֹȨֵ���ƻ���
    #   Init_EpochΪ��ʼ����
    #   Freeze_EpochΪ����ѵ��������
    #   UnFreeze_Epoch��ѵ������
    #   ��ʾOOM�����Դ治�����СBatch_size
    #------------------------------------------------------#
    if True:
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch
        
        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset   = YoloDataset(train_lines, input_shape, num_classes, train = True)
        val_dataset     = YoloDataset(val_lines, input_shape, num_classes, train = False)
        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate)
                        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("���ݼ���С���޷�����ѵ�������������ݼ���")

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()
            
    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch
        
        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset   = YoloDataset(train_lines, input_shape, num_classes, train = True)
        val_dataset     = YoloDataset(val_lines, input_shape, num_classes, train = False)
        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate)
                        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("���ݼ���С���޷�����ѵ�������������ݼ���")

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = True

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()