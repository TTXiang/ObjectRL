import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# 将父目录添加到模块搜索路径
sys.path.append(parent_dir)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


import logging

# 配置 logger
logging.basicConfig(filename='gauai.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


import os.path
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import argparse
from datasets.gauai_dataset import kernel_Dataset
import torch.nn as nn
import utilsmodel.train_distributed_utils as utils_mine
from utilsmodel.resnet34model import resnet34
import time
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import torch.nn.init as init

import random

random.seed(1000)
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.conv1(x)  # [1, 16, 750, 1024]
        x2 = self.bn1(x1)
        x3 = self.relu(x2)
        x4 = self.conv2(x3)
        x5 = self.bn2(x4)
        x6 = self.relu(x5)
        return x6

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels // 2)
            self.final_conv = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, 16)
        self.down1 = Down(16, 32 // factor)
        self.up1 = Up(32, 16, self.bilinear)

        #self.down2 = Down(32, 64)
        # self.down3 = Down(64, 64)
        #self.up2 = Up(512, 512, bilinear)
        #self.up3 = Up(512, 512, bilinear)
        #self.up4 = Up(512, 256, bilinear)
        self.outc = OutConv(8, out_channels)

    def forward(self, x):
        x1 = self.inc(x)  # [1, 16, 750, 1024]
        x2 = self.down1(x1)  # [1, 16, 325, 512]
        x = self.up1(x2, x1)  # [1, 8, 750, 1024]
        output = self.outc(x)  # [1, 1, 750, 1024]
        return output


def apply_custom_convolution_opencv(image, kernel, padding_mode='ZERO_PADDING'):
    # kernel = torch.tensor(kernel, dtype=torch.float32)
    kernel = kernel.astype(image.dtype)
    border_type = cv2.BORDER_CONSTANT
    # Get the padding size based on the kernel size
    pad_height = kernel.shape[0] // 2
    pad_width = kernel.shape[1] // 2
    # Pad the image
    padded_image = cv2.copyMakeBorder(image, pad_height, pad_height, pad_width, pad_width, border_type)

    # Perform the convolution operation using the filtered image
    convolved_image = cv2.filter2D(padded_image, -1, kernel, borderType=border_type)

    # Remove the padding from the convolved image
    convolved_image = convolved_image[pad_height:-pad_height, pad_width:-pad_width]

    return convolved_image


def normalize_array(array):
    '''
    图像归一化
    :param array: 待归一化的图像二维数组
    :return: 归一化的后的图像
    '''
    min_val = array.min()
    max_val = array.max()
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array


def generate_detection_boxes(image):
    # 初始化检测框列表
    boxes = []
    # 遍历图像中的每个像素
    for y in range(10, image.shape[0] - 10):
        for x in range(10, image.shape[1] - 10):
            # 如果像素值为255，则生成检测框
            if image[y, x] == 255:
                # 创建检测框（左上角坐标和右下角坐标）
                box = (max(x - 5, 0), max(y - 5, 0), min(x + 5, image.shape[1]), min(y + 5, image.shape[0]))
                boxes.append(box)

    return boxes


def calculate_iou(boxA, boxB):
    '''
    计算检测框之间的IOU
    :param boxA: 检测框1
    :param boxB: 检测框2
    :return: IOU值
    '''
    # 计算两个矩形框的交集面积
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    intersectionArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # 计算两个矩形框的并集面积
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    unionArea = boxAArea + boxBArea - intersectionArea

    # 计算交并比
    iou = intersectionArea / unionArea
    return iou


def merge_boxes_2(boxA, boxB):
    '''
    融合检测框
    :param boxA: 检测框1
    :param boxB: 检测框2
    :return: 合并后的检测框
    '''
    # 合并两个检测框
    return (min(boxA[0], boxB[0]), min(boxA[1], boxB[1]),
            max(boxA[2], boxB[2]), max(boxA[3], boxB[3]))


def merge_boxes(boxes, iou_threshold=0.3):
    '''
    对于 IOU 值大于一定阈值的检测框进行合并合并检
    :param boxes: 检测框列表
    :param iou_threshold: 决定是否合并的IOU阈值
    :return: 合并后的检测框列表
    '''
    # 将检测框从大到小排序（基于面积）
    sorted_boxes = sorted(boxes, key=lambda box: np.abs(box[0] - box[2]) * np.abs(box[1] - box[3]), reverse=True)
    merged_boxes = []

    while len(sorted_boxes) > 0:
        # 取出最大的检测框
        current_box = sorted_boxes.pop(0)

        # 初始化一个列表，用于存储可能与当前框合并的检测框
        boxes_to_merge = []

        # 遍历剩余的检测框，计算IoU
        for box in sorted_boxes:
            iou_value = calculate_iou(current_box, box)
            if iou_value > iou_threshold:
                # 如果IoU大于阈值，则将检测框添加到合并列表
                boxes_to_merge.append(box)

        # 如果有检测框需要与当前框合并
        if boxes_to_merge:
            # 合并所有检测框
            merged_box = current_box
            for box in boxes_to_merge:
                merged_box = merge_boxes_2(merged_box, box)
            # 将合并后的检测框添加到最终结果中
            merged_boxes.append(merged_box)
            # 从待处理列表中移除已合并的检测框
            for box in boxes_to_merge:
                sorted_boxes.remove(box)

        # 如果没有合并，则将当前检测框添加到合并后的检测框列表中
        else:
            merged_boxes.append(current_box)

    return merged_boxes


def box_parameter(boxes):
    '''
    计算检测框的相关表征参数
    :param boxes: 检测框
    :return: 输出检测框对应的表征参数，包括中心位置，两边长
    '''
    boxes_parameters = np.zeros((len(boxes), 4))
    for ii in range(len(boxes)):
        x1, y1, x2, y2 = boxes[ii]
        x_c = np.round((x1 + x2) / 2)
        y_c = np.round((y1 + y2) / 2)
        # ratio = np.abs((x2-x1)/(y2-y1))
        # diagonal = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        boxes_parameters[ii] = np.array([x_c, y_c, np.abs(x1 - x2), np.abs(y1 - y2)])

    return boxes_parameters



def box_iou(box1, box2):
    """
    计算两个边界框的IoU
    :param box1: 第一个边界框，格式为 [x1, y1, x2, y2]
    :param box2: 第二个边界框，格式为 [x1, y1, x2, y2]
    :return: IoU值
    """
    # 计算交集区域
    xi1 = np.maximum(box1[0], box2[0])
    yi1 = np.maximum(box1[1], box2[1])
    xi2 = np.minimum(box1[2], box2[2])
    yi2 = np.minimum(box1[3], box2[3])
    inter_width = np.maximum(0, xi2 - xi1)
    inter_height = np.maximum(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # 计算每个框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def box_center_distance(box1, box2):
    """
    计算两个边界框的中心点距离
    :param box1: 第一个边界框，格式为 [x1, y1, x2, y2]
    :param box2: 第二个边界框，格式为 [x1, y1, x2, y2]
    :return: 中心点距离
    """
    center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    return np.linalg.norm(np.array(center1) - np.array(center2))


def box_enclosing_area(box1, box2):
    """
    计算包含两个边界框的最小矩形框的面积
    :param box1: 第一个边界框，格式为 [x1, y1, x2, y2]
    :param box2: 第二个边界框，格式为 [x1, y1, x2, y2]
    :return: 最小矩形框的面积
    """
    x1_min = np.minimum(box1[0], box2[0])
    y1_min = np.minimum(box1[1], box2[1])
    x2_max = np.maximum(box1[2], box2[2])
    y2_max = np.maximum(box1[3], box2[3])
    return (x2_max - x1_min) * (y2_max - y1_min)


def diou_loss(box1, box2):
    iou = box_iou(box1, box2)
    if iou.item() <= 0.1:
        ob_yes = False
    else:
        ob_yes = True
    # 计算中心点距离
    center_distance = box_center_distance(box1, box2)
    # 计算包含两个边界框的最小矩形框的面积
    enclosing_area = box_enclosing_area(box1, box2)
    # 计算DIoU中的ρ² / c²
    c = enclosing_area ** 0.5
    rho2 = center_distance ** 2
    # 计算DIoU
    diou = iou - (rho2 / c ** 2)
    # 计算损失
    loss = 1 - diou
    if ob_yes == False:
        loss += 3.0
    return loss, ob_yes


def diou_loss_all(pred_boxes, target_boxes):
    # 处理误检情况（真值框为空，预测框不为空）
    if target_boxes.numel() == 0 and pred_boxes.numel() != 0:
        # 误检惩罚系数，可根据实际情况调整
        false_detection_penalty = 1.0
        # 误检惩罚项，这里简单地用一个固定值乘以预测框数量
        penalty_loss = false_detection_penalty * pred_boxes.size(0)
        return torch.tensor(penalty_loss, dtype=torch.float32, requires_grad=True)
    # 处理漏检情况（预测框为空，真值框不为空）
    elif pred_boxes.numel() == 0 and target_boxes.numel() != 0:
        # 漏检惩罚系数，可根据实际情况调整
        miss_detection_penalty = 1.0
        # 漏检惩罚项，用固定值乘以真值框数量
        penalty_loss = miss_detection_penalty * target_boxes.size(0)
        return torch.tensor(penalty_loss, dtype=torch.float32, requires_grad=True)
    # 处理预测框和真值框都为空的情况
    elif pred_boxes.numel() == 0 and target_boxes.numel() == 0:
        return torch.tensor(0., requires_grad=True)

    N = pred_boxes.size(0)
    M = target_boxes.size(0)

    pred_boxes = pred_boxes.unsqueeze(1).expand(N, M, 4)
    target_boxes = target_boxes.unsqueeze(0).expand(N, M, 4)

    # 计算 IoU
    x1 = torch.max(pred_boxes[..., 0], target_boxes[..., 0])
    y1 = torch.max(pred_boxes[..., 1], target_boxes[..., 1])
    x2 = torch.min(pred_boxes[..., 2], target_boxes[..., 2])
    y2 = torch.min(pred_boxes[..., 3], target_boxes[..., 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area_pred = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    area_target = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
    union = area_pred + area_target - intersection
    iou = intersection / (union + 1e-6)  # 加一个小的常数避免除零

    # 计算预测框和真实框中心点的坐标
    pred_center_x = (pred_boxes[..., 0] + pred_boxes[..., 2]) / 2
    pred_center_y = (pred_boxes[..., 1] + pred_boxes[..., 3]) / 2
    target_center_x = (target_boxes[..., 0] + target_boxes[..., 2]) / 2
    target_center_y = (target_boxes[..., 1] + target_boxes[..., 3]) / 2

    # 计算中心点之间的欧氏距离
    d = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2

    # 计算能够同时包含预测框和真实框的最小外接矩形的对角线长度
    c_x1 = torch.min(pred_boxes[..., 0], target_boxes[..., 0])
    c_y1 = torch.min(pred_boxes[..., 1], target_boxes[..., 1])
    c_x2 = torch.max(pred_boxes[..., 2], target_boxes[..., 2])
    c_y2 = torch.max(pred_boxes[..., 3], target_boxes[..., 3])
    c = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2

    # 计算 DIOU 损失
    diou = iou - d / (c + 1e-6)
    diou_loss = 1 - diou

    return diou_loss.mean()








def train_one_epoch(preUNet, clsNet, optimizer, data_loader, device, epoch, tb_writer, warmup=False, scaler=None):
    tags = ["single_frame_loss", "single_video_loss", "single_video_acc", "single_epoch_loss", "single_epoch_acc"]

    metric_logger = utils_mine.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils_mine.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_epoch = 0.0

    preUNet.apply(initialize_weights)
    preUNet.train()
    kernel = np.array([[3, 3, 3], [3, 1, 3], [3, 3, 3]])

    acc_loss = torch.zeros(1).to(device)  # mean losses
    videos_num = len(data_loader)

    for i, [images, targets, _, _] in enumerate(data_loader):
        optimizer.zero_grad()
        for tup_labels in targets:
            tup_targets = tup_labels

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            subtractor = {}
            if targets != None:
                targets = targets[0]
            for tup in images:
                # tup: 单个视频，多帧，比如5帧
                video_loss = 0.0
                video_acc = 0
                for index in range(len(tup)):
                    output_num = []
                    mask = 0
                    mask_9 = np.zeros([8, 750, 1024], dtype=np.uint8)
                    mask_index = 0
                    if index < 1:
                        frame = tup[index]
                        for key in frame:
                            subtractor[key] = cv2.createBackgroundSubtractorMOG2()

                    frame = tup[index]

                    for key in frame:
                        fgmask = subtractor[key].apply(frame[key])
                        fgmask = np.ascontiguousarray(fgmask)

                        if fgmask.max() == fgmask.min():
                            # 如果相等，说明数组中所有元素都相同，将 fgmask 设为 0 或 255，这里设为 0 示例
                            fgmask = np.zeros_like(fgmask, dtype=np.uint8)
                        else:
                            fgmask = ((fgmask - fgmask.min()) * (255.0 / (fgmask.max() - fgmask.min()))).round().astype(
                                    np.uint8)
                        mask = mask + fgmask.astype(np.float32)

                        mask[mask >= 255] = 255
                        mask_9[mask_index, :, :] = mask
                        mask_index += 1

                    mask_9 = torch.tensor(mask_9)
                    mask_9 = mask_9.reshape(1, 8, fgmask.shape[0], fgmask.shape[1])
                    x_input = mask_9.to(device)
                    x_input = x_input.to(torch.float32)  # [1, 1, 1024, 750]
                    x_input = preUNet(x_input)
                    mask_9 = x_input.reshape(x_input.shape[2], x_input.shape[3])
                    mask_9 = mask_9.to('cpu').detach().numpy().astype(np.float32)
                    convolved_image = apply_custom_convolution_opencv(mask_9, kernel, padding_mode='ZERO_PADDING')
                    convolved_image = (255 * normalize_array(convolved_image ** 2)).astype(np.uint8)

                    if index < 1:
                        pass
                    else:
                        accu_single = 0  # 检测的准确率mAP可以在函数中选择
                        _, binary_image = cv2.threshold(convolved_image, 60, 255, cv2.THRESH_BINARY)
                        # 生成检测框
                        # 对输入的二值化图像产生检测框
                        boxes = generate_detection_boxes(binary_image)  # 对输入的二值化图像产生检测框
                        iou_threshold = 0.1

                        for k in range(3):
                            boxes = merge_boxes(boxes, iou_threshold)
                            ious = np.zeros((len(boxes), len(boxes)))
                            for i in range(len(boxes)):
                                for j in range(i + 1, len(boxes)):
                                    ious[i, j] = calculate_iou(boxes[i], boxes[j])

                        merged_boxes = boxes
                        # Ima_cam_color = frame['color']
                        Ima_cam_color2 = frame['Cam5']

                        # 展示图像
                        # for index_boxes in range(len(merged_boxes)):
                        #     # print('merged boxes index is' + str(index_boxes))
                        #     x1, y1, x2, y2 = merged_boxes[index_boxes]
                        #     color = (0, 0, 255)  # 红色
                        #     thickness = 1
                        #     cv2.rectangle(Ima_cam_color, (x1, y1), (x2, y2), color, thickness)
                        # cv2.imshow('Training Images', Ima_cam_color)
                        # cv2.waitKey(100)
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break

                        # 展示合并后的检测框
                        for index_boxes in range(len(merged_boxes)):
                            x1, y1, x2, y2 = merged_boxes[index_boxes]
                            thickness = 1
                            # Ima_cam_colorc = cv2.resize(Ima_cam_color, (1024 * 4, 750 * 4))
                            # box_image_one = Ima_cam_colorc[y1 * 4:y2 * 4, x1 * 4:x2 * 4, :]
                            ssize = (y2 - y1) * (x2 - x1)

                            if ssize <= 50 or ssize >= 1200:  # 预先删除运动较大的或超级小的目标，加速分类
                                continue
                            else:
                                # box_image = Image.fromarray(box_image_one)

                                data_transform = transforms.Compose(
                                    [transforms.Resize(224),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

                                # box_image = data_transform(box_image)
                                # box_image = torch.unsqueeze(box_image, dim=0)
                                pred_value = 1.0

                                # output = torch.squeeze(clsNet(box_image.to(device))).cpu()
                                # pred_value, pred_cls = torch.max(output, dim=0)

                                if True:
                                # if int(pred_cls) == 0:  # 判断目标是否是无人机, train preNet可以不用
                                    output_box = {'xyxy': [x1, y1, x2, y2], 'pred_value': pred_value}
                                    output_num.append(output_box)
                                    color2 = (0)
                                    cv2.rectangle(Ima_cam_color2, (x1, y1), (x2, y2), color2, thickness)


                            # single frame 计算结果
                            # cv2.imshow('PIC2 Images', Ima_cam_color2)


                        mloss = 0.0  # 每一帧的损失loss
                        d_loss = 0.0  # 求每个检测框的损失
                        labels = tup_targets[index]  # 该帧的真值标签
                        nL = len(output_num)  # 该帧输出检测框数量(用于评估检测速度)
                        # print('This video nL:' + str(nL))
                        ob_dets = False  # 看看这一帧是否检测到目标(有目标的情况下检测到就是True, 没检测到就是False)
                        if nL:
                            for box_index in range(nL):
                                pred = output_num[box_index]['xyxy']
                                x_center = (labels[0, 2] * 1024).round()
                                y_center = (labels[0, 3] * 750).round()
                                w = (labels[0, 4] * 1024 / 2).round()
                                h = (labels[0, 5] * 750 / 2).round()
                                xyxy2 = [x_center - w, y_center - h, x_center + w, y_center + h]
                                d_loss, ob_det = diou_loss(pred, xyxy2)
                                mloss += d_loss
                                if ob_det:
                                    ob_dets = True
                                    accu_single += 1
                                    # print('d_loss: ' + str(d_loss))
                                else:
                                    # plenaty
                                    # 该框是虚警框
                                    d_loss += 0.5
                                d_loss =  torch.tensor(d_loss)
                                d_loss1 = d_loss.clone().detach()
                                # d_loss1 = torch.tensor(d_loss, dtype=torch.float32)
                                d_loss1.requires_grad_(True)
                                d_loss1.backward()

                            if not ob_dets:
                                # 该框是虚警框
                                d_loss = d_loss + 2.0
                                mloss += 10.0
                                d_loss =  torch.tensor(d_loss)
                                d_loss1 = d_loss.clone().detach()
                                # d_loss1 = torch.tensor(d_loss, dtype=torch.float32)
                                # print('d_loss: ' + str(d_loss))
                                d_loss1.requires_grad_(True)
                                d_loss1.backward()

                        elif not nL and labels.numel() == 0:
                            # 这一帧没目标且也没虚警目标
                            accu_single += 1
                            d_loss = 0.0  # 便于后续漏检损失loss的增续
                            d_loss1 = torch.tensor(0., requires_grad=True)
                            d_loss1.backward()
                            mloss += 0.0

                        else:
                            # 这一帧有目标但漏掉, 要加上一点点的penalty
                            # 为了防止扩散, 可以加上一点d_loss
                            d_loss = d_loss + 2.0
                            mloss += 15.0
                            d_loss =  torch.tensor(d_loss)
                            d_loss1 = d_loss.clone().detach()
                            # d_loss1 = torch.tensor(d_loss, dtype=torch.float32)
                            d_loss1.requires_grad_(True)
                            d_loss1.backward()

                        # accu_single = float(accu_single) / 19.0
                        video_acc += accu_single
                        tb_writer.add_scalar(tags[0], float(mloss), epoch)

                        video_loss += mloss
                        # print_loss = mloss / (len(output_num) - 1)
                        # print("acc_frame:" + str(accu_single))
                        # print('single frame loss:' + str(mloss))
                        optimizer.step()


                # video results
                tb_writer.add_scalar(tags[1], float(video_loss / 19.0), i)
                tb_writer.add_scalar(tags[2], float(float(video_acc) / 19.0), i)
                acc_loss += video_loss / 19.0
                accu_num += float(video_acc) / 19.0
                # print('single video loss:' + str(video_loss))
                print('single video acc:' + str(float(video_acc) / 19.0))
                logger.info('single video acc:' + str(float(video_acc) / 19.0))


        # epoch loss
        avg_loss = float(acc_loss.item()/videos_num)
        # epoch acc
        tb_writer.add_scalar(tags[3], float(avg_loss), epoch)
        tb_writer.add_scalar(tags[4], float(accu_num.item()), epoch)

    return avg_loss, accu_epoch





@torch.no_grad()
def evaluate(preUNet, clsNet, data_loader, device, epoch, tb_writer):
    tags = ["val_video_loss", "val_video_acc", "val_epoch_loss", "val_epoch_acc"]
    preUNet.eval()
    kernel = np.array([[3, 3, 3], [3, 1, 3], [3, 3, 3]])
    acc_loss = torch.zeros(1).to(device)  # mean losses
    accu_epoch = 0.0
    videos_num = len(data_loader)
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数

    for i, [images, targets, _, _] in enumerate(data_loader):
        for tup_labels in targets:
            tup_targets = tup_labels
            subtractor = {}
            if targets != None:
                targets = targets[0]
            for tup in images:
                video_loss = 0.0
                video_acc = 0
                for index in range(len(tup)):
                    output_num = []
                    mask = 0
                    mask_9 = np.zeros([8, 750, 1024], dtype=np.uint8)
                    mask_index = 0
                    if index < 1:
                        frame = tup[index]
                        for key in frame:
                            subtractor[key] = cv2.createBackgroundSubtractorMOG2()

                    frame = tup[index]
                    for key in frame:
                        fgmask = subtractor[key].apply(frame[key])
                        fgmask = np.ascontiguousarray(fgmask)
                        
                        if fgmask.max() == fgmask.min():
                            # 如果相等，说明数组中所有元素都相同，将 fgmask 设为 0 或 255，这里设为 0 示例
                            fgmask = np.zeros_like(fgmask, dtype=np.uint8)
                        else:
                            fgmask = ((fgmask - fgmask.min()) * (255.0 / (fgmask.max() - fgmask.min()))).round().astype(
                                    np.uint8)

                        mask = mask + fgmask.astype(np.float32)
                        mask[mask >= 255] = 255
                        mask_9[mask_index, :, :] = mask
                        mask_index += 1

                    mask_9 = torch.tensor(mask_9)
                    mask_9 = mask_9.reshape(1, 8, fgmask.shape[0], fgmask.shape[1])
                    x_input = mask_9.to(device)
                    x_input = x_input.to(torch.float32)  # [1, 1, 1024, 750]
                    x_input = preUNet(x_input)
                    mask_9 = x_input.reshape(x_input.shape[2], x_input.shape[3])
                    mask_9 = mask_9.to('cpu').detach().numpy().astype(np.float32)
                    convolved_image = apply_custom_convolution_opencv(mask_9, kernel, padding_mode='ZERO_PADDING')
                    convolved_image = (255 * normalize_array(convolved_image ** 2)).astype(np.uint8)

                    if index < 1:
                        pass
                    else:
                        accu_single = 0  # 检测的准确率mAP可以在函数中选择
                        _, binary_image = cv2.threshold(convolved_image, 60, 255, cv2.THRESH_BINARY)
                        # 生成检测框
                        # 对输入的二值化图像产生检测框
                        boxes = generate_detection_boxes(binary_image)  # 对输入的二值化图像产生检测框
                        iou_threshold = 0.1

                        for k in range(3):
                            boxes = merge_boxes(boxes, iou_threshold)
                            ious = np.zeros((len(boxes), len(boxes)))
                            for i in range(len(boxes)):
                                for j in range(i + 1, len(boxes)):
                                    ious[i, j] = calculate_iou(boxes[i], boxes[j])

                        merged_boxes = boxes
                        # Ima_cam_color = frame['color']
                        Ima_cam_color2 = frame['Cam5']

                        # 展示图像
                        # for index_boxes in range(len(merged_boxes)):
                        #     # print('merged boxes index is' + str(index_boxes))
                        #     x1, y1, x2, y2 = merged_boxes[index_boxes]
                        #     color = (0, 0, 255)  # 红色
                        #     thickness = 1
                        #     cv2.rectangle(Ima_cam_color, (x1, y1), (x2, y2), color, thickness)
                        # cv2.imshow('Training Images', Ima_cam_color)
                        # cv2.waitKey(100)
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break

                        # 展示合并后的检测框
                        for index_boxes in range(len(merged_boxes)):
                            x1, y1, x2, y2 = merged_boxes[index_boxes]
                            thickness = 1
                            # Ima_cam_colorc = cv2.resize(Ima_cam_color, (1024 * 4, 750 * 4))
                            # box_image_one = Ima_cam_colorc[y1 * 4:y2 * 4, x1 * 4:x2 * 4, :]
                            ssize = (y2 - y1) * (x2 - x1)

                            if ssize <= 50 or ssize >= 600:  # 预先删除运动较大的或超级小的目标，加速分类
                                continue
                            else:
                                # box_image = Image.fromarray(box_image_one)

                                data_transform = transforms.Compose(
                                    [transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485], [0.224])])

                                # box_image = data_transform(box_image)
                                # box_image = torch.unsqueeze(box_image, dim=0)
                                # output = torch.squeeze(clsNet(box_image.to(device))).cpu()
                                # pred_value, pred_cls = torch.max(output, dim=0)
                                pred_value = 1.0
                                pred_cls = 0

                                # if int(pred_cls) != 2:  # 判断目标是否是无人机
                                if True:
                                    output_box = {'xyxy': [x1, y1, x2, y2], 'pred_value': pred_value}
                                    output_num.append(output_box)
                                    color2 = (0)
                                    cv2.rectangle(Ima_cam_color2, (x1, y1), (x2, y2), color2, thickness)

                        # single frame 计算结果
                        # print(output_num)
                        # cv2.imshow('PIC2 Images', Ima_cam_color2)


                        mloss = 0.0  # 每一帧的损失loss
                        d_loss = 0.0  # 求每个检测框的损失
                        labels = tup_targets[index]  # 该帧的真值标签
                        nL = len(output_num)  # 该帧输出检测框数量(用于评估检测速度)
                        print('This video nL:' + str(nL))
                        logger.info('This video nL:' + str(nL))
                        ob_dets = False  # 看看这一帧是否检测到目标(有目标的情况下检测到就是True, 没检测到就是False)
                        if nL:
                            for box_index in range(nL):
                                pred = output_num[box_index]['xyxy']
                                x_center = (labels[0, 2] * 1024).round()
                                y_center = (labels[0, 3] * 750).round()
                                w = (labels[0, 4] * 1024 / 2).round()
                                h = (labels[0, 5] * 750 / 2).round()
                                xyxy2 = [x_center - w, y_center - h, x_center + w, y_center + h]
                                d_loss, ob_det = diou_loss(pred, xyxy2)
                                mloss += d_loss
                                # 看看检测到目标没
                                if ob_det:
                                    ob_dets = True
                                    accu_single += 1
                                    print('d_loss: ' + str(d_loss))
                                    logger.info('d_loss: ' + str(d_loss))
                                else:
                                    # plenaty
                                    # 该框是虚警框
                                    d_loss += 0.5

                            if not ob_dets:
                                # 该框是虚警框
                                d_loss = d_loss + 2.0
                                mloss += 10.0
                                print('d_loss: ' + str(d_loss))
                                logger.info('d_loss: ' + str(d_loss))


                        elif not nL and labels.numel() == 0:
                            # 这一帧没目标且也没虚警目标
                            accu_single += 1
                            d_loss = 0.0  # 便于后续漏检损失loss的增续
                            mloss += 0.0

                        else:
                            # 这一帧有目标但漏掉, 要加上一点点的penalty
                            # 为了防止扩散, 可以加上一点d_loss
                            d_loss = d_loss + 2.0
                            mloss += 15.0


                        video_loss += mloss
                        video_acc += accu_single


                # video results
                acc_loss += video_loss
                accu_num += float(video_acc) / 19.0
                # print_loss = mloss / (len(output_num) - 1)
                print("val_acc_video:" + str(video_acc))
                print('val single video loss:' + str(video_loss))
                logger.info("val_acc_video:" + str(video_acc))
                logger.info('val single video loss:' + str(video_loss))
                tb_writer.add_scalar(tags[0], float(video_loss), i)
                tb_writer.add_scalar(tags[1], float(float(video_acc) / 19.0), i)


            # epoch loss
            avg_loss = float(acc_loss.item() / videos_num)
            # epoch acc
            tb_writer.add_scalar(tags[2], float(avg_loss), epoch)
            tb_writer.add_scalar(tags[3], float(accu_num), epoch)

        return avg_loss, accu_epoch



def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 或者使用 init.kaiming_uniform_，但需要调整参数 a
        # init.kaiming_uniform_(m.weight, a=math.sqrt(5), mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # 偏置通常初始化为0




def kernel_detect(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))
    logger.info(print("Using {} device training.".format(device.type)))

    # results_file
    tb_writer = SummaryWriter()

    # data_path
    # train_data_path = os.path.join(args.data_path, 'train')
    # val_data_path = os.path.join(args.data_path, 'val')
    # 回头自己划分一下, 找一下划分文件
    train_data_path = args.data_path
    val_data_path = args.data_path

    # check root
    if os.path.exists(train_data_path) is False:
        raise FileNotFoundError("train_data_path dose not in path:'{}'.".format(train_data_path))
    if os.path.exists(val_data_path) is False:
        raise FileNotFoundError("val_data_path dose not in path:'{}'.".format(val_data_path))

    total_seconds = time.time()
    # dataset
    train_dataset = kernel_Dataset(train_data_path, rank=-1, single_cls=True,
                                ratio=args.ratio, image_size=(4096, 3000), logger=logger)

    val_dataset = kernel_Dataset(val_data_path, rank=-1, single_cls=True,
                                ratio=args.ratio, image_size=(4096, 3000), logger=logger)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    logger.info('Using %g dataloader workers' % nw)

    # print(len(train_dataset))
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=nw,
                                                    collate_fn=train_dataset.collate_fn)
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=nw,
                                                    collate_fn=val_dataset.collate_fn)
    total_seconds1 = time.time()
    minutes_1 = (total_seconds1 - total_seconds) // 60
    print("Dataloader resumes " + str(minutes_1) + 'mins')
    logger.info(print("Dataloader resumes " + str(minutes_1) + 'mins'))


    # create model
    # 注意：不包含背景
    clsNet = resnet34(num_classes=3)
    clsNet.to(device)
    clsweights_path = os.path.join(ROOT, 'weights/cls3.pth')
    assert os.path.exists(clsweights_path), "file: '{}' dose not exist.".format(clsweights_path)
    clsNet.load_state_dict(torch.load(clsweights_path, map_location=device))
    clsNet.eval()


    preUNet = UNet(in_channels=8, out_channels=1)
    preUNet.to(device)
    params = [p for p in preUNet.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01,
                                momentum=0.9, weight_decay=0.005)
    scaler = torch.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.33)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location=device)
        preUNet.load_state_dict(checkpoint)
        args.start_epoch = 5
        # preUNet.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # args.start_epoch = checkpoint['epoch'] + 1
        # if args.amp and "scaler" in checkpoint:
        #     scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(args.start_epoch))
        logger.info("the training process from epoch{}...".format(args.start_epoch))

    best_loss = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        time_2 = time.time()
        mean_loss, mean_acc = train_one_epoch(preUNet, clsNet, optimizer,
                                    train_data_loader, device,
                                    epoch, tb_writer, warmup=True, scaler=scaler)
        print('train: epoch ' + str(epoch) + ' loss: ' + str(mean_loss))
        print('train: epoch ' + str(epoch) + ' acc: ' + str(mean_acc))
        logger.info('train: epoch ' + str(epoch) + ' loss: ' + str(mean_loss))
        logger.info('train: epoch ' + str(epoch) + ' acc: ' + str(mean_acc))

        total_seconds2 = time.time()
        minutes_2 = (total_seconds2 - time_2) // 60
        print("Training epoch resumes" + str(minutes_2) + 'mins')
        logger.info("Training epoch resumes" + str(minutes_2) + 'mins')
        torch.save(preUNet.state_dict(), "./weights/model-kernelnet-{}.pth".format(epoch + 1))


        # validate
        val_loss, val_acc = evaluate(preUNet=preUNet, clsNet=clsNet, data_loader=val_data_loader, device=device, epoch=epoch, tb_writer=tb_writer)
        print('val: epoch ' + str(epoch) + ' loss: ' + str(val_loss))
        print('val: epoch ' + str(epoch) + ' acc: ' + str(val_acc))
        logger.info('val: epoch ' + str(epoch) + ' loss: ' + str(val_loss))
        logger.info('val: epoch ' + str(epoch) + ' acc: ' + str(val_acc))

        # if epoch == 0:
        #     best_loss = val_loss.item()
        #     torch.save(preUNet.state_dict(), "model-kernelnet-best.pth")
        #
        # else:
        #     if best_loss >= val_loss.item():
        #         best_loss = val_loss.item()
        #         print("best loss is" + str(best_loss))
        #         torch.save(preUNet.state_dict(), "model-kernelnet-best.pth")

        tags = ["val_loss", "learning_rate"]
        tb_writer.add_scalar(tags[0], float(val_loss), epoch)
        tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)
        # 每个epoch结束后更新学习率
        lr_scheduler.step()




if __name__ == '__main__':
    # kernel
    ROOT = os.getcwd()
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--ratio', type=int, default=4)
    parser.add_argument('--data-path', type=str,
                       default=r"/mnt/nfs_200T/optics/data/middlepics")
    # parser.add_argument('--data-path', type=str,
    #                     default=r"\\10.6.4.46\data\DataSet\easypics")
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights_path', type=str, default=None, help='initial weights path')
    parser.add_argument('--device', default=0, help='device id (i.e. 0 or 0,1 or cpu)')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default=r'weights/model-kernelnet-6.pth', type=str, help='resume from checkpoint')


    args = parser.parse_args()
    print(args)
    logger.info(args)
    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # cv2.namedWindow('Training Images', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Training Images', 1024, 750)
    # cv2.namedWindow('PIC2 Images', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('PIC2 Images', 1024, 750)
    kernel_detect(args)

    # 释放窗口
    # cv2.destroyAllWindows()

