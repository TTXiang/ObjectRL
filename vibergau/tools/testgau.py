import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# 将父目录添加到模块搜索路径
sys.path.append(parent_dir)

import os.path
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import argparse
from datasets.dataset_viber import kernel_Dataset
import torch.nn as nn
import torch
import torch.nn.functional as F
from utilsmodel.utilsgpu import ImageProcessor


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=3, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)




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

        # self.down2 = Down(32, 64)
        # self.down3 = Down(64, 64)
        # self.up2 = Up(512, 512, bilinear)
        # self.up3 = Up(512, 512, bilinear)
        # self.up4 = Up(512, 256, bilinear)
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


@torch.no_grad()
def evaluate(clsNet, data_loader, device):
    processor = ImageProcessor()
    matrices = np.tile(np.eye(3), (8, 1, 1)).astype(np.float32)

    for i, [images, targets, _, _] in enumerate(data_loader):
        if targets != None:
            targets = targets[0]
        for tup in images:
            for index in range(len(tup)):
                output_num = []
                output_none = []
                
                frame = tup[index]
                mask, rgb_img, rgb_img_show, boxes = processor.process(frame, matrices)

                if index < 1:
                    pass
                else:
                    merged_boxes = boxes
                    Ima_cam_color = rgb_img_show
                    Ima_cam_color2 = rgb_img

                    #  展示一下未经处理的原始图像
                    # cv2.imshow('Row Images', Ima_cam_color)
                    # cv2.waitKey(100)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break

                    # -------------------------------------------------------
                    for index_boxes in range(len(merged_boxes)):
                        x1, y1, x2, y2 = merged_boxes[index_boxes]
                        color = (0, 0, 255)  # 红色
                        thickness = 1
                        #  故意把框画大点
                        cv2.rectangle(rgb_img_show, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), color, thickness)

                    #  展示一下Viber建模后的原始图像
                    # cv2.imshow('Viber Images', rgb_img_show)
                    
                    # -------------------------------------------------------
                    # 分个类
                    for index_boxes in range(len(merged_boxes)):
                        x1, y1, x2, y2 = merged_boxes[index_boxes]
                        box_image_one = Ima_cam_color2[int(y1*8):int(y2*8), int(x1*8):int(x2*8), :].astype(np.uint8)
                        box_image = Image.fromarray(box_image_one)
                        data_transform = transforms.Compose(
                            [transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

                        box_image = data_transform(box_image)
                        box_image = torch.unsqueeze(box_image, dim=0)
                        output = torch.squeeze(clsNet(box_image.to(device))).cpu()
                        pred_value, pred_cls = torch.max(output, dim=0)
                        output_box1 = {'xyxy': [x1, y1, x2, y2], 'pred_value': pred_value}
                        output_none.append(output_box1)

                        # 画图!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        if int(pred_cls) != 2:  # 判断目标是否是无人机
                        # if True:  # 判断目标是否是无人机
                            output_box = {'xyxy': [x1+3, y1+3, x2-3, y2-3], 'pred_value': pred_value}
                            output_num.append(output_box)
                            color2 = (0)
                            cv2.rectangle(Ima_cam_color, (x1, y1), (x2, y2), color2, thickness)

                    # cv2.imshow('Cls Images', Ima_cam_color)


def infergauai(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device validating.".format(device.type))

    # data_path
    val_data_path = args.data_path
    if os.path.exists(val_data_path) is False:
        raise FileNotFoundError("val_data_path dose not in path:'{}'.".format(val_data_path))

    # dataset
    val_dataset = kernel_Dataset(val_data_path, rank=-1, single_cls=True,
                                 ratio=args.ratio, image_size=(4096, 3000))

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=val_dataset.collate_fn)
    
    # create model
    # 注意：不包含背景
    clsNet = resnet34(num_classes=3)
    clsNet.to(device)
    clsweights_path = args.cls_weights_path
    assert os.path.exists(clsweights_path), "file: '{}' dose not exist.".format(clsweights_path)
    clsNet.load_state_dict(torch.load(clsweights_path, map_location=device))
    clsNet.eval()

    
    preUNet = UNet(in_channels=9, out_channels=1)
    preUNet.to(device)
    preUNet_path = args.gau_weights_path
    assert os.path.exists(preUNet_path), "file: '{}' dose not exist.".format(preUNet_path)
    preUNet.load_state_dict(torch.load(preUNet_path, map_location=device))
    preUNet.eval()
    
    # validate
    evaluate(clsNet=clsNet, data_loader=val_data_loader, device=device)



if __name__ == '__main__':
    # kernel
    ROOT = os.getcwd()
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--ratio', type=int, default=1)
    parser.add_argument('--data-path', type=str,
                        default=r"vibergau/testpics")
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--cls_weights_path', type=str, default=r"weights/cls3.pth", help='weights path')
    parser.add_argument('--gau_weights_path', type=str, default=r"weights/kernel-best.pth", help='weights path')
    parser.add_argument('--device', default=0, help='device id (i.e. 0 or 0,1 or cpu)')
    # 文件保存地址
    parser.add_argument('--output-pics', default=r'outputs/gauoutputs', help='path where to save')
    args = parser.parse_args()
    print(args)
    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_pics):
        os.makedirs(args.output_pics)

    infergauai(args)

