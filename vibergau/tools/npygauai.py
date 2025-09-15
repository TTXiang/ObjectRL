import numpy as np
import torch
import torch.nn as nn
import os
import cv2
import numba
import warnings
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
warnings.filterwarnings("ignore", category=DeprecationWarning)
# 忽略 RuntimeWarning 警告
warnings.filterwarnings("ignore", category=RuntimeWarning)
color_show = (0, 0, 255)
thickness = 1


@numba.jit(nopython=True, parallel=True)
def resize_images(matrix):
    num_images = matrix.shape[0]
    num_channels = matrix.shape[1]
    new_height = 750
    new_width = 1024
    resized_matrix = np.zeros((num_images, num_channels, new_height, new_width), dtype=np.uint8)

    for i in numba.prange(num_images):
        for j in range(num_channels):
            # 这里不能直接用 cv2.resize，需将其逻辑拆分
            img = matrix[i, j]
            # 简单的缩放示例，这里实际需要实现 resize 逻辑
            # 为简化，这里只是示例，实际要实现完整的插值算法
            for y in range(new_height):
                for x in range(new_width):
                    src_y = int(y * img.shape[0] / new_height)
                    src_x = int(x * img.shape[1] / new_width)
                    resized_matrix[i, j, y, x] = img[src_y, src_x]

    return resized_matrix





def LoadImages(filepath="/mnt/nfs_200T/optics/data/test_images/img/"):
    imgs = np.zeros((99, 8, 3000, 4096), dtype=np.uint8)
    for ii in range(8):
        file_name = filepath + 'Cam' + str(ii) + '.npy'
        imgs[:, int(ii), :, :] = np.load(file_name)
    print('载入数据完成')
    
    #  载入后再做一次resize
    resized_matrix = resize_images(imgs)
    
    return resized_matrix








#UNet---------------------------------------------------------------------------
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



#Cls------------------------------------------------------------------------------------------
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

        self.conv1 = nn.Conv2d(num_classes, self.in_channel, kernel_size=7, stride=2,
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


#-----与建模相关的函数------------------------------------------------
def apply_custom_convolution_opencv(image, kernel):
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








#-----------------------快删!!!!--------------------------------------
def tensor_to_uint8_img(x: torch.Tensor) -> np.ndarray:
    """
    x: [N, C, H, W] 或 [C, H, W] 或 [H, W] 的 torch.Tensor
    返回: uint8 的灰度或 BGR numpy 图像，满足 cv2.imwrite
    """
    # 取到 CPU 并去梯度
    x = x.detach().to('cpu')

    # squeeze 到 [H, W] 或 [C, H, W]
    if x.ndim == 4:
        x = x[0]          # 取第一个样本 -> [C, H, W]
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]          # 单通道 -> [H, W]

    x = x.float().clone()
    # 处理异常值
    x[~torch.isfinite(x)] = 0

    # 线性归一化到 [0,1]
    xmin, xmax = x.min(), x.max()
    if float(xmax - xmin) < 1e-12:
        x = torch.zeros_like(x)  # 常数图，直接置零
    else:
        x = (x - xmin) / (xmax - xmin)

    # 转为 numpy
    img = (x.numpy() * 255.0).round().astype(np.uint8)

    # 如果是 3 通道 [3, H, W]，需要从 CHW -> HWC，并从 RGB -> BGR
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))        # HWC
        img = img[:, :, ::-1]                     # RGB -> BGR 以适配 cv2

    # 单通道时已经是 [H, W]
    return img









#----------------evalute-----------------------------------------------------------
def evaluate(preUNet, clsNet, imgs, device, subtractor):
    for ii in range(98):
        frame = np.squeeze(imgs[ii+1]).astype(np.uint8)
        mask_9 = np.zeros([8, 750, 1024], dtype=np.uint8)
        output_num = []
        output_num1 = []
        mask = 0
        
        for i in range(8):
            fgmask = subtractor[i].apply(frame[i])
            fgmask = np.ascontiguousarray(fgmask)
            fgmask = ((fgmask - fgmask.min()) * (255.0 / (fgmask.max() - fgmask.min()))).round().astype(np.uint8)
            # cv2.imwrite(f"mask_ch{i}.png", fgmask)
            mask = mask + fgmask.astype(np.float32)
            mask[mask >= 255] = 255
            mask_9[i, :, :] = mask
        
        mask_9 = torch.tensor(mask_9)
        mask_9 = mask_9.reshape(1, 8, fgmask.shape[0], fgmask.shape[1])
        x_input = mask_9.to(device)
        x_input = x_input.to(torch.float32)  # [1, 1, 1024, 750]
        x_input = preUNet(x_input)
        # x_input = tensor_to_uint8_img(x_input)
        # cv2.imwrite(f"preUNet.png", x_input)
        mask_9 = x_input.reshape(x_input.shape[2], x_input.shape[3])
        mask_9 = mask_9.to('cpu').detach().numpy().astype(np.float32)
        convolved_image = apply_custom_convolution_opencv(mask_9, kernel)
        convolved_image = (255 * normalize_array(convolved_image ** 2)).astype(np.uint8)
        # cv2.imwrite(f"concolved.png", convolved_image)
        _, binary_image = cv2.threshold(convolved_image, 60, 255, cv2.THRESH_BINARY)
        # cv2.imwrite(f"binary.png", binary_image)
        boxes = generate_detection_boxes(binary_image)  # 对输入的二值化图像产生检测框
        iou_threshold = 0.1

        for k in range(3):
            boxes = merge_boxes(boxes, iou_threshold)
            ious = np.zeros((len(boxes), len(boxes)))
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    ious[i, j] = calculate_iou(boxes[i], boxes[j])
            
        merged_boxes = boxes
        Ima_cam_color2 = np.stack((frame[4], frame[6], frame[7]), axis=-1)
        Ima_cam_colorc = cv2.resize(Ima_cam_color2, (512, 375))
        # 展示合并后的检测框
        for index_boxes in range(len(merged_boxes)):
            x1, y1, x2, y2 = merged_boxes[index_boxes]
            # Ima_cam_colorc = cv2.resize(Ima_cam_color2, (1024 * 4, 750 * 4))
            box_image_one = Ima_cam_color2[y1:y2, x1:x2, :]
            ssize = (y2 - y1) * (x2 - x1)

            if ssize <= 50 or ssize >= 1200:  # 预先删除运动较大的或超级小的目标，加速分类
                continue
            else:
                box_image = Image.fromarray(box_image_one)

                data_transform = transforms.Compose(
                                [transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

                box_image = data_transform(box_image)
                box_image = torch.unsqueeze(box_image, dim=0)
                output = torch.squeeze(clsNet(box_image.to(device))).cpu()
                _, pred_cls = torch.max(output, dim=0)

                # box_image = data_transform(box_image)
                # box_image = torch.unsqueeze(box_image, dim=0)
                # output = torch.squeeze(clsNet(box_image.to(device))).cpu()
                # _, pred_cls = torch.max(output, dim=0)

                if int(pred_cls) != 2:  # 判断目标是否是无人机
                    # output_box = {'xyxy': [x1, y1, x2, y2], 'pred_value': pred_value}
                    # output_num.append([x1, y1, x2, y2])
                    # output_num1.append([int(x1/2), int(y1/2), int(x2/2), int(y2/2)])
                    color_show = (0, 0, 255)
                    thickness = 1
                    cv2.rectangle(Ima_cam_colorc, (int(x1/2), int(y1/2)), (int(x2/2), int(y2/2)), color_show, thickness)
                    # cv2.rectangle(Ima_cam_color2, (x1, y1), (x2, y2), color2, thickness)
        
        # 打印所有的目标框
        cv2.imwrite(os.path.join('vibeoutputs', 'gauoutputs', str(ii)+'.jpg'), Ima_cam_colorc)
        # cv2.imshow('detect', Ima_cam_colorc)
        # cv2.waitKey(1)
        
        # print(output_num1)







if __name__ == "__main__":
    device = torch.device(0 if torch.cuda.is_available() else "cpu")
    print("Using {} device validating.".format(device.type))
    config_path = './logs/config.json'
    preUNet_path = r'weights/model-kernelnet-4.pth'
    cls_path = r'weights/cls3.pth'
    
    #-------------------------------load--Network---------------------------------------------------#
    preUNet = UNet(in_channels=8, out_channels=1)
    preUNet.to(device)
    assert os.path.exists(preUNet_path), "file: '{}' dose not exist.".format(preUNet_path)
    preUNet.load_state_dict(torch.load(preUNet_path, map_location=device))
    preUNet.eval()
    
    
    clsNet = resnet34(num_classes=3)
    clsNet.to(device)
    assert os.path.exists(cls_path), "file: '{}' dose not exist.".format(cls_path)
    clsNet.load_state_dict(torch.load(cls_path, map_location=device))
    clsNet.eval()
    #-----------------------------------------------------------------------------------#
    
    #----Imgs---------------------------------------------------------------------------#
    subtractor = {}
    for i in range(8):
        subtractor[i] = cv2.createBackgroundSubtractorMOG2()
    imgs = LoadImages()
    kernel = np.array([[3, 3, 3], [3, 1, 3], [3, 3, 3]])
    mask = 0
    mask_9 = np.zeros([8, 750, 1024], dtype=np.uint8)
    
    #---------------load第一帧------------------------------------------------#
    init_cam = imgs[0]
    for i in range(8):
        fgmask = subtractor[i].apply(init_cam[i])
        fgmask = np.ascontiguousarray(fgmask)

        fgmask = ((fgmask - fgmask.min()) * (255.0 / (fgmask.max() - fgmask.min()))).round().astype(np.uint8)
        mask = mask + fgmask.astype(np.float32)
        mask[mask >= 255] = 255
        mask_9[i, :, :] = mask
    #-------后面来新帧frame，就可以丢弃上述代码，上述这个init_cam仅作为程序初始化处理----------#
    
    
    # validate
    evaluate(preUNet=preUNet, clsNet=clsNet, imgs=imgs, device=device, subtractor=subtractor)
        
        
        
        
        
    
    