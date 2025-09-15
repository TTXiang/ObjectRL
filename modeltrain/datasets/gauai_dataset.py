import numpy as np
from torch.utils.data import Dataset
import os
import torch
import json
from lxml import etree
from pathlib import Path
from tqdm import tqdm
import cv2



def pick_frame(frame_num, file_path):
    images = {}
    for jj in range(8):
        file_name = file_path + '/Cam' + str(jj) + '/' + str(frame_num) + '.jpg'
        # images['Cam' + str(jj)] = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        images['Cam' + str(jj)] = file_name

    # file_name = file_path + '/color' + '/' + str(frame_num) + '.jpg'  # 8
    # # images['color'] = cv2.imread(file_name)
    # images['color'] = file_name
    return images


def load_image(self, video_index, logger=None):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img_eight = self.imgs_eight[video_index]
    aaa = img_eight[0]['Cam0']
    bbb = aaa.split('/Cam0')[0].split('\\')[-1]
    print(bbb)
    logger.info(bbb)
    ratio = self.ratio
    new_size0 = int(self.image_size[0] / ratio)
    new_size1 = int(self.image_size[1] / ratio)
    imgs_re = []
    for frame in img_eight:
        imgs_resized = {}
        for key in frame:
            if key == 'color':
                # print(frame[key])
                pic = cv2.imread(frame[key])
                imgs_resized[key] = cv2.resize(pic, (new_size0, new_size1))
            else:
                pic = cv2.imread(frame[key], cv2.IMREAD_GRAYSCALE)
                imgs_resized[key] = cv2.resize(pic, (new_size0, new_size1))
            # imgs_resized[key] = cv2.resize(frame[key], (new_size0, new_size1))
            # img_size 设置的是预处理后输出的图片尺寸
            # imgs_resized[key] = torch.tensor(imgs_resized[key])
        imgs_re.append(imgs_resized)

    return imgs_re, self.image_size, (new_size0, new_size1)  # img, hw_original, hw_resized



def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y




class kernel_Dataset(Dataset):
    def __init__(self, files_row_root, rank=-1,
                 single_cls=True, ratio=4, image_size=(4096, 3000), logger=None):
        self.ratio = ratio
        self.image_size = image_size
        self.files_row_root = files_row_root
        self.shapes = (4096, 3000)
        self.files_num = len(os.listdir(files_row_root))
        self.label_files = [None] * self.files_num
        self.imgs_eight = [None] * self.files_num
        self.labels = [None] * self.files_num
        self.logger = logger

        video_index = 0
        nm, nf, ne, nd = 0, 0, 0, 0  # number mission, found, empty, duplicate

        for video_file in os.listdir(self.files_row_root):
            # video_files_path: pwd+file
            video_files = os.path.join(self.files_row_root, video_file)

            # imgs_files
            # label_files
            label_single = []
            imgs_single = []
            frame_num = len(os.listdir(os.path.join(video_files, 'color')))
            frame_num = 20

            # img, file
            if frame_num > 2:
                for i in range(frame_num):
                    # label
                    label_file = os.path.join(video_files, 'Cam5', str(i)+'.txt')
                    label_single.append(label_file)
                    # img
                    frame = pick_frame(frame_num=i, file_path=video_files)
                    imgs_single.append(frame)

                # video labels
                # video imgs
                self.label_files[video_index] = label_single
                self.imgs_eight[video_index] = imgs_single

                labels = [np.zeros((0, 5), dtype=np.float32)] * frame_num
                np_labels_path = str(Path(label_single[0]).parent) + ".norect.npy"
                print('Processing video ' + str(video_files))
                pbar = tqdm(label_single)

                # 遍历载入标签文件
                for i, file in enumerate(pbar):
                    # 从文件读取标签信息
                    try:
                        with open(file, "r") as f:
                            # 读取每一行label，并按空格划分数据
                            l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                    except Exception as e:
                        l = np.zeros((0, 5), dtype=np.float32)
                        print("An error occurred while loading the file {}: {}".format(file, e))
                        nm += 1  # file missing
                        continue

                    # 如果标注信息不为空的话
                    if l.shape[0]:
                        # 标签信息每行必须是五个值[class, x, y, w, h]
                        assert l.shape[1] == 5, "> 5 label columns: %s" % file
                        assert (l >= 0).all(), "negative labels: %s" % file
                        assert (l[:, 1:] <= 1).all(), "non-normalized or out of bounds coordinate labels: %s" % file

                        # 检查每一行，看是否有重复信息
                        if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                            nd += 1
                        if single_cls:
                            l[:, 0] = 0  # force dataset into single-class mode

                        labels[i] = l
                        nf += 1  # file found
                    else:
                        ne += 1  # file empty


            else:
                del video_files

            if rank in [-1, 0]:
                # 更新进度条描述信息
                pbar.desc = "Caching labels (%g found, %g missing, %g empty, %g duplicate, for %g images)" % (
                    nf, nm, ne, nd, frame_num)

            # 如果标签信息没有被保存成numpy的格式，且训练样本数大于1000则将标签信息保存成numpy的格式
            if frame_num > 1000:
                print("Saving labels to %s for faster future loading" % np_labels_path)
                np.save(np_labels_path, label_single)  # save for next time

            # video imgs
            self.labels[video_index] = labels

            # video_index
            video_index += 1


        assert nf > 0, "No labels found in %s." % os.path.dirname(self.label_files[0]) + os.sep
        # Extract object detection boxes for a second stage classifier



    def __len__(self):
        return self.files_num

    def __getitem__(self, video_index):
        # load image
        imgs_re, self.image_size, (w, h) = load_image(self, video_index, logger=self.logger)
        self.ratio_p = 1 / self.ratio

        # load labels
        labels_num = []
        x = self.labels[video_index]
        for x_label in x:
            labels = []
            if x_label.size > 0:
                labels = x_label.copy()  # label: class, x, y, w, h
                labels[:, 1] = w * self.ratio * (x_label[:, 1] - x_label[:, 3] / 2)
                labels[:, 2] = h * self.ratio * (x_label[:, 2] - x_label[:, 4] / 2)
                labels[:, 3] = w * self.ratio * (x_label[:, 1] + x_label[:, 3] / 2)
                labels[:, 4] = h * self.ratio * (x_label[:, 2] + x_label[:, 4] / 2)
            nL = len(labels)  # number of labels
            if nL:
                # Normalize coordinates 0-1
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
                labels[:, [2, 4]] /= h * self.ratio  # height
                labels[:, [1, 3]] /= w * self.ratio  # width
            labels_out = torch.zeros((nL, 6))  # nL: number of labels

            if nL:
                labels_out[:, 1:] = torch.from_numpy(labels)
            labels_num.append(labels_out)

        return imgs_re, labels_num, self.image_size, (h, w)

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))





class kernel_infer_Dataset(Dataset):
    def __init__(self, files_row_root, rank=-1,
                 single_cls=True, ratio=4, image_size=(4096, 3000)):
        self.ratio = ratio
        self.image_size = image_size
        self.files_row_root = files_row_root
        self.shapes = (4096, 3000)
        self.files_num = len(os.listdir(files_row_root))
        self.imgs_eight = [None] * self.files_num

        video_index = 0

        for video_file in os.listdir(self.files_row_root):
            # video_files_path: pwd+file
            video_files = os.path.join(self.files_row_root, video_file)

            imgs_single = []
            frame_num = len(os.listdir(os.path.join(video_files, 'color')))

            # img, file
            for i in range(frame_num):
                # img
                frame = pick_frame(frame_num=i, file_path=video_files)
                imgs_single.append(frame)

            # video imgs
            self.imgs_eight[video_index] = imgs_single

            print('Processing video ' + str(video_files))
            # video_index
            video_index += 1


    def __len__(self):
        return self.files_num

    def __getitem__(self, video_index):
        # load image
        imgs_re, self.image_size, (w, h) = load_image(self, video_index, self.logger)
        self.ratio_p = 1 / self.ratio

        return imgs_re, self.image_size, (h, w)

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))



