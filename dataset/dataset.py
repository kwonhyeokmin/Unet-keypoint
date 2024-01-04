import os.path
from skimage.draw import polygon2mask
import cv2
import numpy as np
import copy
import torch
from common.utils.etc_utils import rle_decode
from torch.utils.data import Dataset
import pydicom
import random
random.seed(0)


class CTDataset(Dataset):
    def __init__(self, db, transforms=None):
        _data = copy.deepcopy(db.data)
        random.shuffle(_data)
        self.db = _data
        print(f'The number of data: {len(self.db)}')

        self.cat_name = db.cat_name
        self.transforms = transforms
        self.rle = db.rle
        self.color_space = 'HSV'

    def load_img(self, path, colorspace, extension='jpg'):
        if extension == 'jpg':
            img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        elif extension == 'dcm':
            dcm = pydicom.dcmread(path)
            dcm_img = dcm.pixel_array
            dcm_img = dcm_img.astype(float)
            # dcm_img = dcm_img * dcm.RescaleSlope + dcm.RescaleIntercept
            # Rescaling grey scale between 0-255
            dcm_img_scaled = (np.maximum(dcm_img, 0) / dcm_img.max()) * 255
            # Convert to uint
            dcm_img_scaled = np.uint8(dcm_img_scaled)

            img_bgr = cv2.cvtColor(dcm_img_scaled, cv2.COLOR_GRAY2BGR)
        if colorspace == 'HSV':
            input_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        else:
            input_img = img_bgr

        img = input_img.astype('float32')  # original is uint16
        mx = np.max(img)
        if mx:
            img /= mx  # scale image to [0, 1]
        return img, img_bgr

    def extract_mask(self, annos, height, width):
        masks = np.zeros((len(self.cat_name), height, width))
        for anno in annos:
            cat_n = anno['category_id']
            pz, py = anno['keypoint_2d']
            # masks[cat_n, py, pz] = 1

            gauss_kernel = self.gaussian_2d(shape=(5, 5), sigma=0.3)

            ey, ex = [py - 5 // 2, pz - 5 // 2]
            v_range1 = slice(max(0, ey), max(min(ey + gauss_kernel.shape[0], height), 0))
            h_range1 = slice(max(0, ex), max(min(ex + gauss_kernel.shape[1], width), 0))
            v_range2 = slice(max(0, -ey), min(-ey + height, gauss_kernel.shape[0]))
            h_range2 = slice(max(0, -ex), min(-ex + width, gauss_kernel.shape[1]))
            try:
                masks[cat_n, v_range1, h_range1] += gauss_kernel[v_range2, h_range2]
            except:
                print("error")
        # plt.imshow(mask)
        # plt.show()
        # mask = mask[..., np.newaxis]
        return np.transpose(masks, (1, 2, 0))

    @staticmethod
    def gaussian_2d(shape=(3, 3), sigma=0.5):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        data = copy.deepcopy(self.db[index])
        extension = data['image_path'].split('.')[-1]
        img, img_bgr = self.load_img(data['image_path'], colorspace=self.color_space, extension=extension)
        data_id = os.path.basename(data['image_path']).replace('.dcm', '')
        mask = self.extract_mask(data['anno'], data['height'], data['width'])
        if self.transforms:
            transformed_data = self.transforms(image=img, mask=mask)
            transformed_ori_data = self.transforms(image=img_bgr, mask=mask)
            img = transformed_data['image']
            mask = transformed_data['mask']
            img_bgr = transformed_ori_data['image']
        img = np.transpose(img, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        area = np.sum(np.where(img_bgr > 64, 1, 0)) / 32

        return torch.tensor(img), torch.tensor(mask), torch.tensor(img_bgr), area, data_id
