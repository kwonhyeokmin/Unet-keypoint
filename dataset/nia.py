import copy
import os
import os.path as osp
import json
import constants
from glob import glob
import cv2
import pydicom
from collections import Counter
import numpy as np


class NIADataset:
    def __init__(self, data_split, tag='SAG'):
        self.ann_path = []
        self.ann_path += [x.replace(os.sep, '/') for x in
                         glob(f'{constants.DATASET_FOLDER}/{data_split}/1.Datasets/2.라벨링데이터/**/ANGLE/*.json', recursive=True)]
        self.cat_name = [
            'TMA_Line1_Pt1',
            'TMA_Line1_Pt2',
            'TMA_Line2_Pt1',
            'TMA_Line2_Pt2'
        ]
        self.tag_nm = ['SAG', 'AXL', 'COR']
        self.tag = tag

        self.rle = False
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def load_data(self):
        data = []
        cls_n = {}
        for c in self.cat_name:
            cls_n.update({c: 0})
        data_id = 0
        for i, ann_path in enumerate(self.ann_path):
            with open(ann_path) as fp:
                db = json.load(fp)['ArrayOfannotation_info'][0]['anglePoint_list']
            path_split = ann_path.split('/')
            dir_name, file_name = '/'.join(path_split[0:-1]), path_split[-1]
            dir_name = dir_name.replace('2.라벨링데이터', '1.원천데이터').replace('ANGLE', f'{self.tag}')
            m_image_path = file_name.replace('_Angle', '').replace('.json', '')
            kpt_3ds = []
            _ann = {
                'id': data_id,
                'anno': []
            }
            for anno in db:
                try:
                    kpt_3d = [anno['pt_pos']['X'], anno['pt_pos']['Y'], anno['pt_pos']['Z']]
                    kpt_2d = [x for i,x in enumerate(kpt_3d) if i != self.tag_nm.index(self.tag)]
                    cat = {
                        'category_id': self.cat_name.index(anno['pt_name']),
                        'keypoint_3d': kpt_3d,
                        'keypoint_2d': kpt_2d
                    }
                    kpt_3ds.append(kpt_3d)
                    cls_n[anno['pt_name']] += 1
                    _ann['anno'].append(cat)
                except KeyError as e:
                    print(f'Annotation file has invalid Key: {ann_path}')
                    continue
                except ValueError as e:
                    # print(e)
                    # print(ann_path)
                    continue
            std_kpt_x = str(Counter([x[self.tag_nm.index(self.tag)] for x in kpt_3ds]).most_common(1)[0][0])
            # for kpt in kpt_3ds:
            # std_kpt_x = str(kpt[self.tag_nm.index(self.tag)])
            image_path = f'{dir_name}/{m_image_path}_{self.tag}_{std_kpt_x.zfill(4)}.dcm'
            if not os.path.exists(image_path):
                continue
            dcm = pydicom.dcmread(image_path)
            dcm_img = dcm.pixel_array
            dcm_img = dcm_img.astype(float)
            # dcm_img = dcm_img * dcm.RescaleSlope + dcm.RescaleIntercept
            # Rescaling grey scale between 0-255
            dcm_img_scaled = (np.maximum(dcm_img, 0) / dcm_img.max()) * 255
            # Convert to uint
            dcm_img_scaled = np.uint8(dcm_img_scaled)
            img_bgr = cv2.cvtColor(dcm_img_scaled, cv2.COLOR_GRAY2BGR)
            h, w, _ = img_bgr.shape

            _ann['image_path'] = image_path
            _ann['height'] = h
            _ann['width'] = w

            if len(_ann['anno']) > 0:
                data.append(_ann)
                data_id += 1
                # vis_img = copy.deepcopy(img_bgr)
                # for item in _ann['anno']:
                #     x, y, z = item['keypoint_3d']
                #     print('item: ', item['keypoint_3d'])
                #     vis_img = cv2.circle(vis_img, (y, z), 5, (0, 0, 255), -1)
                #     vis_img = cv2.putText(vis_img, self.cat_name[item['category_id']], (y-2, z), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
                # vis_path = f'/workspace/data/vis/{os.path.basename(image_path).replace(".dcm", ".jpg")}'
                # cv2.imwrite(vis_path, vis_img)
                # print(f'Visualize in path {vis_path}')
        print(f'End loading data. The number of data: {len(data)}')
        print('**************************************************')
        for k, v in cls_n.items():
            print(f'{k}: {v}')
        print('**************************************************')
        return data


if __name__ == '__main__':
    train_dataset = NIADataset(data_split='train')
    val_dataset = NIADataset(data_split='val')
    test_dataset = NIADataset(data_split='test')