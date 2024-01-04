from glob import glob
import json
from tools.data_sampling import preprocess_disease, divided_algorithm, print_statistics, copy_files_with_parents
import math
import pandas as pd
from collections import deque
import os
from tqdm import tqdm
from collections import Counter


if __name__ == '__main__':
    # Directory setting
    DATA_ROOT = '/workspace/data/keypoints/ori/1.Datasets'
    DES_ROOT = '/workspace/data/keypoints/subset'
    clinical_Info_paths = glob(f'{DATA_ROOT}/1.원천데이터/**/CLINICALINFO/*.json', recursive=True)
    data_ratios = [.8, .1, .1]

    infos = []
    for path in clinical_Info_paths:
        with open(path, 'r') as fp:
            db = json.load(fp)
        # assert len(db['Disease']) == 1
        disease = db['Disease'][0]
        disease_detail = db['DiseaseDetail'][0] if len(db['DiseaseDetail']) == 1 else ''
        disease_detail = preprocess_disease(disease_detail)
        _info = {
            'case_id': db['Case_ID'],
            'disease': disease,
            'sex': db['Sex'],
            'Age': db['Age'],
            'disease_detail': disease_detail,
            'generation': db['Age'] // 10
        }
        infos.append(_info)

    df = pd.DataFrame.from_records(infos)
    print('The number of entire patients: ', len(df.index))
    print('Sampling ratios: ', data_ratios)
    print(len(df.index))
    df_for_divided = df.copy()
    df_subset = {}
    std_ratio = 1.
    for i, (subset, ratio) in enumerate(zip(['train', 'val', 'test'], data_ratios)):
        # Data
        N_p = len(df_for_divided.index)
        if i == 0:
            N = math.ceil(N_p * ratio)
        else:
            N = math.ceil(N_p * ratio / std_ratio)

        df_divided, _ = divided_algorithm(df_for_divided, deque(['disease_detail', 'generation', 'sex']), N)
        # calculate left dataframe
        for _, row in df_divided.iterrows():
            df_for_divided.drop(index=df_for_divided[df_for_divided['case_id'] == row['case_id']].index, inplace=True)
        print(f'{subset}: {len(df_divided.index)}')
        df_subset[subset] = df_divided
        std_ratio -= ratio

    for subset, df_subset in df_subset.items():
        desc_path = f'{DES_ROOT}/{subset}/1.Datasets'
        for idx, row in tqdm(df_subset.iterrows(), total=df_subset.shape[0]):
            _info = {'case_id': row["case_id"]}
            # keypoint
            keypoint_label_paths = [x for x in glob(f'{DATA_ROOT}/2.라벨링데이터/**/{row["case_id"]}/**/ANGLE/*.json',
                                                    recursive=True)]
            # clinical info
            if len(keypoint_label_paths) < 1:
                continue
            clinical_info_paths = glob(f'{DATA_ROOT}/1.원천데이터/**/{row["case_id"]}/CLINICALINFO/*.json', recursive=True)
            for info_path in clinical_info_paths:
                copy_files_with_parents(info_path, os.path.dirname(info_path.replace(DATA_ROOT, '')), desc_path)
            for lpath in keypoint_label_paths:
                # copy images
                for i, tag in enumerate(['SAG', 'COR', 'AXL']):
                    with open(lpath, 'r') as fp:
                        db = json.load(fp)['ArrayOfannotation_info'][0]
                    kpt_3ds = []
                    for anno in db['anglePoint_list']:
                        kpt_3d = [anno['pt_pos']['X'], anno['pt_pos']['Y'], anno['pt_pos']['Z']]
                        cat = {
                            'keypoint_3d': kpt_3d,
                        }
                        kpt_3ds.append(kpt_3d)

                    # std_kpt_x = str(Counter([x[0] for x in kpt_3ds]).most_common(1)[0][0])
                    std_kpt_x = str(Counter([x[i] for x in kpt_3ds]).most_common(1)[0][0])
                    img_dir = os.path.abspath(f'{lpath}/..') \
                        .replace('2.라벨링데이터', '1.원천데이터') \
                        .replace('/ANGLE', f'/{tag}')
                    img_paths = glob(f'{img_dir}/*{std_kpt_x.zfill(4)}.dcm', recursive=True)
                    for img_path in img_paths:
                        if os.path.exists(img_path):
                            copy_files_with_parents(img_path, os.path.dirname(img_path.replace(img_dir, '')),
                                                    img_dir.replace(DATA_ROOT, desc_path))
                        else:
                            print(img_path)

                    # for kpt in kpt_3ds:
                    #     std_kpt_x = str(kpt[i])
                    #     img_dir = os.path.abspath(f'{lpath}/..')\
                    #         .replace('2.라벨링데이터', '1.원천데이터')\
                    #         .replace('/ANGLE',f'/{tag}')
                    #     img_paths = glob(f'{img_dir}/*{std_kpt_x.zfill(4)}.dcm', recursive=True)
                    #     for img_path in img_paths:
                    #         if os.path.exists(img_path):
                    #             copy_files_with_parents(img_path, os.path.dirname(img_path.replace(img_dir, '')),
                    #                                     img_dir.replace(DATA_ROOT, desc_path))
                    #         else:
                    #             print(img_path)
                # copy label
                copy_files_with_parents(lpath, os.path.dirname(lpath.replace(DATA_ROOT, '')), desc_path)
        print('End split')
