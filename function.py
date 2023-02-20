from cv2 import dilate, erode
import SimpleITK as sitk
import os
import numpy as np
from skimage.segmentation import slic
import pandas as pd
import six
from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape, ngtdm, gldm, \
    featureextractor, imageoperations
import yaml
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
from tqdm import tqdm
from joblib import dump, load


def mask_read_mode(mask_array, model, kernel=3):
    # 选择亚分区提取区域
    # ['initial','peritumor','tumor_ring']
    if model == 'initial':
        return mask_array
    elif model == 'peritumor':
        mask_dilate = dilate(mask_array.astype('uint8'), kernel=(kernel, kernel, kernel), iterations=4)
        return mask_dilate
    elif model == 'tumor_ring':
        mask_dilate = dilate(mask_array.astype('uint8'), kernel=(kernel, kernel, kernel), iterations=2)
        mask_erode = erode(mask_array.astype('uint8'), kernel=(kernel, kernel, kernel), iterations=1)
        return mask_dilate - mask_erode


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def readYaml(file):
    with open(file, 'r', encoding='utf-8') as f:
        return yaml.load(f, yaml.Loader)


# SLIC 功能函数

def extract_main(file_path, mask_path, sv_path, out_path, mode, kernel):
    makedirs(out_path)
    paths = []
    paths.append(file_path)
    paths.append(mask_path)
    paths.append(sv_path)
    paths.append(out_path)
    image_list = os.listdir(file_path)
    image_list.sort()
    print('*' * 100)
    print(f'Your dataset consists of {len(image_list)} cases.')
    print('*' * 100)
    mask_list = os.listdir(mask_path)
    mask_list.sort()
    feature_extract(image_list, mask_list, paths, mode, kernel)


def feature_extract(image_list, mask_list, paths, mode, kernel):
    print(f'Supervoxels extraction begin!')

    for i, j in zip(image_list, mask_list):

        image_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(paths[0], i)))
        mask_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(paths[1], j)))
        if np.sum(mask_array) == 0:
            print(f'{i} has no ROI!')
            continue
        mask_array[mask_array != 1] = 0
        mask_array = mask_read_mode(mask_array, mode, kernel=kernel)
        if mode == 'peritumor':
            makedirs(os.path.join(paths[3], 'peritumor'))
            sitk.WriteImage(sitk.GetImageFromArray(mask_array), os.path.join(paths[3], 'peritumor', i))
        elif mode == 'tumor_ring':
            makedirs(os.path.join(paths[3], 'tumor_ring'))
            sitk.WriteImage(sitk.GetImageFromArray(mask_array), os.path.join(paths[3], 'tumor_ring', i))
        parts = int(np.sum(mask_array) / 500)
        if parts > 20:
            parts = 20
        image_array = (image_array - np.mean(image_array)) / np.std(image_array)
        segments = slic(image_array, n_segments=parts, compactness=10, enforce_connectivity=True, mask=mask_array,
                        start_label=1, channel_axis=None)
        for k in range(1, parts + 1):
            blank = np.zeros(image_array.shape)
            blank[segments == k] = 1
            blank = blank * image_array
            if np.sum(blank) == 0:
                continue
            sitk.WriteImage(sitk.GetImageFromArray(blank),
                            os.path.join(paths[2], i.replace('.nii.gz', '_' + str(k).zfill(3) + '.nii.gz')))
        print(i + ' is done!')
    print(f'Supervoxels extraction is done!')
    print('-' * 100)


# load settings

def filter_function(patch, patch_mask, settings):
    results = []
    keys = []
    patch_image = sitk.GetImageFromArray(patch)
    patch_mask_image = sitk.GetImageFromArray(patch_mask)
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('shape')
    # extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('ngtdm')
    extractor.enableFeatureClassByName('gldm')
    extractor.enableFeaturesByName(glcm=settings)
    result = extractor.execute(patch_image, patch_mask_image)
    num = 0
    for (key, val) in six.iteritems(result):
        if num >= 22:
            results.append(float(val))
            keys.append(key)
        num += 1
    parameters = dict([(k, []) for k in keys])
    for i, j in zip(results, parameters.keys()):
        parameters[j].append(i)
    return parameters


def score_dict_cal(img_path, score_dict, yaml_path):
    yaml_settings = readYaml(yaml_path)['featureClass']['glcm']
    img_array_initial = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    all_zeros = np.zeros(img_array_initial.shape)
    all_zeros[img_array_initial != 0] = 1
    score_dict['SV_id'].append(img_path.split('/')[-1].split('.')[0])
    if len(score_dict.keys()) == 1:
        score_dict.update(filter_function(img_array_initial, all_zeros, yaml_settings))
    else:
        for (key, val) in filter_function(img_array_initial, all_zeros, yaml_settings).items():
            score_dict[key].append(val[0])
    return score_dict


def feature_extract_main(sv_path, out_path, yaml_path):
    print(f'Features extraction begin!')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img_list = os.listdir(sv_path)
    img_list.sort()
    score_dict = {'SV_id': []}
    for i in img_list:
        out = score_dict_cal(os.path.join(sv_path, i), score_dict, yaml_path)
        score_dict = out
        print(i + ' is done!')
    df = pd.DataFrame(score_dict, columns=score_dict.keys())
    csv = os.path.join(out_path, 'feature.csv')
    df.to_csv(csv)
    print(f'Features extraction begin and {csv} is saved!')
    print('-' * 100)


def cluster_main(image_path, csv_path, sv_path, concat_path, out_path):
    print(f'Clutering begin!')
    min_max_scaler = preprocessing.MinMaxScaler()
    df_csv = pd.read_csv(os.path.join(csv_path, 'feature.csv'))
    csv = pd.read_csv(os.path.join(csv_path, 'feature.csv'), usecols=df_csv.columns[2:])
    ID = np.array(pd.read_csv(os.path.join(csv_path, 'feature.csv'), usecols=['SV_id']))
    df = np.array(csv).astype(np.float32)
    df[np.isnan(df)] = 0
    df[np.isinf(df)] = 0
    min_max_scaler.fit(df)
    df = min_max_scaler.transform(df)
    bic_list = []
    id_list = []
    for i in range(1, 6):
        bic = GaussianMixture(n_components=i, random_state=66, covariance_type="full", max_iter=1000).fit(df).bic(df)
        id_list.append(i)
        bic_list.append(bic)
    k = id_list[np.where(bic_list == min(bic_list))[0][0]]
    GMM = GaussianMixture(n_components=k, random_state=66, covariance_type="full", max_iter=1000).fit(df)
    dump(GMM, os.path.join(out_path, 'model.joblib'))
    labels = GMM.predict(df)
    labels = np.resize(np.array(labels), (len(labels), 1))
    sv_label = np.concatenate((ID, labels), axis=1)
    image_list = os.listdir(image_path)
    image_list = [i.split('.')[0] for i in image_list]
    image_list.sort()
    for i in image_list:
        image_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(image_path, i + '.nii.gz')))
        concat_mask = np.zeros(image_array.shape)
        for j in sv_label:
            if j[0].startswith(i):
                sv_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(sv_path, j[0] + '.nii.gz')))
                concat_mask[sv_array != 0] = j[1] + 1
        sitk.WriteImage(sitk.GetImageFromArray(concat_mask), os.path.join(concat_path, i + '.nii.gz'))
        print(f'{i} is done!')
    print(f'Clutering done!')
    print('-' * 100)


def cluster_main_predict(GMM, image_path, csv_path, sv_path, concat_path):
    min_max_scaler = preprocessing.MinMaxScaler()
    df_csv = pd.read_csv(os.path.join(csv_path, 'feature.csv'))
    csv = pd.read_csv(os.path.join(csv_path, 'feature.csv'), usecols=df_csv.columns[2:])
    ID = np.array(pd.read_csv(os.path.join(csv_path, 'feature.csv'), usecols=['SV_id']))
    df = np.array(csv).astype(np.float32)
    df[np.isnan(df)] = 0
    df[np.isinf(df)] = 0
    min_max_scaler.fit(df)
    df = min_max_scaler.transform(df)
    labels = GMM.predict(df)
    labels = np.resize(np.array(labels), (len(labels), 1))
    sv_label = np.concatenate((ID, labels), axis=1)
    image_list = os.listdir(image_path)
    image_list = [i.split('.')[0] for i in image_list]
    image_list.sort()
    for i in image_list:

        image_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(image_path, i)))
        concat_mask = np.zeros(image_array.shape)
        for j in sv_label:
            if j[0].startswith(i):
                sv_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(sv_path, j[0] + '.nii.gz')))
                concat_mask[sv_array != 0] = j[1] + 1
        sitk.WriteImage(sitk.GetImageFromArray(concat_mask), os.path.join(concat_path, i + '.nii.gz'))
        print(f'{i} is done!')


def mask_bbox(tumor_mask, cropped_value=0):
    mask_voxel = np.where(tumor_mask != cropped_value)
    minz_idx = int(np.min(mask_voxel[0]))
    maxz_idx = int(np.max(mask_voxel[0])) + 1
    minx_idx = int(np.min(mask_voxel[1]))
    maxx_idx = int(np.max(mask_voxel[1])) + 1
    miny_idx = int(np.min(mask_voxel[2]))
    maxy_idx = int(np.max(mask_voxel[2])) + 1
    return [[minz_idx, maxz_idx], [minx_idx, maxx_idx], [miny_idx, maxy_idx]]


def filter_function_sw(patch, patch_mask, settings):
    results = []
    keys = []
    patch_image = sitk.GetImageFromArray(patch)
    patch_mask_image = sitk.GetImageFromArray(patch_mask)
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('shape')
    # extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('ngtdm')
    extractor.enableFeatureClassByName('gldm')
    extractor.enableFeaturesByName(glcm=settings)
    result = extractor.execute(patch_image, patch_mask_image)
    num = 0
    for (key, val) in six.iteritems(result):
        if num >= 22:
            results.append(float(val))
            keys.append(key)
        num += 1
    parameters = dict([(k, []) for k in keys])
    for i, j in zip(results, parameters.keys()):
        parameters[j].append(i)
    return parameters


def filter_3d(img_path, mask_path, kernel, score_dict, imgpad_path, maskpad_path, setting):
    img_array_initial = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    mask_array_initial = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    padding = int((kernel - 1) / 2)
    img_pad = np.pad(img_array_initial, padding)
    mask_pad = np.pad(mask_array_initial, padding)
    bbox = mask_bbox(mask_pad)
    num = 1
    sitk.WriteImage(sitk.GetImageFromArray(img_pad), os.path.join(imgpad_path, img_path.split('/')[-1]))
    sitk.WriteImage(sitk.GetImageFromArray(mask_pad), os.path.join(maskpad_path, mask_path.split('/')[-1]))
    for i in tqdm(range(bbox[0][0] + padding, bbox[0][1] - padding + 1)):
        for j in range(bbox[1][0] + padding, bbox[1][1] - padding + 1):
            for k in range(bbox[2][0] + padding, bbox[2][1] - padding + 1):
                if mask_pad[i][j][k] == 1:
                    mask_kernel = np.zeros(img_pad.shape)
                    mask_kernel[
                    i - padding:i + padding + 1, j - padding: j + padding + 1, k - padding:k + padding + 1] = 1
                    score_dict['SV_id'].append(img_path.split('/')[-1].split('.')[0] + '_' + str(num).zfill(4))
                    score_dict['bbox'].append(
                        f'{i - padding},{i + padding + 1},{j - padding},{j + padding + 1},{k - padding},{k + padding + 1}')
                    if len(score_dict.keys()) == 2:
                        score_dict.update(filter_function(img_pad, mask_kernel, setting))
                    else:
                        for (key, val) in filter_function(img_pad, mask_kernel, setting).items():
                            score_dict[key].append(val[0])
                    num += 1

    return score_dict


def feature_extract_main_sw(image_path, mask_path, imgpad_path, maskpad_path, csv_path, yaml_path):
    file_list = os.listdir(image_path)
    file_list.sort()
    yaml_settings = readYaml(yaml_path)['featureClass']['glcm']
    score_dict = {'SV_id': [], 'bbox': []}
    for i in file_list[0:1]:
        score_dict = filter_3d(os.path.join(image_path, i),
                               os.path.join(mask_path, i),
                               9, score_dict, imgpad_path, maskpad_path, yaml_settings)
        print(f'{i} is done!')
    df = pd.DataFrame(score_dict, columns=score_dict.keys())
    df.to_csv(os.path.join(csv_path, 'feature.csv'))


def cluster_main_sw(imgpad_path, maskpad_path, csv_path, concat_path, out_path):
    min_max_scaler = preprocessing.MinMaxScaler()
    df_csv = pd.read_csv(os.path.join(csv_path, 'feature.csv'))
    csv = pd.read_csv(os.path.join(csv_path, 'feature.csv'), usecols=df_csv.columns[16:])
    ID = np.array(pd.read_csv(os.path.join(csv_path, 'feature.csv'), usecols=['SV_id', 'bbox']))
    df = np.array(csv).astype(np.float32)
    df[np.isnan(df)] = 0
    df[np.isinf(df)] = 0
    min_max_scaler.fit(df)
    df = min_max_scaler.transform(df)
    bic_list = []
    id_list = []
    for i in range(1, 6):
        bic = GaussianMixture(n_components=i, random_state=66, covariance_type="full", max_iter=1000).bic(df)
        id_list.append(i)
        bic_list.append(bic)
    k = id_list[np.where(bic_list == min(bic_list))]
    GMM = GaussianMixture(n_components=k, random_state=66, covariance_type="full", max_iter=1000).fit(df)
    dump(GMM, os.path.join(out_path, 'model.joblib'))
    labels = GMM.predict(df)
    labels = np.resize(np.array(labels), (len(labels), 1))
    sv_label = np.concatenate((ID, labels), axis=1)
    image_list = os.listdir(imgpad_path)
    image_list = [i.split('.')[0] for i in image_list]
    image_list.sort()
    for i in image_list[:1]:
        image_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(imgpad_path, i)))
        mask_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(maskpad_path, i)))
        concat_mask = np.zeros(image_array.shape)
        print(concat_mask.shape)
        for j in sv_label:
            if j[0].startswith(i):
                bbox = j[1].split(',')
                bbox = [int(i) for i in bbox]
                concat_mask[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]] = j[2] + 1
        concat_mask[mask_array == 0] = 0
        sitk.WriteImage(sitk.GetImageFromArray(concat_mask), os.path.join(concat_path, i + '.nii.gz'))
        print(f'{i} is done!')


def cluster_main_predict_sw(GMM, imgpad_path, maskpad_path, csv_path, concat_path):
    min_max_scaler = preprocessing.MinMaxScaler()
    df_csv = pd.read_csv(os.path.join(csv_path, 'feature.csv'))
    csv = pd.read_csv(os.path.join(csv_path, 'feature.csv'), usecols=df_csv.columns[16:])
    ID = np.array(pd.read_csv(os.path.join(csv_path, 'feature.csv'), usecols=['SV_id', 'bbox']))
    df = np.array(csv).astype(np.float32)
    df[np.isnan(df)] = 0
    df[np.isinf(df)] = 0
    min_max_scaler.fit(df)
    df = min_max_scaler.transform(df)
    labels = GMM.predict(df)
    labels = np.resize(np.array(labels), (len(labels), 1))
    sv_label = np.concatenate((ID, labels), axis=1)
    image_list = os.listdir(imgpad_path)
    image_list = [i.split('.')[0] for i in image_list]
    image_list.sort()
    for i in image_list[:1]:
        image_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(imgpad_path, i)))
        mask_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(maskpad_path, i)))
        concat_mask = np.zeros(image_array.shape)
        print(concat_mask.shape)
        for j in sv_label:
            if j[0].startswith(i):
                bbox = j[1].split(',')
                bbox = [int(i) for i in bbox]
                concat_mask[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]] = j[2] + 1
        concat_mask[mask_array == 0] = 0
        sitk.WriteImage(sitk.GetImageFromArray(concat_mask), os.path.join(concat_path, i + '.nii.gz'))
        print(f'{i} is done!')
