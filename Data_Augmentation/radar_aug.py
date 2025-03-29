import numpy as np
import os
from tqdm import tqdm

def fft_process(data, axis, shift=True, fft_size=None):
    if shift:
        data = np.fft.fftshift(np.fft.fft(data, n=fft_size, axis=axis), axes=axis)
    else:
        data = np.fft.fft(data, n=fft_size, axis=axis)
    return np.abs(data)

def log_scale_normalization(data):
    return np.log1p(data)

def add_gaussian_noise(data, scale=0.1):
    noise = np.random.normal(0, np.std(data) * scale, data.shape)
    return data + noise

def range_angle_map(data, fft_size=256):
    data = fft_process(data, axis=1, shift=False)
    data -= np.mean(data, axis=2, keepdims=True)
    data = fft_process(data, axis=0, fft_size=fft_size)
    data = np.abs(data).sum(axis=2)
    return data.T

def range_velocity_map(data, fft_size=256):
    data = fft_process(data, axis=1, shift=False)
    data = fft_process(data, axis=2, fft_size=fft_size)
    data = np.abs(data).sum(axis=0)
    return data

def augment_data(data):
    augmented_data = log_scale_normalization(data)
    augmented_data = add_gaussian_noise(augmented_data, scale=0.05)
    return augmented_data

def minmax(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


path_root=["/data/name/code/DeepSense6G_TII-main/Dataset/Adaptation_dataset_multi_modal/scenario31","/data/name/code/DeepSense6G_TII-main/Dataset/Adaptation_dataset_multi_modal/scenario32",
"/data/name/code/DeepSense6G_TII-main/Dataset/Adaptation_dataset_multi_modal/scenario33"]

path_aug_root=["/data/name/code/DeepSense6G_TII-main/Dataset/Adaptation_dataset_multi_modal/scenario31/",
"/data/name/code/DeepSense6G_TII-main/Dataset/Adaptation_dataset_multi_modal/scenario32/",
"/data/name/code/DeepSense6G_TII-main/Dataset/Adaptation_dataset_multi_modal/scenario33/"]

for path_idx in range(len(path_root)):

    path = path_root[path_idx] + "/unit1/radar_data/"
    path_aug_ang = path_aug_root[path_idx] + "/unit1/radar_ang/"
    path_aug_vel = path_aug_root[path_idx] + "/unit1/radar_vel/"

    radarfiles = os.listdir(path)

    if not os.path.isdir(path_aug_ang):
        os.mkdir(path_aug_ang)

    if not os.path.isdir(path_aug_vel):
        os.mkdir(path_aug_vel)


    for filename in tqdm(radarfiles):
        if ".npy" in filename:
            data = np.load(path + filename)
            print('data', data.shape)
            radar_range_ang_data = range_angle_map(data)
            radar_range_vel_data = range_velocity_map(data)

            radar_range_ang_data_aug = []
            for x_idx in range(len(radar_range_ang_data)):
                row_aug = []

                for y_idx in range(len(radar_range_ang_data[x_idx])):
                    angle_range_data_aug = augment_data(radar_range_ang_data[x_idx][y_idx])
                    row_aug.append(angle_range_data_aug)

                radar_range_ang_data_aug.append(row_aug)


            radar_range_vel_data_aug = []
            for x_idx in range(len(radar_range_vel_data)):
                row_aug = []

                for y_idx in range(len(radar_range_vel_data[x_idx])):
                    velocity_range_data_aug = augment_data(radar_range_vel_data[x_idx][y_idx])
                    row_aug.append(velocity_range_data_aug)

                radar_range_vel_data_aug.append(row_aug)

            np.save(path_aug_ang + filename, minmax(np.asarray(radar_range_ang_data_aug)))
            np.save(path_aug_vel + filename, minmax(np.asarray(radar_range_vel_data_aug)))
