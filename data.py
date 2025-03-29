from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize
from scipy import stats
import cv2
import re
import utm
import os


class CARLA_Data(Dataset):
    def __init__(self, root, root_csv, config, test=False, augment={'camera': 0, 'radar': 0}, flip=False):

        self.dataframe = pd.read_csv(root + root_csv)
        self.root = root
        self.seq_len = config.seq_len
        self.gps_data = []
        self.pos_input_normalized = Normalize_loc(root, self.dataframe, angle_norm=config.angle_norm)
        self.formatted_gps = self.format_cartesian_coordinates(self.pos_input_normalized.reshape(-1, 2))
        self.test = test
        self.add_velocity = config.add_velocity
        self.enhanced = config.enhanced
        self.augment = augment

    def __len__(self):
        """Returns the length of the dataset."""
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        """Returns the item at index idx."""
        data = dict()
        data['fronts'] = []
        data['radars'] = []
        data['gps'] = self.pos_input_normalized[index, :, :]
        data['scenario'] = []
        data['loss_weight'] = []

        PT = []
        file_sep = '/'
        add_fronts = []
        add_radars = []
        instanceidx = ['1', '2', '3', '4', '5']  # 5 time instances
        ## data augmentation
        for stri in instanceidx:
            # camera data
            camera_dir = self.dataframe['unit1_rgb_' + stri][index]
            if self.augment['camera'] > 0:  # and 'scenario31' in camera_dir:
                camera_dir = re.sub('camera_data/', 'camera_data_augmix/', camera_dir)
                camera_dir = camera_dir[:-4] + '.jpg'
                add_fronts.append(camera_dir)
            else:
                add_fronts.append(self.dataframe['unit1_rgb_' + stri][index])

            # radar data
            radar_dir = self.dataframe['unit1_radar_' + stri][index]
            if self.augment['radar'] > 0:
                radar_dir = re.sub('radar_data/', 'radar_ang/', radar_dir)
            else:
                radar_dir = re.sub('radar_data/', 'radar_data_ang/', radar_dir)
            add_radars.append(radar_dir)

        self.seq_len = len(instanceidx)

        # check which scenario is the data sample associated
        scenarios = ['scenario31', 'scenario32', 'scenario33', 'scenario34']
        loss_weights = [1.0, 1.0, 1.0, 1.0]

        for i in range(len(scenarios)):
            s = scenarios[i]
            if s in self.dataframe['unit1_rgb_5'][index]:
                data['scenario'] = s
                data['loss_weight'] = loss_weights[i]
                break

        for i in range(self.seq_len):
            if self.augment['camera'] == 0:
                if 'scenario31' in add_fronts[i] or 'scenario32' in add_fronts[i]:
                    if self.augment['camera'] == 0:  # segmentation added to non augmented data
                        imgs = np.array(Image.open(self.root + add_fronts[i]).resize((256, 256)))
                else:
                    if self.enhanced:
                        imgs = np.array(
                            Image.open(self.root + add_fronts[i]).resize((256, 256)))
                    else:
                        imgs = np.array(
                            Image.open(self.root + add_fronts[i][:30] + '_raw' + add_fronts[i][30:]).resize((256, 256)))
            else:
                imgs = np.array(Image.open(self.root + add_fronts[i]).resize((256, 256)))
            # radar data
            radar_ang1 = np.load(self.root + add_radars[i])
            data['fronts'].append(torch.from_numpy(np.transpose(imgs, (2, 0, 1))))
            radar_ang = np.expand_dims(radar_ang1, 0)

            if self.add_velocity:
                radar_vel1 = np.load(self.root + add_radars[i].replace('ang', 'vel'))
                radar_vel = np.expand_dims(radar_vel1, 0)
                data['radars'].append(torch.from_numpy(np.concatenate([radar_ang, radar_vel], 0)))
            else:
                data['radars'].append(torch.from_numpy(radar_ang))

        if not self.test:
            data['beam'] = []
            data['beamidx'] = []
            beamidx = self.dataframe['unit1_beam'][index] - 1
            x_data = range(max(beamidx - 5, 0), min(beamidx + 5, 63) + 1)
            # Gaussian distributed target instead of one-hot
            y_data = stats.norm.pdf(x_data, beamidx, 0.5)
            data_beam = np.zeros((64))
            data_beam[x_data] = y_data * 1.25
            data['beam'].append(data_beam)
            data['beamidx'].append(beamidx)

        return data

    def format_cartesian_coordinates(self, cartesian_coordinates):
        """
        Formats Cartesian coordinates as descriptive text for CLIP.
        """
        formatted_coordinates = []
        for x, y in cartesian_coordinates:
            formatted_coordinates.append(f"The vehicle is located at X coordinate: {x:.2f}, Y coordinate: {y:.2f}.")
        return formatted_coordinates


def xy_from_latlong(lat_long):
    """
    Requires lat and long, in decimal degrees, in the 1st and 2nd columns.
    Returns same row vec/matrix on cartesian (XY) coords.
    """
    # utm.from_latlon() returns: (EASTING, NORTHING, ZONE_NUMBER, ZONE_LETTER)
    x, y, *_ = utm.from_latlon(lat_long[:, 0], lat_long[:, 1])
    return np.stack((x, y), axis=1)


def Normalize_loc(root, dataframe, angle_norm):
    n_samples = dataframe.index.stop
    pos1_rel_paths = dataframe['unit2_loc_1'].values
    pos2_rel_paths = dataframe['unit2_loc_2'].values
    pos_bs_rel_paths = dataframe['unit1_loc'].values
    pos1_abs_paths = [os.path.join(root, path[2:]) for path in pos1_rel_paths]
    pos2_abs_paths = [os.path.join(root, path[2:]) for path in pos2_rel_paths]
    pos_bs_abs_paths = [os.path.join(root, path[2:]) for path in pos_bs_rel_paths]
    pos_input = np.zeros((n_samples, 2, 2))
    pos_bs = np.zeros((n_samples, 2))
    for sample_idx in tqdm(range(n_samples)):
        # unit2 (UE) positions
        pos_input[sample_idx, 0, :] = np.loadtxt(pos1_abs_paths[sample_idx])
        pos_input[sample_idx, 1, :] = np.loadtxt(pos2_abs_paths[sample_idx])
        # unit1 (BS) position
        pos_bs[sample_idx] = np.loadtxt(pos_bs_abs_paths[sample_idx])
    pos_ue_stacked = np.vstack((pos_input[:, 0, :], pos_input[:, 1, :]))
    pos_bs_stacked = np.vstack((pos_bs, pos_bs))
    pos_ue_stacked = np.vstack((pos_input[:, 0, :], pos_input[:, 1, :]))
    pos_bs_stacked = np.vstack((pos_bs, pos_bs))

    pos_ue_cart = xy_from_latlong(pos_ue_stacked)
    pos_bs_cart = xy_from_latlong(pos_bs_stacked)

    pos_diff = pos_ue_cart - pos_bs_cart

    # pos_min = np.min(pos_diff, axis=0)
    # pos_max = np.max(pos_diff, axis=0)
    pos_max = np.array([40.20955233, 52.31386139])
    pos_min = np.array([-7.18029715, -97.55563452])

    # Normalize and unstack
    pos_stacked_normalized = (pos_diff - pos_min) / (pos_max - pos_min)
    if angle_norm:
        pos_stacked_normalized = normalize(pos_diff, axis=1)

    pos_input_normalized = np.zeros((n_samples, 2, 2))
    pos_input_normalized[:, 0, :] = pos_stacked_normalized[:n_samples]
    pos_input_normalized[:, 1, :] = pos_stacked_normalized[n_samples:]
    if angle_norm:
        angle = np.arctan(pos_input_normalized[..., 1] / pos_input_normalized[..., 0]) / np.pi * 180
        for sample_idx in tqdm(range(n_samples)):
            if 'scenario31' in pos_bs_abs_paths[sample_idx]:
                angle[sample_idx] -= -50.52  # -40.94#
            if 'scenario32' in pos_bs_abs_paths[sample_idx]:
                angle[sample_idx] -= 44.8  # 39.61#
            if 'scenario33' in pos_bs_abs_paths[sample_idx]:
                angle[sample_idx] -= 55.6  # 47.85#
            if 'scenario34' in pos_bs_abs_paths[sample_idx]:
                angle[sample_idx] -= -60  # -59.363#
        idx = angle > 90
        angle[idx] -= 180
        idx = angle < -90
        angle[idx] += 180
        pos_input_normalized[:, 0, 1] = angle[:, 0] / 180 * np.pi
        pos_input_normalized[:, 0, 0] = angle[:, 0] / 180 * np.pi
        pos_input_normalized[:, 1, 1] = angle[:, 1] / 180 * np.pi
        pos_input_normalized[:, 1, 0] = angle[:, 1] / 180 * np.pi
    return pos_input_normalized
