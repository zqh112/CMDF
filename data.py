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