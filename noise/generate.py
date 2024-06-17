import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import scipy.io as sio
import os
import numpy as np
import h5py
from tqdm import tqdm
import pdb


def add_noise_to_labels(labels, noise_rate):
    num_samples, num_labels = labels.shape
    num_noise = int(num_samples * noise_rate)
    # 创建一个随机的索引列表，用于选择要添加噪声的样本
    noise_indices = np.random.choice(num_samples, num_noise, replace=False)
    # 随机改变选定样本的标签
    for i in tqdm(noise_indices):
        ones_indices = np.where(labels[i, :] == 1)[0]
        zeros_indices = np.where(labels[i, :] == 0)[0]
        # 随机选择一个值为1的元素，并将其变为0
        if len(ones_indices) > 0:
            j = np.random.choice(ones_indices)
            labels[i, j]=0

        # 随机选择一个值为0的元素，并将其变为1
        if len(zeros_indices) > 0:
            j = np.random.choice(zeros_indices)
            labels[i, j]=1
    return labels

def generate_noise_F(noise):
    noise_rate = noise
    data = h5py.File('./data/MIRFlickr.h5', 'r')
    for i in noise_rate:
        # 添加噪声
        labels_matrix = np.array(list(data['LabTrain']))
        labels_matrix2 = np.array(list(data['LabTrain']))
        noisy_labels_matrix = add_noise_to_labels(labels_matrix, i)

        output_file = h5py.File('./noise/mirflickr25k-lall-noise_{}.h5'.format(i), 'w')

        # # # 将模型输出和损失保存到.h5文件中
        output_file.create_dataset('result', data=noisy_labels_matrix)
        output_file.create_dataset('True', data=labels_matrix2)

        # # # 关闭.h5文件
        output_file.close()

def generate_noise_N(noise):
    noise_rate = noise
    data = h5py.File('./data/NUS-WIDE.h5', 'r')
    for i in noise_rate:
        # 添加噪声
        labels_matrix = np.array(list(data['LabTrain']))
        labels_matrix2 = np.array(list(data['LabTrain']))
        noisy_labels_matrix = add_noise_to_labels(labels_matrix, i)

        output_file = h5py.File('./noise/nus-wide-tc21-lall-noise_{}.h5'.format(i), 'w')

        # 将模型输出和损失保存到.h5文件中
        output_file.create_dataset('result', data=noisy_labels_matrix)
        output_file.create_dataset('True', data=labels_matrix2)

        # 关闭.h5文件
        output_file.close()


def generate_noise_M(noise):
    noise_rate = noise
    data = h5py.File('./data/MS-COCO.h5', 'r')
    for i in noise_rate:
        # 添加噪声
        labels_matrix = np.array(list(data['LabTrain']))
        labels_matrix2 = np.array(list(data['LabTrain']))
        noisy_labels_matrix = add_noise_to_labels(labels_matrix, i)

        output_file = h5py.File('./noise/MSCOCO-lall-noise_{}.h5'.format(i), 'w')

        # 将模型输出和损失保存到.h5文件中
        output_file.create_dataset('result', data=noisy_labels_matrix)
        output_file.create_dataset('True', data=labels_matrix2)

        # 关闭.h5文件
        output_file.close()

def generate_noise_I(noise):
    noise_rate = noise
    data = h5py.File('./data/IAPR.h5', 'r')
    for i in noise_rate:
        # 添加噪声
        labels_matrix = np.array(list(data['LabTrain']))
        labels_matrix2 = np.array(list(data['LabTrain']))
        noisy_labels_matrix = add_noise_to_labels(labels_matrix, i)

        output_file = h5py.File('./noise/IAPR-lall-noise_{}.h5'.format(i), 'w')

        # 将模型输出和损失保存到.h5文件中
        output_file.create_dataset('result', data=noisy_labels_matrix)
        output_file.create_dataset('True', data=labels_matrix2)

        # 关闭.h5文件
        output_file.close()
    
    
if __name__ == "__main__":  
    noise_rate = [0.2,0.5,0.8]
    generate_noise_F(noise_rate)
