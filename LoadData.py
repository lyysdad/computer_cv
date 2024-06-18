from itertools import islice

import csv
import os

import csvUtil
import numpy as np
import tensorflow as tf
import pandas as pd
from scipy.fftpack import dct
import random

from model import processlabel


class LoadData:
    def __init__(self, fea, lab, preload=False):  # fea是训练数据集路径，lab是标签路径
        self.ptr_n = 0
        self.ptr_h = 0
        self.ptr = 0
        self.dat = fea
        self.label = lab
        with open(lab) as f:
            self.maxlen = sum(1 for _ in f)
        if preload:
            print("loading data into the main memory...")
            self.ft_buffer, self.label_buffer = csvUtil.readcsv(self.dat)

    def nextinstance(self):
        temp_fea = []
        label = None
        idx = random.randint(0, self.maxlen)
        for dirname, dirnames, filenames in os.walk(self.dat):
            for i in range(0, len(filenames) - 1):
                if i == 0:
                    file = '/dc.csv'
                    path = self.dat + file
                    with open(path) as f:
                        r = csv.reader(f)
                        fea = [[int(s) for s in row] for j, row in enumerate(r) if j == idx]
                        temp_fea.append(np.asarray(fea))
                else:
                    file = '/ac' + str(i) + '.csv'
                    path = self.dat + file
                    with open(path) as f:
                        r = csv.reader(f)
                        fea = [[int(s) for s in row] for j, row in enumerate(r) if j == idx]
                        temp_fea.append(np.asarray(fea))
        with open(self.label) as l:
            temp_label = np.asarray(list(l)[idx]).astype(int)
            if temp_label == 0:
                label = [1, 0]
            else:
                label = [0, 1]
        return np.rollaxis(np.array(temp_fea), 0, 3), np.array([label])

    def sgd(self, channel=None, delta1=0, delta2=0):
        with open(self.label) as l:
            labelist = np.asarray(list(l)).astype(int)
        length = labelist.size
        idx = random.randint(0, length - 1)
        temp_label = labelist[idx]
        if temp_label == 0:
            label = [1, 0]
        else:
            label = [0, 1]
        ft = self.ft_buffer[idx]

        return ft, np.array(label)

    def sgd_batch_2(self, batch, channel=None, delta1=0, delta2=0):
        with open(self.label) as l:
            labelist = np.asarray(list(l)).astype(int)
            labexn = np.where(labelist == 0)[0]
            labexh = np.where(labelist == 1)[0]
        n_length = labexn.size
        h_length = labexh.size
        if not batch % 2 == 0:
            print('ERROR:Batch size must be even')
            print('Abort.')
            quit()
        else:
            num = batch // 2
        idxn = labexn[(np.random.rand(num) * n_length).astype(int)]
        idxh = labexh[(np.random.rand(num) * h_length).astype(int)]
        label = np.concatenate((np.zeros(num), np.ones(num)))
        label = processlabel(label, 2, 0, 0)
        ft_batch = np.concatenate((self.ft_buffer[idxn], self.ft_buffer[idxh]))
        ft_batch_nhs = self.ft_buffer[idxn]
        label_nhs = np.zeros(num)
        return ft_batch, label

    def sgd_batch(self, batch, channel=None, delta1=0, delta2=0):
        with open(self.label) as l:
            labelist = np.asarray(list(l)).astype(int)
            labexn = np.where(labelist == 0)[0]
            labexh = np.where(labelist == 1)[0]
        n_length = labexn.size
        h_length = labexh.size
        if not batch % 2 == 0:
            print('ERROR:Batch size must be even')
            print('Abort.')
            quit()
        else:
            num = batch // 2
        idxn = labexn[(np.random.rand(num) * n_length).astype(int)]
        idxh = labexh[(np.random.rand(num) * h_length).astype(int)]
        label = np.concatenate((np.zeros(num), np.ones(num)))
        # label = processlabel(label,2, delta1, delta2)
        ft_batch = np.concatenate((self.ft_buffer[idxn], self.ft_buffer[idxh]))
        ft_batch_nhs = self.ft_buffer[idxn]
        label_nhs = np.zeros(num)
        return ft_batch, label, ft_batch_nhs, label_nhs

    '''
    nextbatch_beta: returns the balalced batch, used for training only
    '''

    def nextbatch_beta(self, batch, channel=None, delta1=0, delta2=0):
        def update_ptr(ptr, batch, length):  # 更新指针位置，指针向右偏移一个batch单位
            if ptr + batch < length:
                ptr += batch
            if ptr + batch >= length:
                ptr = ptr + batch - length
            return ptr

        with open(self.label) as l:  # 每次调用这个方法都要读取一遍label.csv
            labelist = np.asarray(list(l)).astype(int)  # 将标签元素转为整型
            labexn = np.where(labelist == 0)[0]  # 获取标签列表里为0的元素的下标值，生成列表赋值给labexn(非热点的位置)
            labexh = np.where(labelist == 1)[0]  # 获取标签列表里为1的元素的下标值，生成列表赋值给labexh(热点位置)
        n_length = labexn.size  # 非热点列表长度
        h_length = labexh.size  # 热点列表长度
        # print("the batch is: ",batch)
        if not batch % 2 == 0:
            print('ERROR:Batch size must be even')
            print('Abort.')
            quit()
        else:
            num = batch // 2  # 确保每次训练，热点图片与非热点图片数量一致，即一半热点，一半非热点，应对数据不平衡问题（过采样）
            if num >= n_length or num >= h_length:
                print('ERROR:Batch size exceeds data size')
                print('Abort.')
                quit()
            else:
                if self.ptr_n + num < n_length:  # 非热点指针位置加上偏移量16
                    idxn = labexn[self.ptr_n:self.ptr_n + num]
                elif self.ptr_n + num >= n_length:
                    idxn = np.concatenate((labexn[self.ptr_n:n_length], labexn[0:self.ptr_n + num - n_length]))
                self.ptr_n = update_ptr(self.ptr_n, num, n_length)

                if self.ptr_h + num < h_length:
                    idxh = labexh[self.ptr_h:self.ptr_h + num]
                elif self.ptr_h + num >= h_length:
                    idxh = np.concatenate((labexh[self.ptr_h:h_length], labexh[0:self.ptr_h + num - h_length]))  # 如果热点指针加上偏移量超出了热点数组长度，从热点数组下标为0开始将数据补入
                self.ptr_h = update_ptr(self.ptr_h, num, h_length)
                # print self.ptr_n, self.ptr_h

                label = np.concatenate((np.zeros(num), np.ones(num)))
                # label = processlabel(label,2, delta1, delta2)
                ft_batch = np.concatenate((self.ft_buffer[idxn], self.ft_buffer[idxh])) # 热点，非热点各取16个拼接成一个批处理group，group.length = 32
                ft_batch_nhs = self.ft_buffer[idxn]
                label_nhs = np.zeros(num)
        return ft_batch, label, ft_batch_nhs, label_nhs

    '''
    nextbatch_without_balance: returns the normal batch. Suggest to use for training and validation
    '''

    def nextbatch_without_balance_alpha(self, batch, channel=None, delta1=0, delta2=0):
        def update_ptr(ptr, batch, length):
            if ptr + batch < length:
                ptr += batch
            if ptr + batch >= length:
                ptr = ptr + batch - length
            return ptr

        if self.ptr + batch < self.maxlen:
            label = self.label_buffer[self.ptr:self.ptr + batch]
            ft_batch = self.ft_buffer[self.ptr:self.ptr + batch]
        else:
            label = np.concatenate(
                (self.label_buffer[self.ptr:self.maxlen], self.label_buffer[0:self.ptr + batch - self.maxlen]))
            ft_batch = np.concatenate(
                (self.ft_buffer[self.ptr:self.maxlen], self.ft_buffer[0:self.ptr + batch - self.maxlen]))
        self.ptr = update_ptr(self.ptr, batch, self.maxlen)
        return ft_batch, label

    def nextbatch(self, batch, channel=None, delta1=0, delta2=0):
        # print('recommed to use nextbatch_beta() instead')
        databat = None
        temp_fea = []
        label = None
        if batch > self.maxlen:
            print('ERROR:Batch size exceeds data size')
            print('Abort.')
            quit()
        if self.ptr + batch < self.maxlen:
            # processing labels
            with open(self.label) as l:
                temp_label = np.asarray(list(l)[self.ptr:self.ptr + batch])
                label = processlabel(temp_label, 2, delta1, delta2)
            for dirname, dirnames, filenames in os.walk(self.dat):
                for i in range(0, len(filenames) - 1):
                    if i == 0:
                        file = '/dc.csv'
                        path = self.dat + file
                        with open(path) as f:
                            temp_fea.append(np.genfromtxt(islice(f, self.ptr, self.ptr + batch), delimiter=','))
                    else:
                        file = '/ac' + str(i) + '.csv'
                        path = self.dat + file
                        with open(path) as f:
                            temp_fea.append(np.genfromtxt(islice(f, self.ptr, self.ptr + batch), delimiter=','))
            self.ptr = self.ptr + batch
        elif (self.ptr + batch) >= self.maxlen:

            # processing labels
            with open(self.label) as l:
                a = np.genfromtxt(islice(l, self.ptr, self.maxlen), delimiter=',')
            # with open(self.label) as l:
            #     b = np.genfromtxt(islice(l, 0, self.ptr + batch - self.maxlen), delimiter=',')
            # processing data
            if self.ptr == self.maxlen - 1 or self.ptr == self.maxlen:
                temp_label = b
            elif self.ptr + batch - self.maxlen == 1 or self.ptr + batch - self.maxlen == 0:
                temp_label = a
            else:
                temp_label = np.concatenate((a, b))
            label = processlabel(temp_label, 2, delta1, delta2)
            # print label.shape
            for dirname, dirnames, filenames in os.walk(self.dat):
                for i in range(0, len(filenames) - 1):
                    if i == 0:
                        file = '/dc.csv'
                        path = self.dat + file
                        with open(path) as f:
                            a = np.genfromtxt(islice(f, self.ptr, self.maxlen), delimiter=',')
                        # with open(path) as f:
                        #     b = np.genfromtxt(islice(f, None, self.ptr + batch - self.maxlen), delimiter=',')
                        if self.ptr == self.maxlen - 1 or self.ptr == self.maxlen:
                            temp_fea.append(b)
                        elif self.ptr + batch - self.maxlen == 1 or self.ptr + batch - self.maxlen == 0:
                            temp_fea.append(a)
                        else:
                            try:
                                temp_fea.append(np.concatenate((a, b)))
                            except:
                                print(a.shape, b.shape, self.ptr)
                    else:
                        file = '/ac' + str(i) + '.csv'
                        path = self.dat + file
                        with open(path) as f:
                            a = np.genfromtxt(islice(f, self.ptr, self.maxlen), delimiter=',')
                        # with open(path) as f:
                        #     b = np.genfromtxt(islice(f, 0, self.ptr + batch - self.maxlen), delimiter=',')
                        if self.ptr == self.maxlen - 1 or self.ptr == self.maxlen:
                            temp_fea.append(b)
                        elif self.ptr + batch - self.maxlen == 1 or self.ptr + batch - self.maxlen == 0:
                            temp_fea.append(a)
                        else:
                            try:
                                temp_fea.append(np.concatenate((a, b)))
                            except:
                                print(a.shape, b.shape, self.ptr)
            self.ptr = self.ptr + batch - self.maxlen
        # print np.asarray(temp_fea).shape
        return np.rollaxis(np.asarray(temp_fea), 0, 3)[:, :, 0:channel], label