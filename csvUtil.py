import numpy as np
import pandas as pd
import os


def readcsv(target, fealen=32):
    path = target + 'label.csv'
    label = np.genfromtxt(path, delimiter=',')
    feature = []
    for dirname, dirnames, filenames in os.walk(target):
        for i in range(0, len(filenames) - 1):
            if i == 0:
                file = '/dc.csv'
            else:
                file = '/ac' + str(i) + '.csv'
            path = target + file
            featemp = pd.read_csv(path, header=None).to_numpy()
            feature.append(featemp)
    return np.rollaxis(np.asarray(feature), 0, 3)[:, :, 0:fealen], label


def writecsv(target, data, label, fealen):
    # flatten data
    os.makedirs(target, exist_ok=True)
    data = data.reshape(len(data), fealen, len(data[0, 0, 0, :]) * len(data[0, 0, :, 0]))
    for i in range(0, fealen):
        if i == 0:
            path = os.path.join(target, 'dc.csv')
            # path = target + '/dc.csv'
            np.savetxt(path,
                       data[:, i, :],
                       fmt='%d',
                       delimiter=',',  # column delimiter
                       newline='\n',  # new line character
                       comments='#',  # character to use for comments
                       )
        else:
            # path = target + '/ac' +str(i) + '.csv'
            path = os.path.join(target, f'ac{i}.csv')
            np.savetxt(path,
                       data[:, i, :],
                       fmt='%d',
                       delimiter=',',
                       newline='\n',
                       comments='#')
    # path = target+'/label.csv'
    path = os.path.join(target, 'label.csv')
    np.savetxt(path,
               label,
               fmt='%d',
               delimiter=',',
               newline='\n',
               comments='#')
