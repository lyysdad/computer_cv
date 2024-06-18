from scipy.fftpack import dct
import numpy as np


def rescale(img):
    return (img / 255)


def cutblock(img, block_size, block_dim):
    blockarray = []
    for i in range(0, block_dim):
        blockarray.append([])
        for j in range(0, block_dim):
            blockarray[i].append(img[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size])

    return np.asarray(blockarray)


def subfeature(imgraw, fealen):
    if fealen > len(imgraw) * len(imgraw[:, 0]):
        print('ERROR: Feature vector length exceeds block size.')
        print('Abort.')
        quit()

    img = dct2(imgraw)
    size = fealen
    idx = 0
    scaled = img
    feature = np.zeros(fealen, dtype=int)
    for i in range(0, size):
        if idx >= size:
            break
        elif i == 0:
            feature[0] = scaled[0, 0]
            idx = idx + 1
        elif i % 2 == 1:
            for j in range(0, i + 1):
                if idx < size:
                    feature[idx] = scaled[j, i - j]
                    idx = idx + 1
                else:
                    break
        elif i % 2 == 0:
            for j in range(0, i + 1):
                if idx < size:
                    feature[idx] = scaled[i - j, j]
                    idx = idx + 1
                else:
                    break

    return feature


def dct2(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')
