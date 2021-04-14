import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
from ofdm import *
from config import *


def train_generator(Htrain):
    temp = np.random.randint(0, len(Htrain))
    HH = Htrain[temp]
    bits0 = np.random.binomial(n=1, p=0.5, size=(K * mu,))
    bits1 = np.random.binomial(n=1, p=0.5, size=(K * mu,))
    X = [bits0, bits1]
    signal_output00, signal_output01, signal_output10, signal_output11,YY = MIMO(X, HH, SNRdb, mode, P)

    # pilotValue, pilotCarriers = pilot(P)
    # pilotCarriers1 = pilotCarriers[0:2 * P:2]
    # pilotCarriers2 = pilotCarriers[1:2 * P:2]
    #
    # P_0 = np.zeros(K, dtype=complex)
    # P_0[pilotCarriers1] = pilotValue[0:2 * P:2]
    # P_0 = np.concatenate((np.real(P_0), np.imag(P_0)))
    # P_1 = np.zeros(K, dtype=complex)
    # P_1[pilotCarriers2] = pilotValue[1:2 * P:2]
    # P_1 = np.concatenate((np.real(P_1), np.imag(P_1)))
    # XP = np.concatenate((P_0, P_1))
    YP00 = np.zeros(2*K)
    YP01 = np.zeros(2 * K)
    YP10 = np.zeros(2 * K)
    YP11 = np.zeros(2 * K)

    for i in range (K):
      YP00[2*i] = signal_output00[i]
      YP00[2*i+1] = signal_output00[i+K]
      YP01[2*i] = signal_output01[i]
      YP01[2*i+1] = signal_output01[i+K]
      YP10[2*i] = signal_output10[i]
      YP10[2*i + 1] = signal_output10[i+K]
      YP11[2*i] = signal_output11[i]
      YP11[2*i + 1] = signal_output11[i+K]


    # YP1 = YY[8 * np.arange(K) + 4] + YY[8 * np.arange(K) + 5] * 1j
    # YP0 = np.concatenate((np.real(YP0), np.imag(YP0)))
    # YP1 = np.concatenate((np.real(YP1), np.imag(YP1)))
    YP11 = np.reshape(YP11, [1, 1, 512])
    YP00 = np.reshape(YP00, [1, 1, 512])
    YP01 = np.reshape(YP01, [1, 1, 512])
    YP10 = np.reshape(YP10, [1, 1, 512])

    YD0 = YY[8 * np.arange(K) + 2] + YY[8 * np.arange(K) + 3] * 1j  # (K,)complex
    YD1 = YY[8 * np.arange(K) + 6] + YY[8 * np.arange(K) + 7] * 1j
    yd0 = np.fft.ifft(YD0)
    yd1 = np.fft.ifft(YD1)
    yd = np.concatenate([np.real(yd0), np.imag(yd0), np.real(yd1), np.imag(yd1)])
    YD = np.concatenate([np.real(YD0), np.imag(YD0), np.real(YD1), np.imag(YD1)])

    bit0_labels = X[0].reshape(-1, mu).transpose().reshape([-1])  # (实K，虚K)  (2K,)
    bit1_labels = X[1].reshape(-1, mu).transpose().reshape([-1])
    bit_labels = np.concatenate([bit0_labels, bit1_labels])  # (4K,)

    H_label = np.fft.fft(HH, K)
    label00 = np.zeros(2 * K)
    label01 = np.zeros(2 * K)
    label10 = np.zeros(2 * K)
    label11 = np.zeros(2 * K)

    for i in range(K):
        label00[2 * i] = np.real(H_label[0, i])
        label00[2 * i + 1] = np.imag(H_label[0, i])
        label01[2 * i] = np.real(H_label[1, i])
        label01[2 * i + 1] = np.imag(H_label[1, i])
        label10[2 * i] = np.real(H_label[2, i])
        label10[2 * i + 1] = np.imag(H_label[2, i])
        label11[2 * i] = np.real(H_label[3, i])
        label11[2 * i + 1] = np.imag(H_label[3, i])


    return YP00, YP01, YP10, YP11, label00, label01, label10, label11, YD, bit_labels


def test_generator(Htest):
    temp = np.random.randint(0, len(Htest))
    HH = Htest[temp]
    bits0 = np.random.binomial(n=1, p=0.5, size=(K * mu,))
    bits1 = np.random.binomial(n=1, p=0.5, size=(K * mu,))
    X = [bits0, bits1]
    signal_output00, signal_output01, signal_output10, signal_output11, YY = MIMO(X, HH, SNRdb, mode, P)

    # pilotValue, pilotCarriers = pilot(P)
    # pilotCarriers1 = pilotCarriers[0:2 * P:2]
    # pilotCarriers2 = pilotCarriers[1:2 * P:2]
    #
    # P_0 = np.zeros(K, dtype=complex)
    # P_0[pilotCarriers1] = pilotValue[0:2 * P:2]
    # P_0 = np.concatenate((np.real(P_0), np.imag(P_0)))
    # P_1 = np.zeros(K, dtype=complex)
    # P_1[pilotCarriers2] = pilotValue[1:2 * P:2]
    # P_1 = np.concatenate((np.real(P_1), np.imag(P_1)))
    # XP = np.concatenate((P_0, P_1))

    YP00 = np.zeros(2*K)
    YP01 = np.zeros(2 * K)
    YP10 = np.zeros(2 * K)
    YP11 = np.zeros(2 * K)

    for i in range (K):
      YP00[2*i] = signal_output00[i]
      YP00[2*i+1] = signal_output00[i+K]
      YP01[2*i] = signal_output01[i]
      YP01[2*i+1] = signal_output01[i+K]
      YP10[2*i] = signal_output10[i]
      YP10[2*i + 1] = signal_output10[i+K]
      YP11[2*i] = signal_output11[i]
      YP11[2*i + 1] = signal_output11[i+K]

    # YP1 = YY[8 * np.arange(K) + 4] + YY[8 * np.arange(K) + 5] * 1j
    # YP0 = np.concatenate((np.real(YP0), np.imag(YP0)))
    # YP1 = np.concatenate((np.real(YP1), np.imag(YP1)))
    YP11 = np.reshape(YP11, [1, 1, 512])
    YP00 = np.reshape(YP00, [1, 1, 512])
    YP01 = np.reshape(YP01, [1, 1, 512])
    YP10 = np.reshape(YP10, [1, 1, 512])

    YD0 = YY[8 * np.arange(K) + 2] + YY[8 * np.arange(K) + 3] * 1j  # (K,)complex
    YD1 = YY[8 * np.arange(K) + 6] + YY[8 * np.arange(K) + 7] * 1j
    yd0 = np.fft.ifft(YD0)
    yd1 = np.fft.ifft(YD1)
    yd = np.concatenate([np.real(yd0), np.imag(yd0), np.real(yd1), np.imag(yd1)])
    YD = np.concatenate([np.real(YD0), np.imag(YD0), np.real(YD1), np.imag(YD1)])

    bit0_labels = X[0].reshape(-1, mu).transpose().reshape([-1])  # (实K，虚K)  (2K,)
    bit1_labels = X[1].reshape(-1, mu).transpose().reshape([-1])
    bit_labels = np.concatenate([bit0_labels, bit1_labels])  # (4K,)

    H_label = np.fft.fft(HH, K)
    label00 = np.zeros(2*K)
    label01 = np.zeros(2 * K)
    label10 = np.zeros(2 * K)
    label11 = np.zeros(2 * K)

    for i in range (K):
      label00[2*i] = np.real(H_label[0,i])
      label00[2*i+1] = np.imag(H_label[0,i])
      label01[2*i] = np.real(H_label[1,i])
      label01[2*i+1] = np.imag(H_label[1,i])
      label10[2*i] = np.real(H_label[2,i])
      label10[2*i+1] = np.imag(H_label[2,i])
      label11[2*i] = np.real(H_label[3,i])
      label11[2*i+1] = np.imag(H_label[3,i])
    return YP00, YP01, YP10, YP11, label00, label01, label10, label11, YD, bit_labels


class train_Dataset(Dataset):

    def __init__(self, Htrain):
        self.H = Htrain

    def __getitem__(self, index):
        YP00, YP01, YP10, YP11, label00, label01, label10, label11, YD, bit_labels = train_generator(self.H)

        return YP00, YP01, YP10, YP11, label00, label01, label10, label11, YD, bit_labels

    def __len__(self):
        return self.H.shape[0]


class test_Dataset(Dataset):

    def __init__(self, Htest):
        self.H = Htest

    def __getitem__(self, index):
        YP00, YP01, YP10, YP11, label00, label01, label10, label11, YD, bit_labels = test_generator(self.H)

        return YP00, YP01, YP10, YP11, label00, label01, label10, label11, YD, bit_labels

    def __len__(self):
        return self.H.shape[0]


if __name__ == '__main__':
    print("test dataset.py")