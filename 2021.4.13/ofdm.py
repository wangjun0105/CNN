from __future__ import division
import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
import struct
import os
import math
from config import *


def Modulation(bits, mu):
    bit_r = bits.reshape((int(len(bits) / mu), mu))
    return 0.7071 * (2 * bit_r[:, 0] - 1) + 0.7071j * (2 * bit_r[:, 1] - 1)  # This is just for QAM modulation


def deModulation(Q):
    Qr = np.real(Q)
    Qi = np.imag(Q)
    bits = np.zeros([256, 2])
    bits[:, 0] = Qr > 0
    bits[:, 1] = Qi > 0
    return bits.reshape([-1])  # This is just for QAM modulation


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)


def addCP(OFDM_time, CP, CP_flag, mu, K):
    if CP_flag == False:
        # add noise CP
        bits_noise = np.random.binomial(n=1, p=0.5, size=(K * mu,))
        codeword_noise = Modulation(bits_noise, mu)
        OFDM_data_nosie = codeword_noise
        OFDM_time_noise = np.fft.ifft(OFDM_data_nosie)
        cp = OFDM_time_noise[-CP:]
    else:
        cp = OFDM_time[-CP:]  # take the last CP samples ...
    # cp = OFDM_time[-CP:]
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning


def channel(signal, channelResponse, SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved ** 2))
    sigma2 = signal_power * 10 ** (-SNRdb / 10)
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*convolved.shape) + 1j * np.random.randn(*convolved.shape))
    return convolved + noise


def removeCP(signal, CP, K):
    return signal[CP:(CP + K)]


def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)


def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest


def get_payload(equalized):
    return equalized[dataCarriers]


def PS(bits):
    return bits.reshape((-1,))


def Clipping(x, CL):
    sigma = np.sqrt(np.mean(np.square(np.abs(x))))
    CL = CL * sigma
    x_clipped = x
    clipped_idx = abs(x_clipped) > CL
    x_clipped[clipped_idx] = np.divide((x_clipped[clipped_idx] * CL), abs(x_clipped[clipped_idx]))
    return x_clipped


def ofdm_simulate(codeword, channelResponse, SNRdb, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers,
                  Clipping_Flag):
    # --- training inputs ----

    CR = 1
    OFDM_data = np.zeros(K, dtype=complex)
    OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarrier
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time, CP, CP_flag, mu, 2 * K)
    # OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    if Clipping_Flag:
        OFDM_TX = Clipping(OFDM_TX, CR)  # add clipping
    OFDM_RX = channel(OFDM_TX, channelResponse, SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX, CP, K)
    OFDM_RX_noCP = np.fft.fft(OFDM_RX_noCP)
    # OFDM_RX_noCP = removeCP(OFDM_RX)
    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword, mu)
    if len(codeword_qam) != K:
        print('length of code word is not equal to K, error !!')
    symbol = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword, CP, CP_flag, mu, K)
    # OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    if Clipping_Flag:
        OFDM_withCP_cordword = Clipping(OFDM_withCP_cordword, CR)  # add clipping
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse, SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword, CP, K)
    OFDM_RX_noCP_codeword = np.fft.fft(OFDM_RX_noCP_codeword)
    AA = np.concatenate((np.real(OFDM_RX_noCP), np.imag(OFDM_RX_noCP)))
    CC = OFDM_RX_noCP / np.max(AA)
    BB = np.concatenate((np.real(OFDM_RX_noCP_codeword), np.imag(OFDM_RX_noCP_codeword)))

    return np.concatenate((AA, BB)), CC  # sparse_mask


def LS(X, HMIMO, SNRdb, P, flag):
    pilotValue, pilotCarriers = pilot(P)
    if flag == 1:
        cpflag, CR = 0, 0
    elif flag == 2:
        cpflag, CR = 0, 1
    else:
        cpflag, CR = 1, 0
    allCarriers = np.arange(K)
    dataCarriers = [val for val in allCarriers if not (val in pilotCarriers)]

    bits0 = X[0]
    bits1 = X[1]
    pilotCarriers1 = pilotCarriers[0:2 * P:2]
    pilotCarriers2 = pilotCarriers[1:2 * P:2]
    signal_output00, para = ofdm_simulate(bits0, HMIMO[0, :], SNRdb, mu, cpflag, K, P, CP, pilotValue[0:2 * P:2],
                                          pilotCarriers1, dataCarriers, CR)
    signal_output01, para = ofdm_simulate(bits0, HMIMO[1, :], SNRdb, mu, cpflag, K, P, CP, pilotValue[0:2 * P:2],
                                          pilotCarriers1, dataCarriers, CR)
    signal_output10, para = ofdm_simulate(bits1, HMIMO[2, :], SNRdb, mu, cpflag, K, P, CP, pilotValue[1:2 * P:2],
                                          pilotCarriers2, dataCarriers, CR)
    signal_output11, para = ofdm_simulate(bits1, HMIMO[3, :], SNRdb, mu, cpflag, K, P, CP, pilotValue[1:2 * P:2],
                                          pilotCarriers2, dataCarriers, CR)

    Xp0 = pilotValue[0:2 * P:2]
    Xp1 = pilotValue[1:2 * P:2]

    H_tilde00 = np.zeros(len(pilotCarriers1), dtype=complex)
    H_tilde01 = np.zeros(len(pilotCarriers1), dtype=complex)
    H_tilde10 = np.zeros(len(pilotCarriers1), dtype=complex)
    H_tilde11 = np.zeros(len(pilotCarriers1), dtype=complex)

    for i in range(len(pilotCarriers1)):
        H_tilde00[i] = (signal_output00[pilotCarriers1[i]] + 1j * signal_output00[K + pilotCarriers1[i]]) / Xp0[i]
        H_tilde01[i] = (signal_output01[pilotCarriers1[i]] + 1j * signal_output01[K + pilotCarriers1[i]]) / Xp0[i]
        H_tilde10[i] = (signal_output10[pilotCarriers2[i]] + 1j * signal_output10[K + pilotCarriers2[i]]) / Xp1[i]
        H_tilde11[i] = (signal_output11[pilotCarriers2[i]] + 1j * signal_output11[K + pilotCarriers2[i]]) / Xp1[i]



    return H_tilde00, H_tilde01, H_tilde10, H_tilde11


def MMSE(H_tilde, K, P, h, SNR, flag):
    P = P * 2
    Nps = K / P

    k = np.arange(len(h))
    hh = np.dot(h, np.transpose(np.conj(h)))
    tmp = h * np.conj(h) * k
    r = sum(tmp) / hh
    r2 = np.dot(tmp, np.transpose(k)) / hh
    tau_rms = math.sqrt(r2 - r * r)
    df = 1 / K
    j2pi_tau_df = 2j * math.pi * tau_rms * df
    K1 = ((np.arange(K)).repeat(int(P / 2))).reshape(K, int(P / 2))

    if flag == 0:
        K2 = (np.tile(np.arange(0, P, 2), K)).reshape(K, int(P / 2))
        K3 = ((np.arange(0, P, 2)).repeat(int(P / 2))).reshape(int(P / 2), int(P / 2))
        K4 = (np.tile(np.arange(0, P, 2), int(P / 2))).reshape(int(P / 2), int(P / 2))

    else:
        K2 = (np.tile(np.arange(1, P, 2), K)).reshape(K, int(P / 2))
        K3 = ((np.arange(1, P, 2)).repeat(int(P / 2))).reshape(int(P / 2), int(P / 2))
        K4 = (np.tile(np.arange(1, P, 2), int(P / 2))).reshape(int(P / 2), int(P / 2))

    rf = np.divide(1, (1 + j2pi_tau_df * (K1 - K2 * Nps)))
    rf2 = np.divide(1, (1 + j2pi_tau_df * (K3 - K4) * Nps))
    Rhp = rf
    Rpp = rf2 + np.divide(np.eye(len(H_tilde), len(H_tilde)), SNR)
    H_MMSE = np.dot(np.dot(Rhp, np.linalg.inv(Rpp)), np.transpose(H_tilde))

    h_MMSE = np.fft.ifft(H_MMSE, K)
    h_MMSE = h_MMSE[0:len(h)]
    H_MMSE = np.fft.fft(h_MMSE, K)
    return H_MMSE


def pilot(P):
    Pilot_file_name = 'Pilot_' + str(P * 2)
    if os.path.isfile(Pilot_file_name):
        # load file
        bits = np.loadtxt(Pilot_file_name, delimiter=',')
    else:
        # write file
        bits = np.random.binomial(n=1, p=0.5, size=(K * mu,))
        np.savetxt(Pilot_file_name, bits, delimiter=',')

    pilotValue = Modulation(bits, mu)

    allCarriers = np.arange(K)
    pilotCarriers = np.arange(0, K, K // (2 * P))

    return pilotValue, pilotCarriers


def MIMO(X, HMIMO, SNRdb, flag, P):
    pilotValue, pilotCarriers = pilot(P)

    if flag == 1:
        cpflag, CR = 0, 0
    elif flag == 2:
        cpflag, CR = 0, 1
    else:
        cpflag, CR = 1, 0
    allCarriers = np.arange(K)
    dataCarriers = [val for val in allCarriers if not (val in pilotCarriers)]

    bits0 = X[0]
    bits1 = X[1]
    pilotCarriers1 = pilotCarriers[0:2 * P:2]
    pilotCarriers2 = pilotCarriers[1:2 * P:2]
    signal_output00, para = ofdm_simulate(bits0, HMIMO[0, :], SNRdb, mu, cpflag, K, P, CP, pilotValue[0:2 * P:2],
                                          pilotCarriers1, dataCarriers, CR)
    signal_output01, para = ofdm_simulate(bits0, HMIMO[1, :], SNRdb, mu, cpflag, K, P, CP, pilotValue[0:2 * P:2],
                                          pilotCarriers1, dataCarriers, CR)
    signal_output10, para = ofdm_simulate(bits1, HMIMO[2, :], SNRdb, mu, cpflag, K, P, CP, pilotValue[1:2 * P:2],
                                          pilotCarriers2, dataCarriers, CR)
    signal_output11, para = ofdm_simulate(bits1, HMIMO[3, :], SNRdb, mu, cpflag, K, P, CP, pilotValue[1:2 * P:2],
                                          pilotCarriers2, dataCarriers, CR)
    signal_output0 = signal_output00 + signal_output10
    signal_output1 = signal_output01 + signal_output11
    output = np.concatenate((signal_output0, signal_output1))
    output=np.reshape(output,[8,-1])
    output=np.transpose(np.reshape(output,[8,-1]),[1,0])#(256,8)

    return signal_output00, signal_output01, signal_output10, signal_output11, np.reshape(output,[-1])


if __name__ == '__main__':
    print("test ofdm.py")