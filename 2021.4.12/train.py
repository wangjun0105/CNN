from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import struct
import os
from ofdm import *
from dataset import *
from model import *
import time
from config import *


def worker_init_fn_seed(worker_id):
    seed = 10
    np.random.seed(seed)


def ML_detection(outputs, yd_receive, bit_label, K):
    symset = np.arange(16, dtype=np.uint8).reshape(16, 1)
    codew_set = np.unpackbits(symset, axis=1)[:, -4:].astype(np.int32)
    x_1 = Modulation(codew_set[:, :2].reshape(-1), 2)  # (16,)complex
    x_2 = Modulation(codew_set[:, 2:].reshape(-1), 2)
    x = np.concatenate([np.real(x_1), np.imag(x_1), np.real(x_2), np.imag(x_2)])
    x = torch.as_tensor(x, dtype=torch.float32).reshape(4, 16).cuda()
    h_refine = outputs.reshape(-1, 8, K).transpose(1, 2)  # K * ( ch00 实部bit，虚部bit,  ch10实部bit，虚部bit,...)
    h_refine = h_refine.reshape(-1, K, 4, 2).transpose(2, 3).reshape(-1, K, 4, 2)
    #  (h_r00 h_r10; h_r01 h_r11; h_i00 h_i10; h_i01 h_i11)
    H_est_1 = torch.cat((torch.cat((h_refine[:, :, :2, :], h_refine[:, :, 2:, :]), dim=2),
                         torch.cat((-h_refine[:, :, 2:, :], h_refine[:, :, :2, :]), dim=2)),
                        dim=3)  # (-1,K,4,4)  torch version
    yd_receive = yd_receive.reshape(-1, 4, K).transpose(1, 2)  # (-1,K,4)   # for numpy
    yd_receive_1 = yd_receive[:, :, [0, 2, 1, 3]]  # torch   r0,r1,i0,i1
    bit_label = bit_label.reshape(-1, 4, K).transpose(1, 2)
    yd_ = torch.matmul(H_est_1, torch.unsqueeze(bit_label[:, :, [0, 2, 1, 3]],
                                                3))  # (-1,K,4,1)  y_r0,y_r1,y_i0,y_i1  # real y to compute mse loss
    eu_dist2 = (torch.matmul(H_est_1, x[[0, 2, 1, 3], :]) - torch.unsqueeze(yd_receive_1, 3)).pow(2).sum(
        dim=2)  # (-1,K,16)
    bit_det = eu_dist2.argmin(dim=2, keepdim=True).cpu().numpy()  # (-1,K,1)   0-15
    bb = np.unpackbits(bit_det.astype(np.uint8), axis=2)[:, :,
         -4:]  # (-1,K,4)   first two columns are txantenna1 and last two are antenna2

    bit_label = bit_label.cpu().numpy()  # -1,K * (tx1 实部bit,虚部bit, tx2 实部bit,虚部bit)
    return np.sum(bit_label != bb), bb


gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data1 = open('/home/wangjun/毕业设计/H.bin', 'rb')
data2 = open('/home/wangjun/毕业设计/H_val.bin', 'rb')

H1 = struct.unpack('f' * 2 * 2 * 2 * 32 * 320000, data1.read(4 * 2 * 2 * 2 * 32 * 320000))
H1 = np.reshape(H1, [320000, 2, 4, 32])
Htrain = H1[:, 1, :, :] + 1j * H1[:, 0, :, :]

H2 = struct.unpack('f' * 2 * 2 * 2 * 32 * 2000, data2.read(4 * 2 * 2 * 2 * 32 * 2000))
H2 = np.reshape(H2, [2000, 2, 4, 32])
Htest = H2[:, 1, :, :] + 1j * H2[:, 0, :, :]
ResNet(BasicBlock, [2, 2, 2, 2]).cuda()

model01 = ResNet(BasicBlock, layers).cuda()
model00 = ResNet(BasicBlock, layers).cuda()
model10 = ResNet(BasicBlock, layers).cuda()
model11 = ResNet(BasicBlock, layers).cuda()
num_workers = 2


def training():
    print("start training! ")
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer00 = optim.Adam(model00.parameters(), lr=learning_rate)
    optimizer01= optim.Adam(model01.parameters(), lr=learning_rate)
    optimizer10 = optim.Adam(model10.parameters(), lr=learning_rate)
    optimizer11= optim.Adam(model11.parameters(), lr=learning_rate)

    start_time = time.perf_counter()

    train_dataset = train_Dataset(Htrain)

    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                   worker_init_fn=worker_init_fn_seed)
    test_dataset = test_Dataset(Htest)
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  worker_init_fn=worker_init_fn_seed)

    end_time = time.perf_counter()

    print("load data time%.4f" % (end_time - start_time))

    epoch_avg_loss = []
    train_loss=[]
    test_loss = []
    train_ber = []
    test_ber = []
    for epoch in range(traing_epochs):
        print("========================================")
        print('Processing the', epoch + 1, 'th epoch')
        start_time = time.perf_counter()
        batch_avg_loss = 0.0
        total_loss = 0.0


        for i, data in enumerate(train_loader):

            YP00, YP01, YP10, YP11, label00, label01, label10, label11, YD, bit_labels = data
            YP00 = torch.as_tensor(YP00, dtype=torch.float32).cuda()
            YP01 = torch.as_tensor(YP01, dtype=torch.float32).cuda()
            YP10 = torch.as_tensor(YP10, dtype=torch.float32).cuda()
            YP11 = torch.as_tensor(YP11, dtype=torch.float32).cuda()
            label00 = torch.as_tensor(label00, dtype=torch.float32).cuda()
            label01 = torch.as_tensor(label01, dtype=torch.float32).cuda()
            label10 = torch.as_tensor(label10, dtype=torch.float32).cuda()
            label11 = torch.as_tensor(label11, dtype=torch.float32).cuda()
            YD = torch.as_tensor(YD, dtype=torch.float32).cuda()
            bit_label = torch.as_tensor(bit_labels, dtype=torch.float32).cuda()  # 1024
            #
            output00 = model00(YP00)
            output01 = model01(YP01)
            output10 = model10(YP10)
            output11 = model11(YP11)
            loss00 = criterion(output00, label00)
            loss01 = criterion(output01, label01)
            loss10 = criterion(output10, label10)
            loss11 = criterion(output11, label11)


            optimizer00.zero_grad()
            loss00.backward()
            optimizer00.step()

            optimizer01.zero_grad()
            loss01.backward()
            optimizer01.step()

            optimizer10.zero_grad()
            loss10.backward()
            optimizer10.step()

            optimizer11.zero_grad()
            loss11.backward()
            optimizer11.step()

            output00 = torch.cat((output00[:, 2*np.arange(K)], output00[:, 2*np.arange(K) + 1]),axis = 1)
            output01 = torch.cat((output01[:, 2*np.arange(K)], output01[:, 2*np.arange(K) + 1]),axis = 1)
            output10 = torch.cat((output10[:, 2*np.arange(K)], output10[:, 2*np.arange(K) + 1]),axis = 1)
            output11 = torch.cat((output11[:, 2*np.arange(K)], output11[:, 2*np.arange(K) + 1]),axis = 1 )




            H_hat = torch.cat((output00, output10, output01, output11), axis=1)

            total_loss += (loss00.item() + loss01.item()+loss10.item() + loss11.item()) / 4

            batch_avg_loss += (loss00.item() + loss01.item()+loss10.item() + loss11.item()) / 4

            if (i + 1) % 10 == 0:
                error_ML = 1 - ML_detection(H_hat, YD, bit_label, K)[0] / (1024 * batch_size)
                print(
                    "Epoch: {}/{}, Batch {}/{}, MSE Loss {},acc{}".format(epoch + 1, traing_epochs, i + 1,
                                                                          len(train_loader),
                                                                          batch_avg_loss / 10, error_ML))
                train_loss.append(batch_avg_loss / 10)
                train_ber.append(error_ML)
                batch_avg_loss = 0
                if (i + 1) % 100 == 0:
                    batchtest_loss = 0.0
                    testerror_ML = 0.0
                    with torch.no_grad():
                        for j, data in enumerate(test_loader):
                            YP00, YP01, YP10, YP11, label00, label01, label10, label11, YD, bit_labels = data
                            YP00 = torch.as_tensor(YP00, dtype=torch.float32).cuda()
                            YP01 = torch.as_tensor(YP01, dtype=torch.float32).cuda()
                            YP10 = torch.as_tensor(YP10, dtype=torch.float32).cuda()
                            YP11 = torch.as_tensor(YP11, dtype=torch.float32).cuda()
                            label00 = torch.as_tensor(label00, dtype=torch.float32).cuda()
                            label01 = torch.as_tensor(label01, dtype=torch.float32).cuda()
                            label10 = torch.as_tensor(label10, dtype=torch.float32).cuda()
                            label11 = torch.as_tensor(label11, dtype=torch.float32).cuda()
                            YD = torch.as_tensor(YD, dtype=torch.float32).cuda()
                            bit_label = torch.as_tensor(bit_labels, dtype=torch.float32).cuda()  # 1024
                            #
                            output00 = model00(YP00)
                            output01 = model01(YP01)
                            output10 = model10(YP10)
                            output11 = model11(YP11)
                            loss00 = criterion(output00, label00)
                            loss01 = criterion(output01, label01)
                            loss10 = criterion(output10, label10)
                            loss11 = criterion(output11, label11)

                            output00 = torch.cat((output00[:, 2 * np.arange(K)], output00[:, 2 * np.arange(K) + 1]),
                                                 axis=1)
                            output01 = torch.cat((output01[:, 2 * np.arange(K)], output01[:, 2 * np.arange(K) + 1]),
                                                 axis=1)
                            output10 = torch.cat((output10[:, 2 * np.arange(K)], output10[:, 2 * np.arange(K) + 1]),
                                                 axis=1)
                            output11 = torch.cat((output00[:, 2 * np.arange(K)], output11[:, 2 * np.arange(K) + 1]),
                                                 axis=1)

                            H_hat = torch.cat((output00, output10, output01, output11), axis=1)

                            batchtest_loss += (loss00.item() + loss01.item()+loss10.item() + loss11.item()) / 4
                            testerror_ML += 1 - ML_detection(H_hat, YD, bit_label, K)[0] / (1024 * batch_size)
                        batchtest_loss = batchtest_loss / len(test_loader)
                        testerror_ML = testerror_ML / len(test_loader)
                        print('MSE on testset{},acc{}'.format(batchtest_loss, testerror_ML))
                        test_loss.append(batchtest_loss)
                        test_ber.append(testerror_ML)

        epoch_avg_loss.append(total_loss / len(train_loader))
        print('第%d次循环，MSE on trainset %.4f' % (epoch + 1, epoch_avg_loss[epoch]))
        end_time = time.perf_counter()
        print("per epoch train time%.4f" % (end_time - start_time))
        if epoch > 0:
            if (epoch_avg_loss[epoch - 1] - epoch_avg_loss[epoch] < 0.01):
                break
    np.array(train_loss)
    np.array(train_ber)
    np.array(test_loss)
    np.array(test_ber)
    # np.savetxt("train_loss 10db 32 pilot", train_loss, newline=" ")
    # np.savetxt("train_ber 10db 32 pilot", train_ber, newline=" ")
    # np.savetxt("test_loss 10db 32 pilot", test_loss, newline=" ")
    # np.savetxt("test_ber 10db 32 pilot", test_ber, newline=" ")
    print("optimization finished")
    model00Save = "resnet18 estimate para 00 10db 8pilot.pth"
    model01Save = "resnet18 estimate para 01 10db 8pilot.pth"
    model10Save = "resnet18 estimate para 10 10db 8pilot.pth"
    model11Save = "resnet18 estimate para 11 10db 8pilot.pth"
    torch.save(model00.state_dict(), model00Save)
    torch.save(model01.state_dict(), model01Save)
    torch.save(model10.state_dict(), model10Save)
    torch.save(model11.state_dict(), model11Save)

training()