# Copyright (c) 2017-2018 Katori Lab. All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)
# NOTE: pcrc7c.pyに基づく。誤差の計算のためにfastdtwを導入する
import numpy as np
import scipy.linalg
from numpy.linalg import svd, inv, pinv
import matplotlib as mpl
#mpl.use('Agg')## サーバ上で画像を出力するための設定。ローカルで表示する際はコメントアウトする
import matplotlib.pyplot as plt
import sys
#from arg2x import *
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

import pcrc7e
rc = pcrc7e.PCRC()

from w2v import *


def plot2():
    fig=plt.figure(figsize=(12,9))#figsize=(8,6)

    #ax = fig.add_subplot(4,1,1)
    #ax.cla()
    #ax.plot(U)
    #ax.set_ylabel("U")
    ax = fig.add_subplot(4,1,1)
    ax.cla()
    ax.plot(rc.X)
    ax.set_ylabel("X")

    ax = fig.add_subplot(4,1,2)
    ax.cla()
    ax.plot(D,label="D")
    ax.set_ylabel("D")
    ax.set_ylim([-0.05,1.05])
    #ax.legend()

    ax = fig.add_subplot(4,1,3)
    ax.cla()
    ax.plot(rc.Y,label="Y")
    ax.set_ylim([-0.05,1.05])
    ax.set_ylabel("Y")

    ax = fig.add_subplot(4,1,4)
    R = D-rc.Y
    ax.cla()
    ax.plot(R,label="R")
    ax.set_ylim([-1,1])
    #ax.set_ylim([0,1])
    ax.set_ylabel("R")
    ax.set_xlabel("n")

    plt.savefig(file_fig1)
    plt.show()


### 基本設定
file_csv  = "data_pcrc.csv"
file_fig1 = "data_pcrc_fig1.png"
display = 1
dataset = 21
seed = 0
id = 0
# def config():
#     global file_csv,file_fig1,display,dataset,seed,id
#     for s in sys.argv:
#         file_csv  = arg2a(file_csv,"file_csv=",s)
#         file_fig1 = arg2a(file_fig1,"file_fig1=",s)
#         display   = arg2i(display,"display=",s)
#         dataset   = arg2i(dataset,"dataset=",s)
#         seed      = arg2i(seed,"seed=",s)
#         id        = arg2i(id,"id=",s)
#         rc.Nx      = arg2i(rc.Nx,"Nx=",s)
#         rc.alpha_r = arg2f(rc.alpha_r,"alpha_r=",s)
#         rc.alpha_b = arg2f(rc.alpha_b,"alpha_b=",s)
#         rc.alpha_e = arg2f(rc.alpha_e,"alpha_e=",s)
#         rc.beta_r = arg2f(rc.beta_r,"beta_r=",s)
#         rc.beta_b = arg2f(rc.beta_b,"beta_b=",s)
#         rc.beta_e = arg2f(rc.beta_b,"beta_e=",s)
#         rc.alpha0  = arg2f(rc.alpha0,"alpha0=",s)
#         rc.tau     = arg2f(rc.tau,"tau=",s)
#         rc.eps     = arg2f(rc.eps,"eps=",s)

def output():
    str="%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" \
    % (dataset,id,seed,rc.Nx,rc.alpha_r,rc.alpha_b,rc.alpha_e,rc.beta_r,rc.beta_b,rc.beta_e,rc.alpha0,rc.tau,rc.eps,e0,e1,e2,e3,e4,me1,me2,mdtw1,mdtw2)
    print(str,end="")
    f=open(file_csv,"a")
    f.write(str)
    f.close()

def test_network(T0,T1,D):
        """
        T0: エラー駆動モードの長さ
        T1: テスト期間の長さ（フリーランモードの長さは T1-T0)
        """
        rc.init_network(T1)
        for n in range(T1 - 1):
            d = D[n, :]
            if n<T0:
                rc.alpha_e = 1.0 # Error driven mode
                rc.step_network2(d)
            else:
                rc.alpha_e = 0.0 # Free-running mode
                rc.step_network2(d)


def execute():
    global D
    global e0,e1,e2,e3,e4,me1,me2,mdtw1,mdtw2
    Ny = 100
    (Ttrans,Ttrain,Ttest,Ted,Ntest) = (300,10000,50,10,20)
    Ttotal = Ttrain + Ttest * Ntest # total length
    # UU, DD = generate_coupled_mu_model(Ttotal, Nu=1, Ny=20, I=0.004, G=0.2, alpha=0.0,sampling_interval=25)
    # save_data("data_mu3",UU,DD)
    # UU, DD = load_data("data_mu3")
    DD = load_data()
    # print(vec2word(DD))
    # DD = np.fmax(DD + 0.1, 0)
    #DD = np.tanh(DD + 0.1)

    ### training
    rc.Ny = Ny
    rc.generate_weight_matrix()
    D = DD[0:Ttrain]
    rc.train_network(Ttrans,Ttrain,D)

    ### test
    SUM = np.zeros(Ttest)
    sum_dtw1=0
    sum_dtw2=0
    for i in range(Ntest):
        D = DD[Ttrain + Ttest * i : Ttrain + Ttest * (i + 1)]
        # vec2word(D)
        test_network(Ted,Ttest,D)

        vec2word(rc.Y)

        error = (rc.Y - D) # error
        rmse = np.sqrt( np.sum(error**2,axis=1)/rc.Ny )
        SUM += rmse**2

        dtw1, path1 = fastdtw(rc.Y[Ttrans:Ted],D[Ttrans:Ted], dist=euclidean)
        dtw1 = dtw1/(Ted-Ttrans)/Ny
        sum_dtw1+=dtw1
        dtw2, path2 = fastdtw(rc.Y[Ted:Ttest],D[Ted:Ttest], dist=euclidean)
        dtw2 = dtw2/(Ttest-Ted)/Ny
        sum_dtw2+=dtw2
        print(dtw1,dtw2)

    RMSE = np.sqrt(SUM / Ntest) # このRMSEは長さTestの１次元配列
    mdtw1 = sum_dtw1/Ntest
    mdtw2 = sum_dtw2/Ntest

    e0 = RMSE[Ted + 0]
    e1 = RMSE[Ted + 5]
    e2 = RMSE[Ted + 10]
    e3 = RMSE[Ted + 15]
    e4 = RMSE[Ted + 20]

    me1=np.sum(RMSE[Ttrans:Ted])/(Ted-Ttrans)
    me2=np.sum(RMSE[Ted:Ttest])/(Ttest-Ted)

    # if display :
    #     plot2()

if __name__ == "__main__":
    # config()
    execute()
    output()
