import numpy as np
from decimal import Decimal
import scipy.linalg
from numpy.linalg import svd, inv, pinv
import matplotlib as mpl
# mpl.use('Agg')## サーバ上で画像を出力するための設定。ローカルで表示する際はコメントアウトする
import matplotlib.pyplot as plt
import sys

from esn import EchoStateNet
from preprocess import *

from nltk.translate import bleu_score
from nltk import word_tokenize
import MeCab

from w2v import *


def plot2():
    fig=plt.figure(figsize=(12,9)) # figsize=(8,6)

    # ax = fig.add_subplot(4,1,1)
    # ax.cla()
    # ax.plot(U)
    # ax.set_ylabel("U")
    ax = fig.add_subplot(3,1,1)
    ax.cla()
    ax.plot(esn.X)
    ax.set_ylabel("X")

    ax = fig.add_subplot(3,1,2)
    ax.cla()
    ax.plot(D, label="D")
    # ax.plot(esn.Y, label="Y")
    # ax.set_ylabel("D,Y")
    ax.set_xlabel("n")
    ax.legend()

    ax = fig.add_subplot(3,1,3)
    ax.cla()
    # ax.plot(D, label="D")
    ax.plot(esn.Y, label="Y")
    # ax.set_ylabel("D,Y")
    ax.set_xlabel("n")
    ax.legend()

    plt.savefig(file_fig1)
    plt.show()

esn = EchoStateNet(5, 1500, 5)

### 基本設定
file_csv  = "data_esn2c.csv"
file_fig1 = "data_esn2c_fig1.png"
display = 1
dataset = 10
seed = 0
id = 0


def config():
    global file_csv,file_fig1,display,dataset,seed,id
    for s in sys.argv:
        file_csv  = arg2a(file_csv,"file_csv=",s)
        file_fig1 = arg2a(file_fig1,"file_fig1=",s)
        display   = arg2i(display,"display=",s)
        dataset   = arg2i(dataset,"dataset=",s)
        seed      = arg2i(seed,"seed=",s)
        id        = arg2i(id,"id=",s)
        esn.Nx      = arg2i(esn.Nx,"Nx=",s)
        esn.alpha_r = arg2f(esn.alpha_r,"alpha_r=",s)
        esn.alpha_b = arg2f(esn.alpha_b,"alpha_b=",s)
        esn.beta_r = arg2f(esn.beta_r,"beta_r=",s)
        esn.beta_b = arg2f(esn.beta_b,"beta_b=",s)
        esn.alpha0  = arg2f(esn.alpha0,"alpha0=",s)
        esn.tau     = arg2f(esn.tau,"tau=",s)


def output():
    str="%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" \
    % (dataset,id,seed,esn.Nx,esn.alpha_r,esn.alpha_b,esn.beta_r,esn.beta_b,esn.alpha0,esn.tau,e0,e1,e2,e3,e4)
    print(str,end="")
    f=open(file_csv,"a")
    f.write(str)
    f.close()


def bleu():
    hy = "私はリンゴを食べた"
    hy = tokenizer(hy)
    hy = np.array(hy)[:, 0]
    # hy = "I have a pen"
    # hy = word_tokenize(hy)
    re = "私はカレーを調理した"
    re = tokenizer(re)
    re = np.array(re)[:, 0]
    # re = "I have a apple"
    # re = word_tokenize(re)
    res = [re]

    print(hy)
    print(re)

    # BLEUscore = bleu_score.sentence_bleu(res, hy, weights=(0.5, 0.5)) # n=2
    BLEUscore = bleu_score.sentence_bleu(res, hy, weights=(0.25, 0.25, 0.25, 0.25)) # n=4だとほぼ0になってしまう
    # print("score : {:f}".format(BLEUscore))
    print("score : " + str(BLEUscore))


def execute():
    global D
    global e0,e1,e2,e3,e4

    np.random.seed(seed)

    Ny = 5
    Ntest = 5
    Ttrans = 50 # length of transient,
    Ttrain = 10000 # length of training data
    Ttest = 50 # length of test data
    Ttotal = Ttrain + Ttest * Ntest # total length
    num_iteration = 1
    # UU, DD = generate_coupled_mu_model(Ttotal, Nu=1, Ny=20, I=-0.01, G=0.2, alpha=0.02)
    # save_data("data_mu1",UU,DD)
    # DD = load_data()
    DD = load_data_poincare()
    print(DD)
    # DD, index = load_data()
    print(np.shape(DD))
    # DD = DD + 0.1

    # training
    esn.Ny=Ny
    esn.generate_weight_matrix()

    D = DD[0:Ttrain]
    for i in range(num_iteration):
        esn.train_network(Ttrans,Ttrain,D)
        print("iteration: {} is finish".format(i))

    # test
    SUM = np.zeros(Ttest)
    for i in range(Ntest):
        D = DD[Ttrain + Ttest * i: Ttrain + Ttest * (i + 1)]
        esn.test_network(Ttrans,Ttest,D)
        print("test: {} is finish\n".format(i))
        print(vec2word_poincare(D[0:50]))
        print('\n')
        print(vec2word_poincare(esn.Y[1:]))
        print('\n')
        SUM += esn.rmse**2

    MSE = SUM / Ntest
    RMSE = np.sqrt(MSE)

    # e0 = RMSE[Ttrans + 0]
    # e1 = RMSE[Ttrans + 5]
    # e2 = RMSE[Ttrans + 10]
    # e3 = RMSE[Ttrans + 20]
    # e4 = RMSE[Ttrans + 30]

    if display:
        plot2()

if __name__ == "__main__":
    # config()
    execute()
    output()
    # bleu()
