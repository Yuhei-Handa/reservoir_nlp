
import numpy as np
import scipy.linalg
from numpy.linalg import svd, inv, pinv
import matplotlib as mpl
# mpl.use('Agg')## サーバ上で画像を出力するための設定。ローカルで表示する際はコメントアウトする
import matplotlib.pyplot as plt
import sys

# from w2v import *
from preprocess import *
from logistic import *

class EchoStateNet:

    def __init__(self, Nu, Nx, Ny):
        self.params = {}
        # self.params['Wo']
        self.Nu = Nu
        self.Nx = Nx
        self.Ny = Ny
        # alpha_r = 0.8
        # alpha_b = 0.8
        # alpha_i = 0.2
        # beta_r = 0.1
        # beta_b = 0.1
        # beta_i = 0.1
        # alpha0 = 0.7
        # tau = 1
        # lambda0 = 0.1
        self.alpha_r = 0.5
        self.alpha_b = 0.5
        # self.alpha_i = 0.2
        self.beta_r = 0.7
        self.beta_b = 0.5
        # self.beta_i = 0.1
        self.alpha0 = 0.8
        self.tau = 1
        self.lambda0 = 0.3

    def generate_weight_matrix(self):
        ### NOTE:ローカル変数を定義する。「self.」と書くと長くなるので。
        Nx = self.Nx
        Ny = self.Ny
        Nu = self.Nu

        ### Wr
        Wr0 = np.zeros(Nx * Nx)
        nonzeros = Nx * Nx * self.beta_r
        Wr0[0:int(nonzeros / 2)] = 1
        Wr0[int(nonzeros / 2):int(nonzeros)] = -1
        np.random.shuffle(Wr0)
        Wr0 = Wr0.reshape((Nx, Nx))
        v = scipy.linalg.eigvals(Wr0)
        lambda_max = max(abs(v))
        self.Wr = Wr0 / lambda_max * self.alpha_r

        # print("lamda_max",lambda_max)
        # print("Wr:")
        # print(Wr)

        ### Wb
        Wb0 = np.zeros(Nx * Ny)
        Wb0[0:int(Nx * Ny * self.beta_b / 2)] = 1
        Wb0[int(Nx * Ny * self.beta_b / 2):int(Nx * Ny * self.beta_b)] = -1
        np.random.shuffle(Wb0)
        Wb0 = Wb0.reshape((Nx, Ny))
        self.Wb = Wb0 * self.alpha_b
        # print("Wb:")
        # print(Wb)

        ### Wo
        Wo = np.ones(Ny * Nx)
        Wo = Wo.reshape((Ny, Nx))
        self.Wo = Wo
        self.Wo = np.ones(Ny * Nx).reshape((Ny, Nx))
        # print(Wo)

        ### Wa
        # Wa = np.ones(Nx * Nx)

    def step_network(self,x,y):
        sum = np.zeros(self.Nx)
        # sum += self.Wi @ u
        sum += self.Wr @ x
        sum += self.Wb @ y
        # x = x + 1.0 / self.tau * (-self.alpha0 * x + np.tanh(sum))
        x = np.tanh(sum)
        # x = self.sigmoid(sum)
        # if np.count_nonzero(np.isnan(self.Wo)) > 0:
        #     print(self.Wo)
        # y = np.tanh(self.Wo @ x) # ここを外す
        y = self.Wo @ x
        # y= self.sigmoid(self.Wo @ x)

        # y = self.softmax(y)

        # print(np.shape(y))

        # p_power = y ** 4

        # y = np.random.choice(np.arange(len(y)), len(y), p=y)
        return x, y

    def train_network(self,T0,T1,D):
        self.X = np.zeros((T1, self.Nx))
        self.Y = np.zeros((T1, self.Ny))
        x = np.random.uniform(-1, 1, self.Nx) * 0.2
        y = np.zeros(self.Ny)
        n = 0
        self.X[n, :] = x
        self.Y[n, :] = y
        for n in range(T1 - 1):
            #u = U[n, :]
            d = D[n,:]
            x, y = self.step_network(x,d) # teacher forcing (y is replaced with d)
            self.X[n + 1, :] = x
            self.Y[n + 1, :] = y

        ### prepare state collecting matrix
        M = self.X[T0:, :]
        # invD = np.arctanh(D)
        # シグモイド関数の逆関数を噛ませる
        # print(np.shape(D))
        invD = D
        # invD = self.logit(D)
        # print(invD[invD==1])
        G = invD[T0:, :]
        ### Ridge regression
        E = np.identity(self.Nx)
        TMP1 = inv( M.T @ M + self.lambda0 * E)
        WoT = TMP1 @ M.T @ G
        self.Wo = WoT.T
        #print("WoT\n", WoT)

    # ソフトマックス関数
    def softmax(self, a):
        # 一番大きい値を取得
        c = np.max(a)
        # 各要素から一番大きな値を引く（オーバーフロー対策）
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        # 要素の値/全体の要素の合計
        y = exp_a / sum_exp_a

        return y

    # シグモイド関数
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # ロジット関数(シグモイド関数の逆関数)
    def logit(self, x):
        ones = np.ones(x.shape)
        return np.log(x / (ones - x))

    # 交差エントロピー誤差(分類問題の場合)
    def cross_entropy_error(y, t):
        delta = 1e-7 # マイナス無限大を発生させないように微小な値を追加する
        return -np.sum(t * np.log(y + delta))

    # 交差エントロピー誤差
    def loss(h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    # ロジスティック回帰
    def get_loss(Y_true,Y_pred):
        return -np.sum(Y_true * np.log(Y_pred)) / Y_true.shape[0]

    def test_network(self,T0,T1,D):
        self.X = np.zeros((T1, self.Nx))
        self.Y = np.zeros((T1, self.Ny))
        x = np.random.uniform(-1, 1, self.Nx) * 0.2
        y = np.zeros(self.Ny)
        n = 0
        self.X[n, :] = x
        self.Y[n, :] = y
        for n in range(T1 - 1):
            #u = U[n, :]
            d = D[n, :]
            if n < T0:  # teacher forcing
                b = d
            else:
                b = y
            x, y = self.step_network(x,b)
            self.X[n + 1, :] = x
            self.Y[n + 1, :] = y

        # evaluation
        error = (self.Y - D)
        mse = np.sum(error**2,axis=1)/self.Ny
        self.rmse = np.sqrt(mse)

        #mse = np.sum( (self.Y[T0:]-D[T0:])**2 )/(T1-T0)/self.Ny
        #rmse = np.sqrt(mse)
        #print(mse,rmse)
        #return rmse

    def init_network(self):
        self.x2 = np.random.uniform(-1, 1, self.Nx) * 0.1
        self.y2 = np.zeros(self.Ny)

    def run_network(self,u):
        self.x2, self.y2 = self.step_network(u,self.x2,self.y2)
        return self.y2