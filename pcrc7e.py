# Copyright (c) 2017-2019 Katori Lab. All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)
# NOTE: pcrc7e.pyに基づく。クラスを別ファイルにした。main_pcrc7x.pyと一緒に使う。
# NOTE: 若干書き換えてあります
import numpy as np
import scipy.linalg
from numpy.linalg import svd, inv, pinv

from w2v import *

class PCRC:
    Nu = 2   #size of input
    Nx = 200 #size of dynamical reservior
    Ny = 2   #size of output

    alpha_r = 0.15
    alpha_b = 0.15
    alpha_e = 0
    beta_r = 0.05
    beta_b = 0.05
    beta_e = 0
    alpha0 = 0.05
    alpha1 = 0.4 # 自己再帰結合
    tau = 3.0
    beta = 1.5
    lambda0 = 0.1
    eps=0.2
    num_iteration = 10

    def __init__(self):
        return None
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
        E = np.identity(Nx)
        self.Wr = self.Wr + self.alpha1 * E
        # print("lamda_max",lambda_max)
        # print("Wr:")
        # print(self.Wr)

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
    def fx(self,x):
        #return np.tanh(x)
        return (np.tanh(x*self.beta)+1.0)/2.0
        #return np.fmax(x,0)-np.fmax(x-1,0)

    def fy(self,x):
        #return np.tanh(x)
        return np.fmax(x,0)
        #return np.fmax(x,x*self.eps)# leaky ReLu
        #return x*x/(self.eps+x)*np.heaviside(x,0)

    def fyi(self,x):
        #return np.arctanh(x)
        return x*1.0
        #return np.fmax(x,x/self.eps)
        #return 0.5*(x + np.sqrt( x*x + 4*self.eps*x))

    def fr(self,x):
        #return np.fmax(x,0)
        return x
        #return np.fmax(x,x*self.eps)

    def step_network(self,x,y,d):
        r = d - y
        s = self.fr(r) * self.alpha_e
        sum = np.zeros(self.Nx)
        #sum += self.Wr @ x
        sum += self.Wr @ (2.0 * x - 1.0)
        sum += self.Wb @ y
        sum += self.Wb @ s
        #x = x + 1.0 / self.tau * (-self.alpha0 * x + np.tanh(sum))
        x = x + 1.0 / self.tau * (-x + self.fx(sum) )
        y = self.fy(self.Wo @ x)

        return x, y
    def step_network2(self,d):
        self.x, self.y, self.r, self.s = self.step_network(self.x,self.y,self.s * self.alpha_e,d)
        self.n += 1
        self.record_network()

    def record_network(self):
        if self.T1 > 0:
            self.X[self.n, :] = self.x
            self.Y[self.n, :] = self.y

    def init_network(self,T1=0):
        # 記録(record)用の配列を初期化
        self.T1=T1
        if self.T1 > 0:
            self.X = np.zeros((self.T1, self.Nx))
            self.Y = np.zeros((self.T1, self.Ny))

        # ネットワークの状態の初期化
        self.n = 0
        self.x = np.random.normal(0, 0.1, self.Nx)
        self.y = np.zeros(self.Ny)
        self.record_network()

    def update_network(self,T0,T1,D):
        self.init_network(T1)
        for n in range(T1 - 1):
            d = D[n,:]
            self.x, self.y = self.step_network(self.x ,d,d)
            self.n += 1
            self.record_network()

        ### prepare state collecting matrix
        M = self.X[T0:, :]
        invD = self.fyi(D)
        G = invD[T0:, :]
        ### Ridge regression
        E = np.identity(self.Nx)
        TMP1 = inv( M.T @ M + self.lambda0 * E)
        WoT = TMP1 @ M.T @ G
        self.Wo = WoT.T
        #print("WoT\n", WoT)

    def train_network(self,T0,T1,D):
        for i in range (self.num_iteration):
            self.update_network(T0,T1,D)
            print("iteration:{}".format(i))
            if i % 10 == 0:
                print(vec2word(self.Y))

    def step_network2(self,d):
        self.x, self.y = self.step_network(self.x,self.y,d)
        self.n += 1
        self.record_network()

    def test_network2(self,T0,T1,D):
        """
        T0: エラー駆動モードの長さ
        T1: テスト期間の長さ（フリーランモードの長さは T1-T0)
        """
        self.init_network(T1)
        for n in range(T1 - 1):
            d = D[n, :]
            if n<T0:
                self.alpha_e = 1.0 # Error driven mode
                self.step_network2(d)
            else:
                self.alpha_e = 0.0 # Free-running mode
                self.step_network2(d)
