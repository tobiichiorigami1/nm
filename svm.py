import sys
import random

import numpy as np
from math import sqrt
from scipy.spatial.distance import squareform,cdist

import matplotlib
matplotlib.use('wx')
import matplotlib.pylab as plt

random.seed(1024*1024)

from common import read_dense_data
from common import map_label

#定义L为内积,lambda为匿名函数的标记
L = lambda Xi,Xj:Xi * Xj.T

#若核函数为线性的，直接求取Xi*Xj
linear_kernel = L

#若核函数非线性，采用高斯核

def rbf_kernel(X_i,X_j.sigma=0.1):
#κ(x,xi)=exp(−||x−xi||2/δ2)为高斯核，首先求取X_i-X_j的第二范数
    pairwise_dists=cdist(X_i,X_j,'euclidean')
    K = np.exp(-(pairwise_dists**2)/(sigma**2)
#返回K，并将K转为矩阵
    return np.matrix(K)

"""SVM向量机是一个约束条件下求极值问题，我们通过拉格朗日数乘法改变式子，
变为二次规划求解问题，可以用二次规划求解，也可以用SMO求解，对于核函数非
线性的，SMO算法好"""
class SMO:
def _init_(self):
               self.model=None
def train(self,X,Y,K=L,,C=1.0,tol=1e-3,max_passes =5):
#根据x的维度确定阿拉法的维度.
               m,n=X.shape
               alphas=np.matrix(np.zeros([m,1]))
               #初始化b
               b=0.0
               #定义Kij:
               K_cache = K(X,X)
               print >> sys.stderr.'Done with K_cache'
#迭代次数
               iter = 0
               passes = 0
               while passes <max_passes:
                   iter +=1
#记录错误输出？
                   if iter % 10==0:sys.stderr.write('.')
                   if iter % 500 ==0:sys.stderr.write('%d Ters\n'% iter)
                   num_changed_alphas=0
                   for i in range(m):
                       #fx=w.TX+b,w=阿拉法*yi*xi累加
                       fx_i=alphas.T*np.multiply(Y,k_chache[:,i])+b
#获取真实y,y=-1or+1
                       y_i=Y[i]
#误差函数
                       E_i=fx_i-y_i
               #获取阿拉法1,一个存来运算new，一个当做old存
                       alpha_i=alpha_ii=alphas[i]                       
                #如果阿拉法1满足KKT条件，且误差小于规定误差（y_i只有1和-1两个值），取阿拉法2
                       if(y_i*E_i<-tol and alpha_i <C)or(y_i*Ei>tol and alpha_i>0.0):
                #取阿拉法2，阿拉法只要不是阿拉法1即可
                           while True:
                               j= random.randint(0,m-1)
                               if i !=j:break
                #同上求fx
                           fx_j=alphas.T*np.multiply(Y,k_cache[:,i])+b
                           y_j=Y[j]
                           E_j=fx_j -y_j
                           alpha_j=alpha_jj=alphas[j]
                        """阿拉法i乘以yi加上阿拉法j乘以yj为一个常数，
                           因为我们考虑，其他阿拉法为常数,当y_i!=y_j时
                           有阿拉法i-阿拉法j等于一个常数或者阿拉法j-阿拉法i等于
                           一个常数，结合约束条件阿拉法大于等于0，小于等于C，两者
                           综合起来确定确界上确界为C与C+两者差值的最小值，下确界为
                           0与两者的差的最大值"""
                           if y_i != y_j:
                               L=max(0.0,alpha_j-alpha_i)
                               H=min(C,C+alpha_j-alpha_i)
                        """如果相等则阿拉法i加阿拉法j等于一个常数，
                           或者其相反数等于一个常数,此时上确界变为
                           该常数和C之间取最小，下确界变为0和该常数减去C
                           取最大"""
                           else:
                               L=max(C,alpha_i+alpha_j-C)
                               M=min(0,alpha_i+alpha_j)
                           #如果上确界等于下确界，则阿拉法的值就确定下来了，跳过本次循环剩余操作
                           if L == H:continue                      

                           eta=2*K_cache[i,j]-K_cache[i,i]-k_cache[j,j]
                           #eta为负数，如果大于0，则不满足条件，跳过本次循环剩余操作
                           if eta>=0.0:continue
                           #更新阿拉法的值
                           alpha_j=alpha_j-(y_j*(E_i-E_j)/eta)
                           #注意到alpha的上下确界，只有alpha处于界中才能取到自己本身
                           if alpha_j>H:alpha_j=H
                           if alpha_j<L:alpha_j=L
                           #如果更新前与更新后相差小于规定，那就不更新了
                           if abs(alpha_jj - alpha_j) <tol ;continue
                           """前面已知这两个阿拉法之和为常数，则更新后还是常数，
                              有y1乘以阿拉法1老+y2乘以阿拉法2老等于y1乘以阿拉法1新
                              加上y2乘以阿拉法2新，等式两边同时乘以y1y2,有y2乘以阿拉法1老
                              加y1乘以阿拉法2老等于y2乘以阿拉法1新加上y1乘以阿拉法2新，所以
                              阿拉法1新等于阿拉法1老加上y1y2乘以阿拉法2老减阿拉法2新"""
                           alpha_i=alpha_i+(y_i*y_j*(alpha_jj-alpha_j))
                           """如果阿拉法能取到自己本身的话，误差是0，又有贝塔阿拉法等于新减旧代入得"""
                           b_i = b - E_i - y_i * (alpha_i - alpha_ii) * K_cache[i, i] - y_j * (alpha_j - alpha_jj) * K_cache[i, j]
                           b_j = b - E_j - y_i * (alpha_i - alpha_ii) * K_cache[i, j] - y_j * (alpha_j - alpha_jj) * K_cache[j, j]

                           if alpha_i>0.0 and alpha_i<C:
                               b= b_j
                           elif alpha_j>0.0 and alpha_j<C:
                               b=b_i
                           else:
                               b=(b_i+b_j)/2

                           alphas[i]=alpha_i
                           alphas[j]=alpha_j

                           num_changed_alphas=num_changed_alphas+1
                    #如果成功迭代次数加一
                    if num_changed_alphas == 0:
                        passes +=1
                    else:
                        passes =0
               sys.stderr.write('\nDone training with Iter %d\n' % iter)
            #存储模型参数值,注意此时的样本只剩下支持向量，满足阿拉法大于0
               self.modle=dict()
               alpha_index=[index for index,alpha in enumerate(alphas) if alpha>0]

               self.model['X']=X[alpha_index]
               self.model['Y']=Y[alpha_index]
               self.model['kernel']=K
               self.model['alphas']=alphas[alphas_index]
               self.model['b']=b
               self.model['w']=X.T*np.multiply(alphas,Y)

            #对模型进行预测性能评估
               def predict(self,X_test):
                   m,n=X_test.shape
               #初始化fx为矩阵
                   fx=np.matrix(np.zeros[m,1])
                   #如果核函数为线性的
                   if self.model['kernel']=L:
                       w=self.model['w']
                       b=self.model['b']
                       fx=X_test*w+b
                   else:
                       alphas=self.model['alphas']
                       X=self.model['X']
                       Y=aelf.model['Y']
                       K=self.model['kernel']
                       b=self.model['b']
                       fx =np.multiply(np.tile(Y,[m]),K(X,X_test)).T*alphas+b
                       return fx
               def test(self,X,Y):
                   fx=self.pridict(X)
                   Y_pred = np.matrix(np.zeros(Y.shape))
                   Y_pred[np.where(fx>=0)] =1
                   Y_pred[np.where(fx<0)]=-1
                   #求出测试集中预测正确的个数
                   P=np.matrix(np.zeros(Y.shape))
                   p[np.where(Y_pred==Y)]=1
                   #求出正确率
                   return 1.0*p.sum()/len(Y)
                #随机梯度下降法
               class Pegasos:
                   def _init_(self):
                       self.w=None
                   def train(self,X,Y,T=1000,lamb=0.01,k=80):
                       
                
                   
               
                       
               
               
                    
                               
                            
                           
        
        
