import matplotlib.pyplot as plt
import numpy as np
import math as mt
import scipy
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import eigh

# 本征值问题
# # 定义参数
# hbar = 1.0   # 普朗克常数
# m = 1.0      # 质量
# a = 1.0      # 宽度
# N = 100      # 网格点数
# dx = a / (N + 1)  # 步长
#
# # 构造有限差分矩阵
# diagonal = np.ones(N) * (-2)
# off_diagonal = np.ones(N-1)
#
# # 拉普拉斯算子离散化
# H = (-hbar**2 / (2 * m * dx**2)) * (np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1))
#
# # 求解特征值和特征向量
# eigenvalues, eigenvectors = eigh(H)
#
# # 取前几个特征值和对应的特征向量
# num_eigenvalues = 5
# eigenvalues = eigenvalues[:num_eigenvalues]
# eigenvectors = eigenvectors[:, :num_eigenvalues]
#
# # 绘制结果
# x = np.linspace(0, a, N)
# for i in range(num_eigenvalues):
#     plt.plot(x, eigenvectors[:, i], label=f'Eigenvalue {i+1}: {eigenvalues[i]:.2f}')
# plt.xlabel('x')
# plt.ylabel('psi(x)')
# plt.legend()
# plt.title('Eigenfunctions in an Infinite Potential Well')
# plt.show()



#abc
# def f(x):
#     return np.array([x[0]**2 + x[1], x[0] + x[1]**2])
#
# # 计算雅可比矩阵
# def jacobian(f, x):
#     n = len(x)
#     m = len(f(x))
#     J = np.zeros((m, n))
#     h = 1e-8  # 一个小的增量
#     for i in range(n):
#         x1 = np.array(x, copy=True)
#         x2 = np.array(x, copy=True)
#         x1[i] += h
#         x2[i] -= h
#         J[:, i] = (f(x1) - f(x2)) / (2 * h)
#     return J
#
# x = np.array([1.0, 2.0])
# J = jacobian(f, x)
# print("Jacobian matrix:\n", J)

# x, y = sp.symbols('x y')
#
# # 定义方程组
# equations = [sp.Eq(x + y, 2), sp.Eq(x - y, 0)]
#
# # 求解方程组
# solution = sp.solve(equations, [x, y])
#
# print(solution)
# 第一次物理实验数据图
# x=np.linspace(0,80,160)
# y=[0,0,0,0,0,0,0,0,0,0,
#    0,0,0,0,0,0,0,0,0,0.2,
#    0.7,1.7,2.9,4.8,7.0,8.9,11.1,13.3,14.5,16.3,
#    17.8,18.6,19.8,21.2,21.9,23.0,23.8,24.2,24.8,25.3,
#    25.8,25.8,25.8,25.8,25.7,25.5,25.2,24.9,24.6,24.3,
#    24.4,24.8,25.6,26.8,28.5,30.5,32.4,34.6,36.7,38.7,
#    39.7,41.3,42.4,42.5,42.8,42.5,41.8,40.5,38.8,37.1,
#    34.8,32.6,20.9,29.4,29.2,30.1,32.2,35.6,39.0,43.3,
#    47.6,51.7,54.5,57.5,59.6,60.4,60.7,59.9,58.5,55.7,
#    52.2,48.7,44.0,39.2,35.5,31.9,30.2,30.8,33.7,38.6,
#    44.7,50.3,57.0,62.6,67.3,72.0,75.3,76.9,77.9,77.5,
#    76.0,73.1,68.8,64.3,58.2,51.5,45.8,39.7,35.1,33.3,
#    34.6,38.9,45.5,52.0,60.2,68.0,74.2,80.5,85.8,89.0,
#    91.0,92.7,92.7,91.3,88.7,85.1,79.6,72.8,66.7,58.9,
#    51.4,45.4,42.4,42.0,45.0,49.7,57.3,65.6,72.7,80.7,
#    88.2,93.3,98.7,102.7,104.8,106.1,106.0,104.6,101.6,97.2]
#
# plt.plot(x,y)
# plt.xlabel('U_G2k/V')
# plt.ylabel('I/nA')
# plt.legend
# plt.show()
from sympy import *
import numpy as np

x = Symbol('x')
y = x**2 + 1

yprime = y.diff(x)
x=np.linspace(-1,1,10)
print(yprime,x)