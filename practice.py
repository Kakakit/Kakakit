import numpy as np
import sympy as sp
levi_civita = np.array([[[0, 0, 0], [0, 0, 1], [0, -1, 0]],
                        [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
                        [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]])
# A = np.array([1, 3, 5])
# B = np.array([2, 4, 6])
# C = np.einsum('ijk,j,k->i', levi_civita, A, B)
# print(C)
t = 1  # ev*A单位 电子伏特乘10^-10米
v = 1
eta = -1  #
m = 0.2  # eV
alpha = 1.48e-11  # eV*A^2二维材料Rashba耦合常数
h_bar = 1.0545e-34
kx = np.linspace(-1, 1, 100)
ky = np.linspace(-1, 1, 100)
kz = np.linspace(-1, 1, 10)
KX, KY = np.meshgrid(kx, ky)
k = KX**2 + KY**2
sigma_x = np.array([[0, 1], [1, 0]])  # 泡利矩阵x分量
sigma_y = np.array([[0, -1j], [1j, 0]])  # 泡利矩阵y分量
sigma_z = np.array([[1, 0], [0, -1]])  # 泡利矩阵z分量

H = np.array([[(h_bar*k)**2/(2*m), alpha*(KX-1j*KY)],
              [alpha*(KX+1j*KY), (h_bar*k)**2/(2*m)]])

print(H.shape)
#eigenvalue,eigenvector=np.linalg.eig(H) #求解特征值和特征向量
# print(eigenvalue)
# print(eigenvector)
#omega_z=
KX, KY = sp.symbols('KX KY')
H_sym = sp.Matrix([[(h_bar * (KX**2 + KY**2))**2 / (2 * m), alpha * (KX - 1j * KY)],
                   [alpha * (KX + 1j * KY), (h_bar * (KX**2 + KY**2))**2 / (2 * m)]])

# 计算对kx的导数
dH_dkx = H_sym.diff(KX)

print(dH_dkx,KX,KY)


#def partial_derivative(f,var,point,epsilon=1e-5):  #f=求导的函数，var变量名，point求导的点，epsilon 微小变化量
   #point=np.array(point,dtype=float)


# def f(x):
#     return np.array()#写出H中的元素


# def jacobian(f,x):
#     n = len(x)
#     m = len(f(x))
#     J = np.zeros(m,n)
#     h = 1e-8  #求导辅助增量
#     for i in range (n)
#         x1 = np.array(x, copy=True)
#         x2 = np.array(x, copy=True)  #作为偏导数的相差2h的两个变量
#         x1[i] += h
#         x2[i] -= h
#         J[:,i] = (f(x1)-f(x2))/(2*h)  #用定义式计算偏导数
#     return J

