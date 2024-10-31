import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# 范德瓦尔斯气体方程 (p+a/Vm^2)(Vm-b)=RT 用变量V，T画图
# mju=V/Cp(T*alpha-1)
a = 3.45 #氦气常量
b = 0.0234  # 氦气的气体常量
N = 1  # 1mol的气体
R = 8.2  # 气体常量
v = np.linspace(0.0000001, 1, 10000)
# x = sp.symbols('x')

#  N*R*V**3 / (p*V**3-a*V+2*a*b ), N*R*V**2/(p*V**3-b*p*V**2+a*V-a*b)
P = (2*a*v-3*a*b)/(b*v**2)
T = ((2*a*v-3*a*b)/(b*v**2)+a/v**2)*(v-b)/R


plt.plot(P, T)
plt.xlim(0,4000) #
plt.ylim(0,60)
plt.xlabel("p/atm")
plt.ylabel("T/K")
plt.show()



