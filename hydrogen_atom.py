import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp

point_num = 101
varphi_values = np.linspace(0,2*np.pi,point_num)
theta_values = np.linspace(0,np.pi,point_num)
cos_theta = np.cos(theta_values)
VARPHI,THETA = np.meshgrid(varphi_values,theta_values)
VARPHI,COS_THETA = np.meshgrid(varphi_values,cos_theta)
def factorial(n):
    result = 1
    if(n==0):
        return result
    for i in range(1,n+1):
        result *= i
    return result

def n_polynomials(n): #只返回表达式不返回函数值 需要搭配sp.lambdify使用
    x = sp.symbols('x')
    f = (x**2-1)**n
    result = 1/(2**n * factorial(n)) * sp.diff(f,x,n)
    return result

def l_m_polynomials(m,f):
    x = sp.symbols('x')
    result = (-1)**m * ((1-x**2)**(m/2))* sp.diff(f,x,m)
    return result

def spherical_harmonics(l,m):
    if(m>l):
        return False
    coefficient = np.sqrt((2*l+1)/(4*np.pi) * factorial((l-np.abs(m))) / factorial(l+np.abs(m)))
    P_l = n_polynomials(l)
    phi = sp.symbols('phi')
    if(m == 0):
        return coefficient*P_l
    if(m>0):
        P_l_m = l_m_polynomials(m,P_l)
        return sp.sqrt(2)*coefficient*sp.cos(m*phi)*P_l_m
    if(m<0):
        P_l_minus_m = l_m_polynomials(-m,P_l)
        return sp.sqrt(2)*coefficient*sp.sin(-m*phi)*P_l_minus_m

x,phi = sp.symbols('x phi')
cos_theta = np.cos(theta_values)
cos_THETA,PHI = np.meshgrid(cos_theta,varphi_values)

func = spherical_harmonics(3,2) #切换不同的角动量

func_numpy = sp.lambdify((x,phi),func,'numpy')

r = func_numpy(COS_THETA,VARPHI)
X = r * np.sin(THETA) * np.cos(VARPHI)
Y = r * np.sin(THETA) * np.sin(VARPHI)
Z = r * np.cos(THETA)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(X, Y ,Z , cmap='viridis',alpha = 0.6)
plt.show()
