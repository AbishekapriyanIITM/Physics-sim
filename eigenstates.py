import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal as ttk

n = 2000
dy = 1/n
y = np.linspace(0,1,n+1)

def mL2V(y):
    hvn = [1000] + np.zeros(n+1) + [1000]
    return hvn

plt.plot(y,mL2V(y))
plt.show()

d = 1/dy**2 +  mL2V(y)[1:-1]
e = -1/(2*dy**2)*np.ones(len(d)-1)

w,v = ttk(d,e)

plt.plot(v.T[0]**2)
plt.plot(v.T[1]**2)
plt.plot(v.T[2]**2)
plt.plot(v.T[3]**2)
plt.show()

plt.bar(np.arange(0,10,1),w[0:10])
plt.show()
