import numpy as np
import scipy as sp
from scipy.fft import fft2
from scipy.fft import ifft2
from scipy.fft import fftfreq
from scipy.fft import fftshift
import imageio
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import animation
from matplotlib.animation import PillowWriter
import pint

u = pint.UnitRegistry()

#Exp1 - single slit

#set slit distance and wavelength of light
D = 0.1*u.mm
lam = 660*u.nm

# set grid from -2 to 2 mm in 1600 steps
x = np.linspace(-2,2,1600)*u.mm
xv,yv = np.meshgrid(x,x)

# set rectangular slit of height 1mm and width D
Uo = (np.abs(xv)<D/2)*(np.abs(yv)<0.5*u.mm)
Uo = Uo.astype(float)
#plot the slit
plt.figure(figsize=(5,5))
plt.pcolormesh(xv,yv,Uo, cmap='inferno')
plt.xlabel("x (mm)")
plt.ylabel("y(mm)")
plt.show()

#compute fourier trannsform to see how it propagates through slit

A = fft2(Uo)
kx = fftfreq(len(x),np.diff(x)[0])*2*np.pi
kxv , kyv = np.meshgrid(kx,kx)

# plot the fourier transform
plt.figure(figsize=(5,5))
plt.pcolormesh(fftshift(kxv.magnitude),fftshift(kyv.magnitude),np.abs(fftshift(A)))
plt.xlim(-100,100)
plt.ylim(-100,100)
plt.show()

#To get back intensity of light from fouruer transform, we compute the inverse

def get_U(z,k):
    return ifft2(A*np.exp(1j*z*np.sqrt(k**2-kxv**2-kyv**2)))
k = 2*np.pi/lam
d = 3*u.cm # distance of screen from slit

U = get_U(d,k)
plt.figure(figsize=(5,5))
plt.pcolormesh(xv,yv,np.abs(U), cmap='inferno')
plt.show()

#Expt 2 = Double slit pendulum
s = 0.2*u.mm #distance bw slits
D = 0.05*u.mm #slit width
x = np.linspace(-4,4,3200)*u.mm
xv,yv = np.meshgrid(x,x)
Uo = (np.abs(xv-s/2)<D/2)*(np.abs(yv)<2*u.mm) + (np.abs(xv+s/2)<D/2)*(np.abs(yv)<2*u.mm)
Uo = Uo.astype(float)

plt.figure(figsize=(5,5))
plt.pcolormesh(xv,yv,Uo, cmap='inferno')
plt.xlabel("x (mm)")
plt.ylabel("y(mm)")
plt.show()

A = fft2(Uo)
kx = fftfreq(len(x),np.diff(x)[0])*2*np.pi
kxv , kyv = np.meshgrid(kx,kx)

U = get_U(d,k)
plt.figure(figsize=(5,5))
plt.pcolormesh(xv,yv,np.abs(U), cmap='inferno')
plt.show()




