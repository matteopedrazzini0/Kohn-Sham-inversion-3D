import numpy as np
from consts3D import b_x, b_y, b_z
from scipy.integrate import simpson
from math import factorial
from numpy.polynomial.hermite import Hermite

def HO3D_1(nx, ny, nz, bx, by, bz):
    def f(x, n, b):
        Hn=Hermite.basis(n)
        h=Hn(x/b)*np.exp(-(x/b)**2/2)
        norm=simpson(h**2, x)
        return h/np.sqrt(norm)
    def psi(x, y, z):
        fx=f(x, nx, bx)[:, None, None]
        fy=f(y, ny, by)[None, :, None]
        fz=f(z, nz, bz)[None, None, :]
        return fx*fy*fz
    return psi

def HO3D(nx, ny, nz, bx, by, bz, dV):

    def f1d(x, n, b):
        Hn = Hermite.basis(n)
        h  = Hn(x/b) * np.exp(-(x/b)**2/2)
        return h

    def psi(x, y, z):
        fx = f1d(x, nx, bx)[:, None, None]
        fy = f1d(y, ny, by)[None, :, None]
        fz = f1d(z, nz, bz)[None, None, :]

        psi_raw = fx * fy * fz

        norm=np.sum(psi_raw**2)*dV
        return psi_raw/np.sqrt(norm)
    return psi

def EigenF_Well(nx, ny, nz, ax, ay, az, dV):
    def f1D(x, n, a):
        if n%2!=0:
            return np.sin(n*np.pi*x/(2*a))
        else: return np.cos(n*np.pi*x/(2*a))
    
    def psi(x, y, z):
        fx=f1D(x, nx, ax)[:, None, None]
        fy=f1D(y, ny, ay)[None, :, None]
        fz=f1D(z, nz, az)[None, None, :]

        psi_raw=fx*fy*fz
        norm=np.sum(psi_raw**2)*dV
        if norm <= 1e-20:
            print(norm)
            raise ValueError("Normalizzazione zero → griglia o L errati")

    
        return psi_raw/np.sqrt(norm)
    return psi

#sistemare efficienza delle chiamate