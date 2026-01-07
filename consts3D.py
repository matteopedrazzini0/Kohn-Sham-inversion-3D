import numpy as np

hbc=197.327053  #MeV*fm
m=938.91869     #MeV

a=hbc**2/(2*m)

def nuclear_omega_x(A):  #in fermi
    return 41./(hbc*np.power(A, 1./3))

def b_x(A):
    return np.sqrt(hbc/(m*nuclear_omega_x(A)))

def nuclearNu_x(A): #1/2mw^2 con w in fermi
    return 0.5*m*nuclear_omega_x(A)**2

def nuclear_omega_y(A):  #in fermi
    return 41./(hbc*np.power(A, 1./3))

def b_y(A):
    return np.sqrt(hbc/(m*nuclear_omega_y(A)))

def nuclearNu_y(A): #1/2mw^2 con w in fermi
    return 0.5*m*nuclear_omega_y(A)**2

def nuclear_omega_z(A):  #in fermi
    return 41./(hbc*np.power(A, 1./3))

def b_z(A):
    return np.sqrt(hbc/(m*nuclear_omega_z(A)))

def nuclearNu_z(A): #1/2mw^2 con w in fermi
    return 0.5*m*nuclear_omega_z(A)**2