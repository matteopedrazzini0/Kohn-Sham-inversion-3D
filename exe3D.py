from consts3D import nuclearNu_x, nuclearNu_y, nuclearNu_z, b_x, b_y, b_z
from orbitals3D import uncoupledHObasis, OrbitalSet
from eigenfunctions3D import EigenF_Well
from utility3D import loadData, findPlateu
from optimizer3D import iKS_problem, getSampleDensity
from plot_methods_3D import plot_density_3D, plot_potential_3D, plot_potential, plot_orbital_3D

import numpy as np
import os
os.environ["OMP_NUM_THREADS"]="8"
os.environ["MKL_NUM_THREADS"]="8"
os.environ["OPENBLAS_NUM_THREADS"]="8"
import time


start=time.time()

print("***************************************************************************** \n Ipopt legend for the more important parameters \n")
print("inf_pr: primal infeasibilty, represent the primal violation of contraints, have to be 0")
print("inf_du: dueal infeasibilty, represent the stationarity violation of the lagrangian gradient, " \
"how the lagrangian is far form the minimum")
print("lg(mu): logarithm of the barrier parameter, linked to the interior point method, have to be 0")
print("||d||: norm of step, represent the norm of the newton step.\n       big->solver is moving with big step (far from convergence) \n" \
"       small->solver is moving with little step (near to convergence)")
"""
fvar=input("Hai modificato il nome dei file di slavataggio? (se si mettere y/n): ")
if fvar!='y':
    raise InterruptedError("operazione annullata dall'utente.")
"""


Za=2; N=0
A=Za+N
norb=1
basis=uncoupledHObasis()
#generate grid and density
ub=[2, 2, 2]; lb=[-2, -2, -2]
dx=0.1; dy=dx; dz=dx; dV=dx*dy*dz
nx=int( (ub[0]-lb[0])/dx )+1
ny=int( (ub[1]-lb[1])/dy )+1
nz=int( (ub[2]-lb[2])/dz )+1
print("Numero di punti per asse: ", nx)
npunti=nx*ny*nz
print("Numero di punti totale della griglia: ", npunti)
x=np.linspace(lb[0], ub[0], nx)
y=np.linspace(lb[1], ub[1], ny)
z=np.linspace(lb[2], ub[2], nz)
grid=(x, y, z)
X, Y, Z=np.meshgrid(x, y, z, indexing='ij')

def density_square_well(n_orb, dV, basis=uncoupledHObasis()):
    basis=OrbitalSet([ c for c in basis[:n_orb] ])
    n_part=basis.count_particles()
    wf=[]
    for j, oo in enumerate(basis):
        wf.append( EigenF_Well(oo.nx+2, oo.ny, oo.nz, ub[0]-lb[0], ub[1]-lb[0], ub[2]-lb[2], dV) )
    def rho(x, y, z):
        arr=np.array( [basis[j].occupation*wf[j](x, y, z)**2 for j in range(n_orb)] )
        return np.sum(arr, axis=0)
    return rho


rho_function=getSampleDensity(n_orb=norb, dV=dx**3, basis=basis)#density_square_well(norb, dV=dV, basis=uncoupledHObasis())#
rho_target=rho_function(x, y, z)
print(rho_target.shape)
data=[x, y, z, rho_target]

#instance to ipopt problem
prob=iKS_problem(Z=Za, N=0, lb=lb, ub=ub, data=data, hx=dx, basis=uncoupledHObasis(), reg_parameter=0.1,
                 output_folder="data_2_particles", exact_hess=False)
results, info=prob.solve()

end=time.time()
print(f"Tempo di esecuzione: {end - start:.4f} secondi")
if end-start<=60:
    pass
elif end-start<=3600:
    print(f"Tempo di esecuzione: {(end - start)/60.:.2f} minuti")
else:
    print(f"Tempo di esecuzione: {(end - start)/3600.:.2f} ore")





"""  vecchio caricamneto, si può eliminare
results=loadData("/Users/matteopedrazzini/Desktop/SIM/3D_CODE/test_3D_data/data_2_particles/data")
phi=results['u']
lagrange=results['lagrange']

#ricostruzione variabili
phi=np.array(phi)
rho_calc=2*np.sum(phi**2, axis=0)
V_analitico=nuclearNu_x(A)*X**2+nuclearNu_y(A)*Y**2+nuclearNu_z(A)*Z**2
V_ipopt=(np.reshape(lagrange[:npunti], (nx, ny, nz))/rho_target)
#V=np.reshape(V_ipopt, (nx, ny, nz))
#V=V/rho_target
V_ipopt-=V_ipopt[nx//2, ny//2, nz//2]
#j=findPlateu(V, 5, 0.1)
#cost=V[j]
#V-=cost
#V=np.reshape(V, (nx, ny, nz))

V_analitico_1D=V_analitico[:, ny//2, nz//2]
V_calc_1D=V_ipopt[:, ny//2, nz//2]

#plot
plot_density_3D(rho_target, rho_calc, grid, "/Users/matteopedrazzini/Desktop/SIM/3D_CODE/test_3D/3D_density_controllo.png")
plot_potential_3D(V_analitico, V_ipopt, grid, "/Users/matteopedrazzini/Desktop/SIM/3D_CODE/test_3D/3D_potential_controllo.png")
plot_potential(x, V_analitico_1D, V_calc_1D, "/Users/matteopedrazzini/Desktop/SIM/3D_CODE/test_3D/1D_potential_slice_controllo.png")
"""