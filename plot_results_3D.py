import numpy as np
from consts3D import nuclearNu_x, nuclearNu_y, nuclearNu_z, b_x, b_y, b_z
from eigenfunctions3D import HO3D, EigenF_Well
from orbitals3D import uncoupledHObasis, OrbitalSet
from optimizer3D import getSampleDensity
from utility3D import loadData, findPlateu
from plot_methods_3D import plot_density_3D, plot_potential_3D, plot_orbital_3D, plot_potential, plot_potential_3D_no_mod_grid

fvar=input("Hai modificato il nome dei file di slavataggio? (se si mettere y/n): ")
if fvar!='y':
    raise InterruptedError("operazione annullata dall'utente.")


def density_square_well(n_orb, dV, basis=uncoupledHObasis()):
    basis=OrbitalSet([ c for c in basis[:n_orb] ])
    n_part=basis.count_particles()
    wf=[]
    for j, oo in enumerate(basis):
        wf.append( EigenF_Well(oo.nx+2, oo.ny, oo.nz, 6, 6, 6, dV) )
    def rho(x, y, z):
        arr=np.array( [basis[j].occupation*wf[j](x, y, z)**2 for j in range(n_orb)] )
        return np.sum(arr, axis=0)
    return rho

results=loadData("test_3D_data/data_2_particles/data")
grid=results['grid']
phi=results['u']
lagrange=results['lagrange']


A=2 #Z=2 and N=0
norb=1
rho_function=getSampleDensity(n_orb=norb, dV=0.001, basis=uncoupledHObasis())#density_square_well(norb, dV=0.001, basis=uncoupledHObasis())#
nx=len( grid[0] ); ny=len( grid[1] ); nz=len( grid[2] )
npunti=nx*ny*nz
X, Y, Z=np.meshgrid(grid[0], grid[1], grid[2], indexing='ij')
rho_target=rho_function(grid[0], grid[1], grid[2])

#ricostruzione variabili
phi=np.array(phi)
rho_calc=2*np.sum(phi**2, axis=0)
V_analitico=nuclearNu_x(A)*X**2+nuclearNu_y(A)*Y**2+nuclearNu_z(A+80)*Z**2
V_ipopt=( np.reshape( lagrange[:npunti], (nx, ny, nz) ) /rho_target)
#j=findPlateu(V_flat, 5, 0.1)
#cost=V[j]
#V-=cost
V_ipopt-=V_ipopt[nx//2, ny//2, nz//2]


#states=[(2,0,0)]#, (1,0,0)]#, (0,1,0), (0,0,1)]
#phi_ex=[EigenF_Well(nx, ny, nz, 6, 6, 6, 0.001)(grid[0], grid[1], grid[2]) for nx, ny, nz in states]

states=[(0,0,0)]#, (1,0,0)]#, (0,1,0), (0,0,1)]
phi_ex=[HO3D(nx, ny, nz, b_x(A), b_y(A), b_z(A+80), 0.001)(grid[0], grid[1], grid[2]) for nx, ny, nz in states]
name_list=[f"3Dorbitale_{i+1}_2_particles_+80_retry.png" for i in range(len(states))]

V_analitico_1D=V_analitico[:, ny//2, nz//2]
V_calc_1D=V_ipopt[:, ny//2, nz//2]

#plot
plot_potential_3D_no_mod_grid(V_analitico, V_ipopt, grid, "3D_Potential_2_particles_+80_no_mod_grid_retry.png")
plot_density_3D(rho_target, rho_calc, grid, "3D_Density_2_particles_+80_retry.png")
plot_potential_3D(V_analitico, V_ipopt, grid, "3D_Potential_2_particles_+80_retry.png")
plot_potential(grid[0], V_analitico_1D, V_calc_1D, "1DPot_2_particles_+80_retry.png")
for i, (wf_ex, name) in enumerate( zip(phi_ex, name_list) ):
    plot_orbital_3D(wf_ex, phi[i], grid, f"{name}")