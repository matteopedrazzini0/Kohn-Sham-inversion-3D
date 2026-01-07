import numpy as np
import numexpr as ne
print(ne.get_num_threads())
ne.set_num_threads(8)
import os

from cyipopt import Problem as ipopt_Problem
from findiff import FinDiff
from numba import njit, prange

from consts3D import a, b_x, b_y, b_z, nuclearNu_x, nuclearNu_y, nuclearNu_z
from eigenfunctions3D import HO3D, HO3D_1,EigenF_Well
from orbitals3D import uncoupledHObasis, OrbitalSet, getOrbitalSet
from utility3D import saveData, loadData
from plot_methods_3D import plot_density_3D, plot_potential_3D


class iKS_problem(ipopt_Problem):
    """
    Classe usata per rappresentare il problema inverso (KS) in caso nucleare"
    
    Schema:

    obiettivo: il funzionale da minimizzare (energia cinetica)
    
    gradiente: gradiente dell'obiettivo

    constraints:
            constraint functions [density and overlap integral (orthonormality)] 
    jacobianstructure:
        non-zero elements of the jacobian (sparse matrix)
    jacobian:
        jacobian matrix of the constraints
    hessianstructure:
        non-zero elements of the hessian (sparse matrix)
    hessian:
        hessian matrix of the full objective function
    .
    .
    .
    Parameters
    ----------
    Z : int
        Number of protons
    N : int
        Number of neutrons (default 0)
    rho : function
        target density function (default None)
    lb : float
        lower bound of the mesh (default 0.1)
    ub : float
        upper bound of the mesh (default 10.)
    h : float
        step (default 0.1)
    n_type : string
        run calculations for either protons ('p') or neutrons ('n') (default 'p')
    data : list
        if rho is None, generate target density by interpolating data[0] (r) and data[1] (rho) with a spline (default [])
    basis : orbital.OrbitalSet
        basis, described by quantum numbers nlj or nl (default orbital.ShellModelBasis, i.e. nlj)
    max_iter : int
        maximum number of iteration of ipopt
    rel_tol : float
        relative tolerance on the value of the objective function 
    constr_viol : float
        max. absolute tolerance on the value of the constraints
    output_folder : str
        name of the folder inside Results where the output is saved (default Output)
    exact_hess : bool
        use the provided exact Hessian or, if False, an automatic approximate one (default True)
    com_correction: bool
        use center of mass correction (A-1)/A (default True)
    debug : str
        (not implemnted yet)
        
    """

    def __init__(self, Z, N, lb, ub, data, rho=None, exp=True, hx=0.1, hy=None, hz=None, n_type="p", 
                 basis=None, reg_parameter=0.1, max_iter=2000, rel_tol=1e-3, constr_viol=1e-3,
                 output_folder="Output", exact_hess=True, com_correction=False, debug='n'):
        
        #info
        self.Z=Z
        self.N=N
        self.A=Z+N
        self.n_type=n_type if (n_type=='p' or n_type=='n') else "p"
        self.n_particles=Z if self.n_type=='p' else self.A
        self.beta=reg_parameter

        #coefficient
        self.alpha=a

        #orbitals
        self.basis=basis if basis is not None else uncoupledHObasis()
        self.orbital_set=getOrbitalSet(self.n_particles, self.basis)
        self.n_orbitals=len(self.orbital_set)
        print("Numero di orbitali usati nel problema: ", self.n_orbitals)
        self.pairs=self.getPairs()  #pairs of orthonality constraints

        #density
        self.data=data
        self.rho=rho if rho is not None else self.getRhoFromData(data[0], data[1], data[2], data[3])
        self.exponential=exp

        #initialize grid
        self.set_grid(lb, ub, hx, hy, hz)

        self.exact_hess=exact_hess
        self.debug=True if debug=='y' else False

        #output directory
        self.output_folder="test_3D_data/"+output_folder if len(output_folder)>0 else ""
        if len(self.output_folder)>0 and not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        #output file
        #scaled orbitals
        self.f_file=f"{self.output_folder}/f.dat"
        #true orbitals
        self.phi_file=f"{self.output_folder}/phi.dat"
        #lagrangian_multipliers, ci scrivo dentro in solve()
        self.pot_file=f"{self.output_folder}/potential.dat"
        self.epsilon_file=f"{self.output_folder}/epsilon.dat"
        self.eigen_file=f"{self.output_folder}/eigenvalues.dat"

        #dictionary of results
        self.results=dict()
        self.datafile=f"{self.output_folder}/data"

        #initialize ipopt
        self.setIpopt(max_iter, rel_tol, constr_viol, exact_hess)
        self.n_runs=0

    """
    configure mesh
    """
    def set_grid(self, lb, ub, hx, hy, hz):
        #grid
        self.lb=lb
        self.ub=ub
        self.hx=hx
        self.hy=hy if hy is not None else self.hx ; self.hz=hz if hz is not None else self.hx
        self.dV=self.hx*self.hy*self.hz

        self.n_points_x=round( (ub[0]-lb[0])/self.hx )+1
        self.n_points_y=round( (ub[1]-lb[1])/self.hy )+1
        self.n_points_z=round( (ub[2]-lb[2])/self.hz )+1
        self.n_points=self.n_points_x*self.n_points_y*self.n_points_z

        x=np.linspace(lb[0], ub[0], self.n_points_x)
        y=np.linspace(lb[1], ub[1], self.n_points_y)
        z=np.linspace(lb[2], ub[2], self.n_points_z)
        self.grid=(x, y, z)
        X, Y, Z=np.meshgrid(x, y, z, indexing='ij')
        #self.meshgrid=(X, Y, Z)

        #differential operator
        self.d_dx=FinDiff(0, self.hx, 1, acc=4)
        self.d_dy=FinDiff(1, self.hy, 1, acc=4)
        self.d_dz=FinDiff(2, self.hz, 1, acc=4)

        #tabulate
        self.tab_rho=self.rho( (X, Y, Z) )
        s=np.sqrt(self.tab_rho)
        self.grad_log_s=self.grad(np.log(s))
        self.norm_grad_log_s=np.einsum('ijkm,ijkm->ijk', self.grad_log_s, self.grad_log_s)

        #number of variables
        self.n_variables=self.n_orbitals*self.n_points
        #number of constraints
        self.n_constr=self.n_points+len(self.pairs)

        #check jacobian
        #j_dense=self.jacobian(self.getStartingPoint())
        #rows, cols=self.jacobianstructure()
        #j_sparse=self.jac(self.getStartingPoint())
        #if np.allclose(j_dense, j_sparse):
        #    print("Tutto ok, i valori corrispondono")
        #else:
        #    print("ATTENZIONE: discrepanza nei valori della Jacobiana!")
        #assert np.allclose(j_dense, j_sparse), "Valori della Jacobiana non corrispondono!"

    def setDensity(self, rho=None, data=[], lb=0, ub=0, stepx=0, stepy=0, stepz=0,output_folder=""):
        """
        In case one need to change density, is not necessary to reallocate the whole class
        """
        if rho is None and len(data)==0: raise ValueError("Error: no correct density provided")
        else: 
            if lb!=0: self.lb=lb
            if ub!=0: self.ub=ub
            if stepx!=0: self.hx=stepx
            if stepy!=0: self.hy=stepy 
            else: self.hy=None
            if stepz!=0: self.hz=stepz
            else: self.hz=None
            self.basis.reset()
            self.__init__(self.Z, self.N, self.lb, self.ub, data, rho, self.hx, self.hy, self.hz, self.n_type, 
                          self.basis, self.beta, self.max_iter, self.rel_tol, self.constr_viol, output_folder)
    #"""        
    def objective(self, x):
        x=np.reshape(x, (self.n_orbitals, self.n_points_x, self.n_points_y, self.n_points_z))
        grad_1x=self.derive(x)
        arr=np.array([ self.integral_k(x[k], grad_1x[k], k) for k in range(self.n_orbitals) ])
        return np.sum(arr)
    

    def integral_k(self, xk, grad_1_x_k, k):
        grad_mod2=np.sum(grad_1_x_k**2, axis=-1)
        dot_grad=np.sum(self.grad_log_s*grad_1_x_k, axis=-1)
        arr=ne.evaluate( "occ*( (alpha*rho+beta)*grad_mod2 + alpha*rho*( xk**2*norm_grad_log_s + 2*xk*dot_grad) )",
                        {
                            "occ": self.orbital_set[k].occupation,
                            "alpha": self.alpha,
                            "beta": self.beta,
                            "rho": self.tab_rho,
                            "grad_mod2": grad_mod2,
                            "norm_grad_log_s": self.norm_grad_log_s,
                            "dot_grad": dot_grad,
                            "xk": xk,
                        }
                        )
        #arr=self.orbital_set[k].occupation*( (self.alpha*self.tab_rho+self.beta)*grad_mod2+
        #    self.alpha*self.tab_rho*( xk**2*self.norm_grad_log_s+2*xk*dot_grad ) )
        return np.sum(arr*self.dV)
    
    
    def gradient(self, x):
        x=np.reshape(x, (self.n_orbitals, self.n_points_x, self.n_points_y, self.n_points_z))
        grad=np.zeros_like(x)
        grad1_x=self.derive(x)
        for k in range(self.n_orbitals):
            xk=x[k]
            deg=self.orbital_set[k].occupation
            derivand=self.alpha*x[k][..., None]*self.tab_rho[..., None]*self.grad_log_s+(self.alpha*self.tab_rho[..., None]+self.beta)*grad1_x[k]
            der_x=self.d_dx( derivand[..., 0] )
            der_y=self.d_dy( derivand[..., 1] )
            der_z=self.d_dz( derivand[..., 2] )
            scalar_dlog_s_df=np.sum( self.grad_log_s*grad1_x[k, :, :, :], axis=-1)
            grad[k]=ne.evaluate( "2*deg*( alpha*rho*( xk*norm_grad_log_s + scalar_dlog_s_df) - (der_x+der_y+der_z) )", 
                                {
                                    "deg": deg,
                                    "alpha": self.alpha,
                                    "rho": self.tab_rho,
                                    "xk": xk,
                                    "norm_grad_log_s": self.norm_grad_log_s,
                                    "scalar_dlog_s_df": scalar_dlog_s_df,
                                    "der_x": der_x,
                                    "der_y": der_y,
                                    "der_z": der_z,
                                })
            #grad[k, :, :, :]=2*deg*( self.alpha*self.tab_rho*( x[k]*self.norm_grad_log_s+scalar_dlog_s_df)-(der_x+der_y+der_z) )
        return grad.ravel()
    
    
    def constraints(self, x):
        x=np.reshape(x, (self.n_orbitals, self.n_points_x, self.n_points_y, self.n_points_z))
        dens=self.tab_rho*np.sum( [self.orbital_set[n].occupation*x[n]**2 for n in range(self.n_orbitals) ], axis=0)-self.tab_rho
        orto=np.zeros( len(self.pairs) )
        #<i|j>=\int dx (rho(x) f_i(x)f_j(x))
        for n, (i, j) in enumerate(self.pairs):
            f_i=x[i]; f_j=x[j]
            orto[n]=np.sum(self.tab_rho*f_i*f_j)*self.dV-1 if (i==j) else np.sum(self.tab_rho*f_i*f_j)*self.dV
        return np.concatenate( (dens.ravel(), orto) )
    
    """
    def jacobianstructure(self):
        jac=np.zeros( (self.n_constr, self.n_orbitals, self.n_points) )
        #density constraints
        for k in range(self.n_orbitals):
            np.fill_diagonal( jac[:self.n_points, k, :], np.ones(self.n_points) )
        #orthonormality constraints
        for k, (i, j) in enumerate(self.pairs):
            counter=self.n_points+k
            jac[counter, i, :]+=np.ones(self.n_points)
            jac[counter, j, :]+=np.ones(self.n_points)
        jac=jac.reshape( (self.n_constr, self.n_variables) )
        return np.nonzero(jac)
    """
    def jacobianstructure(self):
        rows=[]
        cols=[]
        #density
        #for each k there is diagonal block in position base of size npoints
        for k in range(self.n_orbitals):
            base=k*self.n_points
            for r in range(self.n_points):
                rows.append(r)
                cols.append(base+r)
        #orthonormality
        #for each pair there are npoints non-zero values at orbital
        for k, (i, j) in enumerate(self.pairs):
            counter=self.n_points+k
            i_index=i*self.n_points
            j_index=j*self.n_points
            for r in range(self.n_points):
                if i==j:
                    rows.append(counter)
                    cols.append(i_index+r)
                else:
                    rows.append(counter)
                    cols.append(i_index+r)
                    rows.append(counter)
                    cols.append(j_index+r)
        rows=np.array(rows, dtype=int)
        cols=np.array(cols, dtype=int)
        return rows, cols
        
    #"""
    """
    def jacobian(self, x):
        x=np.reshape(x, (self.n_orbitals, self.n_points_x, self.n_points_y, self.n_points_z))
        jac=np.zeros( (self.n_constr, self.n_orbitals, self.n_points) )
        #density constraints
        for k in range(self.n_orbitals):
            deg=self.orbital_set[k].occupation
            np.fill_diagonal( jac[:self.n_points, k, :], 2*deg*x[k] )
        #orthonormality constraints
        x=x.reshape(self.n_orbitals, self.n_points)
        for k, (i, j) in enumerate(self.pairs):
            counter=self.n_points+k
            jac[counter, i, :]=self.tab_rho.ravel()*x[j]*self.dV
            jac[counter, j, :]=self.tab_rho.ravel()*x[i]*self.dV
        #reshape in 2D matrix
        jac=jac.reshape( (self.n_constr, self.n_variables) )
        return jac[self.jacobianstructure()]
    """
    
    def jacobian(self, x):
        x4=np.reshape(x, (self.n_orbitals, self.n_points_x, self.n_points_y, self.n_points_z))
        x=np.reshape(x4, (self.n_orbitals, self.n_points) )

        rows, cols=self.jacobianstructure()
        data=np.zeros( len(rows), dtype=float )
        ptr=0
        #density constraint
        for k in range(self.n_orbitals):
            deg=self.orbital_set[k].occupation
            for r in range(self.n_points):
                data[ptr]=2*deg*x[k, r]
                ptr+=1
        #orthonormality constraints
        rho=self.tab_rho.ravel()
        for kk, (i, j) in enumerate(self.pairs):
            counter=self.n_points+kk
            for r in range(self.n_points):
                if i==j:
                    data[ptr]=2*rho[r]*x[j, r]*self.dV
                    ptr+=1
                else:
                    #value for (i, k)
                    data[ptr]=rho[r]*x[j, r]*self.dV
                    ptr+=1
                    #value for (j, k)
                    data[ptr]=rho[r]*x[i, r]*self.dV
                    ptr+=1
        return data
    #"""

    #aggiungi hessian structure e hessian

    def solve(self):
        st=self.getStartingPoint()
        x, info=super().solve(st)
        print(info['status_msg'])
        self.results=dict()
        keys=['status','x','u','obj','lagrange','summary', 'grid', 'start']
        entries=[ info['status_msg'], x, self.getU(x), info['obj_val'], info['mult_g'], str(self), self.grid, st  ]
        for k, ee in zip(keys, entries):
            self.results[k]=ee
        saveData(self.datafile, self.results)
        with open(self.f_file, 'w') as fx:
            with open(self.phi_file, 'w') as fu:
                with open(self.pot_file, 'w') as fp:
                    u=self.getU(x)
                    x=np.reshape(x, (self.n_orbitals, self.n_points_x, self.n_points_y, self.n_points_z))

                    #plot debug potentials
                    #mult=self.results['lagrange']
                    #pot=( np.reshape( mult[:self.n_points], (self.n_points_x, self.n_points_y, self.n_points_z) ) /self.tab_rho)
                    #pot-=pot[self.n_points_x//2, self.n_points_y//2, self.n_points_z//2]
                    #V_ana=nuclearNu_x(self.A)*X**2+nuclearNu_y(self.A)*Y**2+nuclearNu_z(self.A)*Z**2
                    #plot_potential_3D(V_ana, pot, self.grid, "/Users/matteopedrazzini/Desktop/SIM/3D_CODE/test_3D_out/HO_iso_2_particles/Pot_2_part_in_solve.png")

                    X, Y, Z=np.meshgrid(*self.grid, indexing='ij')
                    Xf, Yf, Zf=X.ravel(), Y.ravel(), Z.ravel()

                    for k in range(self.n_orbitals):
                        for ff in (fx, fu):
                            ff.write(f"# {self.orbital_set[k].getName()}\n")

                        x_flat=x[k].ravel()
                        u_flat=u[k].ravel()

                        for xx, yy, zz, psi, uu in zip(Xf, Yf, Zf, x_flat, u_flat):
                            fx.write(f"{xx:.6e}\t{yy:.6e}\t{zz:.6e}\t{psi:.18E}\n")
                            fu.write(f"{xx:.6e}\t{yy:.6e}\t{zz:.6e}\t{uu:.18E}\n")
                            #fp.write(f"{xx:.6e}\t{yy:.6e}\t{zz:.6e}\t{pp:.10e}\n")
        self.kinetic=float( self.results['obj'] )

        return self.results, info            
    

    def setIpopt(self, max_iter, rel_tol, constr_viol, exact_hess=True):
        self.max_iter=max_iter
        self.rel_tol=rel_tol
        self.constr_viol=constr_viol
        self.exact_hess=exact_hess
        # Calling ipopt_Problem constructor
        ub_x=np.ones(self.n_variables)
        lb_x=-1.*ub_x
        super().__init__(n=self.n_variables, m=self.n_constr, lb=lb_x, ub=ub_x, cl=np.zeros(self.n_constr), cu=np.zeros(self.n_constr))
        self.setSolverOptions(exact_hess)

    """
    Start from HO orbitals
    """
    def getStartingPoint(self):
        st=np.zeros( shape=(self.n_orbitals, self.n_points_x, self.n_points_y, self.n_points_z) )
        self.orbital_set.reset()
        x, y, z=self.grid
        Lx=self.ub[0]-self.lb[0]
        Ly=self.ub[1]-self.lb[1]
        Lz=self.ub[2]-self.lb[2]
        for j, oo in enumerate(self.orbital_set):
            #wf2=EigenF_Well( oo.nx, oo.ny, oo.nz, Lx, Ly, Lz, self.dV)
            wf=HO3D( oo.nx, oo.ny, oo.nz, b_x(self.A), b_y(self.A), b_z(self.A+60), self.dV)
            st[j]=wf(x, y, z)/np.sqrt(self.tab_rho)
        return np.ndarray.flatten(st)


    """
    Solver options: relative tolerance on the objective function; absolute tolerance on the violation of constraints;
    maximum number of iterations
    """
    def setSolverOptions(self, exact_hess):
        # Watch out! Put b (binary) in front of option strings
        self.add_option(b'mu_strategy', b'adaptive')
        self.add_option(b'max_iter', self.max_iter)
        self.add_option('tol', self.rel_tol)
        self.add_option(b'constr_viol_tol', self.constr_viol)
        self.add_option(b"output_file", b"ipopt.out")
        self.add_option(b"hessian_constant", b"no")
        self.add_option(b"hessian_approximation", b"exact") if exact_hess \
        else self.add_option(b'hessian_approximation', b'limited-memory')



    def getRhoFromData(self, x, y, z, rho):
        ff=get_rho_from_data(x, y, z, rho)
        return ff
    

    def grad(self, f):  #of a single scalar function
        fx=self.d_dx(f)
        fy=self.d_dy(f)
        fz=self.d_dz(f)
        return np.stack( (fx, fy, fz), axis=-1 )
    
    def derive(self, x):   #of a sequences of orbitals scalar function
        return np.stack([self.grad( x[k, :, :, :] ) for k in range(self.n_orbitals)], axis=0)
    
    

    def getPairs(self):
        ll=[]
        if self.n_orbitals==1: return []
        for i in range(self.n_orbitals):
            for j in range(i+1):
                ll.append( [i, j] )
        return ll
    

    def getU(self, x):
        x=np.reshape(x, (self.n_orbitals, self.n_points_x, self.n_points_y, self.n_points_z) )
        u=np.zeros_like(x)
        for k in range(self.n_orbitals):
            u[k, :, :, :]=x[k, :, :, :]*np.sqrt(self.tab_rho)
        return u
    

    def getXformU(self, u):
        x=np.zeros_like(u)
        for k in range(self.n_orbitals):
            x[k, :, :, :]=u[k, :, :, :]/self.s
        x=x.flatten()
        return x

#end class
#useful functions
def get_rho_from_data(x, y, z, rho):
    from scipy.interpolate import RegularGridInterpolator
    points=(x, y, z)
    assert( rho.shape==( len(x), len(y), len(z) ) )
    interp_rho=RegularGridInterpolator( points, rho, method='cubic', bounds_error=True)
    ff=lambda p: interp_rho(p)
    return ff


def getSampleDensity(n_orb, dV, basis=uncoupledHObasis()):
    basis=OrbitalSet([ c for c in basis[:n_orb] ])
    n_part=basis.count_particles()
    wf=[]
    for j, oo in enumerate(basis):
        wf.append( HO3D(oo.nx, oo.ny, oo.nz, b_x(n_part), b_y(n_part), b_z(n_part+80), dV) )
    #wf=[HO3D(oo.nx, oo.ny, oo.nz, n_part) for oo in basis]
    def rho(x, y, z):
        arr=np.array( [basis[j].occupation*wf[j](x, y, z)**2 for j in range(n_orb)] )
        return np.sum(arr, axis=0)
    return rho


"""
    #vectorialize
    
    def objective(self, x):
        x=np.reshape(x, (self.n_orbitals, self.n_points_x, self.n_points_y, self.n_points_z))
        grad_1x=self.derive(x)
        grad_mod_2=np.sum(grad_1x**2, axis=-1)
        scalar_dS_dX=np.sum(self.d1_S*grad_1x, axis=-1)
        occ_broad=np.array( [orb.occupation for orb in self.orbital_set] )[:, None, None, None]
        arr=occ_broad*( (self.alpha*self.tab_rho+self.beta)*grad_mod_2+
                 self.alpha*( x**2*self.norm_grad_S+2*x*scalar_dS_dX) )
        return np.sum(arr)*self.dV
    
    #parallelizzata con torch
    def objective(self, x):
        x=torch.tensor(x, dtype=dtype)
        x=x.reshape(self.n_orbitals, self.n_points_x, self.n_points_y, self.n_points_z)
        x=x.to(device)
        tab_rho=torch.tensor(self.tab_rho, device=device, dtype=dtype)
        d1_log_s=torch.tensor(self.d1_log_s, device=device, dtype=dtype)
        norm_grad_log_s=torch.tensor(self.norm_grad_log_s, device=device, dtype=dtype)
        occ_broad=torch.tensor( [orb.occupation for orb in self.orbital_set], device=device).reshape(-1, 1, 1, 1)
        grad_1x=torch.tensor( self.derive( x.cpu().numpy() ), device=device, dtype=dtype)

        grad_mod_2=torch.sum(grad_1x**2, dim=-1)
        scalar_dlogS_dx=torch.sum(d1_log_s*grad_1x, dim=-1)

        arr=occ_broad*( (self.alpha*tab_rho+self.beta)*grad_mod_2+
                 self.alpha*tab_rho*( x**2*norm_grad_log_s+2*x*scalar_dlogS_dx) )
        return torch.sum(arr)*self.dV
    """  


"""

    #directly sparse
    def jacobian(self, x):
        x=np.reshape(x, (self.n_orbitals, self.n_points) )
        rows, cols=self.jacobianstructure()
        data=np.zeros_like(rows)
        #density
        for k in range(self.n_orbitals):
            mask=(rows<self.n_points) & (k==cols//self.n_points)
            data[mask]=2*self.orbital_set[k].occupation*x[k]
        #orthonormality
        for k, (i, j) in enumerate(self.pairs):
            counter=self.n_points+k
            mask_i=(rows==counter)&(i==cols//self.n_points)
            mask_j=(rows==counter)&(j==cols//self.n_points)
            if i==j:
                data[mask_i]=2*self.tab_rho.ravel()*x[j]*self.dV
            else:
                data[mask_i]=self.tab_rho.ravel()*x[j]*self.dV
                data[mask_j]=self.tab_rho.ravel()*x[i]*self.dV

        return data
    """