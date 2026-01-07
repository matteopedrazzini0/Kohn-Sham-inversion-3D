class Orbital(object):
    def __init__(self, nx, ny, nz):
        self.nx=nx
        self.ny=ny
        self.nz=nz
        self.N=self.nx+self.ny+self.nz
        self.name=self.getName()
        self.degeneracy=2*(self.N+1)*(self.N+2)/2
        self.occupation=2

    def getName(self):   #migliora
        ss=""
        if self.N==0: ss="(0, 0, 0)"
        if self.N==1: ss="nx, ny, nz=1"
        if self.N==2: ss="3s"
        if self.N==3: ss="4s"
        name=str(self.N)+ss
        return name
    
    def __str__(self):
        return self.getName()
    
class OrbitalSet(object):
    def __init__(self, orb_list):
        self.orb_list=orb_list
        self.index=0
        self.n_particles=self.count_particles()


    def __iter__(self):
        return self


    def __next__(self):
        try: result=self.orb_list[self.index]
        except IndexError: raise StopIteration
        self.index+=1
        return result


    def __getitem__(self,k):
        return self.orb_list[k]


    def __len__(self):
        return len(self.orb_list)


    def count_particles(self):
        s=0
        for oo in self.orb_list:
            s+=oo.occupation
        return s
    

    """
    reset counter
    """
    def reset(self):
        self.index=0
        self.n_particles=self.count_particles()


class uncoupledHObasis(OrbitalSet):
    def __init__(self):
        orbs=[  Orbital(0, 0, 0),   Orbital(1, 0, 0),   Orbital(0, 1, 0),   Orbital(0, 0, 1),   Orbital(2, 0, 0),   Orbital(0, 2, 0),   Orbital(0, 0, 2),   Orbital(1, 1, 0),   
                Orbital(1, 0, 1),   Orbital(0, 1, 1),   Orbital(3, 0, 0),   Orbital(0, 3, 0),   Orbital(0, 0, 3),   Orbital(2, 1, 0),   Orbital(2, 0, 1),   Orbital(1, 2, 0) ]
        # call parent constructor
        super().__init__(orbs)

def getOrbitalSet(n_particles, basis=uncoupledHObasis()):
    ll=[]
    nn=n_particles
    for oo in basis:
        nn-=oo.occupation
        if nn>=0:
            ll.append(oo)
            if nn==0: break
        else:
            print ("Most orbitals are required")
            break
    return OrbitalSet(ll)
