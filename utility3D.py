import pickle  # for input/output
import numpy as np
from scipy import interpolate as interp

def saveData(filename, data):
    "Save an (almost) arbitrary python object to disc."
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def loadData(filename):
    "Load and return data saved to disc with the function `save_data`."
    with open(filename, 'rb') as f:
        data=pickle.load(f)
    return data


def quickload(file="densiies/"):
    file=open(file)
    ff=file.readlines()    
    file.close()
    x=[]; y=[]; z=[]; dp=[]
    for ll in ff:
        if str(ll).startswith("#"): pass
        else:
            ll=[ float(x) for x in ll.split() ]
            x.append( ll[0] )
            y.append( ll[1] )
            z.append( ll[2] )
            dp.append( ll[3] )
    x=np.array(x); y=np.array(y); z=np.array(z); dp=np.array(dp)
    return x, y, z, dp


def read(filename):
    lists=[]
    file=open(filename)
    lines=file.readlines()
    lines=[ll.strip() for ll in lines if len(ll.strip())>0]  # remove empty lines
    file.close()
    for line in lines:
        if str(line).startswith('#'): continue
        else:
            # parse line
            line=[ x for x in line.split() ]
            # lists 
            while len(lists)<len(line):
                lists.append( list() )
            # load data
            for j in range( len(lists) ):
                lists[j].append( line[j] )
    # convert to np.array
    for j in range(len(lists)):
        if is_float_try(lists[j][0]):
            lists[j]=np.array( lists[j], dtype=float )
    return lists


def is_float_try(stri):
    try:
        float(stri)
        return True
    except ValueError:
        return False
    

def findPlateu(v, n=3, tol=0.05, st=0):    #da mettere nel caso in cui si lavora con oggetto 3D
    v=np.array(v)
    for j in range(st, v.shape[0]-1 ):
        flag=True
        for k in range(0,n+1):
            #print(j,j+k)
            ind=np.min((v.shape[0]-1, j+k))
            if np.abs(v[j]-v[ind])>tol:
                flag=False
                break
        if flag==True:
            return j
    
    # choose between the two:
        
    # print("No plateau satisfying the inserted conditions has been found, returning the ten last index")
    # return -10
    
    # or
    
    return findPlateu(v, n=n, tol=tol*1.2, st=st)