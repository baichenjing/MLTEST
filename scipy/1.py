import numpy as np
import scipy.linalg as sl
def caculate_norm():
    m,n=10,9
    A=np.random.random((m,n))
    b=np.random.random(m)
    x=sl.lstsq(A,b)[0]
    print(np.linalg.norm(x))

def caculate_optimization():
     import scipy.optimize as op
     f=lambda x:-(np.sin(x-2)**2)*(np.exp(-(x**2)))
     ans=op.minimize_scalar(f)
     ans['fun']=-ans['fun']
     print(ans)

def caculate_dist():
    import scipy.spatial.distance as dst
    cities=np.random.random((5,2))
    ans=dst.cdist(cities,cities)
    print(ans)


