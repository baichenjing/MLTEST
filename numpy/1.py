import re

import numpy
import numpy.matlib
import numpy as np
dt=np.dtype([('age',np.int8)])
#a=np.array([(10,),(20,),(30,)],dtype=dt)
# student=np.dtype([('name','S20'),('age','i1')])
# #a=np.array([('abc',21),('xyz',18)],dtype=student)
# a=np.array([[1,2,3],[4,5,6]])
#x=np.empty([3,2],dtype=int)
x=np.zeros(5)
y=np.zeros((5,),dtype=np.int)
z=np.zeros((2,2),dtype=[('x','i4'),('y','i4')])
z=numpy.ones([2,2],dtype=int,order='C')

s=b'Hello,World'
a=np.frombuffer(s,dtype='S1')

list=range(5)
it=iter(list)


a=np.arange(8)
b=a.reshape(4,2)
c= b.flatten(order='C')
print(c)
a,b=13,17
print(bin(a),bin(b))
print(np.bitwise_and(13,17))
print(np.bitwise_or(13,17))
print(np.invert(np.array([13],dtype=np.uint8)))
print(np.binary_repr(13,width=8))
print(np.binary_repr(242,width=8))
print(np.char.add(['hello'],[' xyz']))
a=np.array([0,30,45,60,90])
print(a)
print(np.sin(a*np.pi/180))
print(np.cos(a*np.pi/180))
print(np.tan(a*np.pi/180))
print(np.tan(a*np.pi/180))
a=np.arange(9,dtype=np.float).reshape(3,3)
print(np.ptp(a,axis=1))
print(np.average((1,2,3,4)))
print(numpy.matlib.zeros((2,2)))
a=np.array([[1,2],[3,4]])
print(np.dot(a,b))
np.save('outfile.npy',a)
print(np.__version__)
np.show_config()
Z=np.zeros(10)
print(Z)
Z=np.zeros((10,10))
print(Z.size * Z.itemsize)
np.info(np.add)
Z=np.zeros(10)
Z[4]=1
print(Z)
Z=np.arange(10,50)
print(Z)
Z=np.arange(50)
z=Z[::-1]
print(z)
Z=np.arange(9).reshape(3,3)
print(Z)
nz=np.nonzero([1,2,0,0,4,0])
print(nz)
Z=np.eye(3)
print(z)
Z=np.random.random((3,3,3))
print(Z)
Z=np.random.random((10,10))
Zmax,Zmin=Z.max(),Z.min()
print(Z.max,Z.min)
Z=np.random.random(30)
mean=Z.mean()
print(mean)
Z=np.ones((10,10))
Z[1:-1,1:-1]=0
print(Z)
Z=np.ones((10,10))
Z=np.pad(Z,pad_width=1,mode='constant',constant_values=0)
print(Z)
0*np.nan
np.nan==np.nan
np.inf>np.nan
np.nan-np.nan
0.3==3*0.1
Z=np.diag([1,2,3,4],k=-1)
print(Z)
Z=np.zeros((8,8),dtype=int)
Z[1::2,::2]=1
Z[::2,1::2]=1
print(Z)
print(np.unravel_index(100,(6,7,8)))

def checkprime(x):
    if x<=1:
        return False
    prime=True
    for i in range(2,1+x/2):
        if x%i==0:
            prime=False
            break
        return prime

def run():
    nsteps=1000
    draws=np.random.randint(-1,2,size=nsteps)
    walk=draws.cumsum()
    from matplotlib import pyplot as plt
    plt.plot(walk)
    plt.show()
def run2():
    from matplotlib import pyplot as plt
    nsteps=1000
    draws=np.random.randint(-1,2,size=(2,nsteps))
    walks=draws.cumsum(1)
    plt.plot(walks[0,:],walks[1,:])
    plt.show()
def CompIntegralbyladder(func,x0,x1):
    wholearea=0
    step=0.1
    for i in np.arange(x0,x1,step):
        wholearea+=(func(i)+func(i+step))*step/2
    return wholearea
def caclulateExp():
    import numpy as np
    x=np.linspace(-5,5,num=100)
    y=np.exp(x)
    from matplotlib import pyplot as plt
    plt.plot(x,y)
    plt.show()
if __name__ == '__main__':
    #checkprime(10)
    #run2()
    caclulateExp()




