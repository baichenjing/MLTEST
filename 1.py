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




