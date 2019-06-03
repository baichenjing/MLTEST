def test_mtb():
    import numpy as np
    import matplotlibExample as plt
    X=np.linspace(-np.pi,np.pi,256,endpoint=True)
    C,S=np.cos(X),np.sin(X)

    plt.plot(X,C)
    plt.plot(X,S)
    plt.show()

def test_func():
    import numpy as np
    a=np.array([[1,2,3],[2,3,4]],ndmin=2)
    dt=np.dtype([('age',np.int8)])
    na=np.array([(10,),(20,),(30,)],dtype=dt)
    print(na)
    student=np.dtype([('name',np.str),('age',np.int),('marks',np.float)])
    a=np.array([("abc",21,50),('xyz',18,75)],dtype=student)
    print(a)
    dt=np.dtype(['age',np.int8])
    a=np.array([(10,),(20,),(30,)],dtype=dt)
    print(a['age'])


if __name__ == '__main__':
    test_func()

