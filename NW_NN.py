import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
plt.close("all")

np.random.seed(1)


n=100 #number of samples
L=2 #Lipschitz constant
noise_var=0.1 #noise variance
h=0.06 #kernel bandwith parameter
delta=0.001 #probability 1-\delta


limits=np.array([[0],[3]]) #domain

d=limits.shape[1] #explanatory data dimmension R^d

grid_step=np.array([0.05]) # grid step for every explanatory sub data d x 1

fun2d = lambda x: np.exp(-x)*np.sin(4*x)


fun_main=fun2d

Kernel_fun=lambda x :1*(np.linalg.norm(x,axis=0)<1).astype(float)

def generate_data(limits,n,noise_var,fun):
    '''
    takes limits (domain), n (number of samples), noise variance and function and calculates uniformly sampled
    explanatory data and corresponding noisy values
    '''
    d=limits.shape[1]
    explanatory_data=np.random.uniform(limits[0,:],limits[1,:],(n,d)).T #explanatory data matrix d x n
    residuals = np.random.normal(0,noise_var, size=[n]) #noise value n x 1
    value=fun(*np.split(explanatory_data,d))+residuals
    return explanatory_data,value

def calculate_bounds(mesh,kappa,h,delta,L,noise_var):
    alpha = np.zeros(mesh.shape[1:])
    alpha[kappa <= 1] = np.sqrt(np.log(np.sqrt(2)/delta))
    alpha[kappa > 1] = np.sqrt(kappa[kappa > 1]*np.log(np.sqrt(1 + kappa[kappa > 1])/delta))
    NW_bound= L*h + 2*noise_var*alpha/kappa
   # print(str(2*noise_var*alpha/kappa)+" : "+str(L*h))
    return NW_bound

def NW(data,value,mesh,h,delta,L,noise_var, ker_NW):

    m_tmp=np.array([(mesh[i].reshape(mesh[i].shape+(1,))-data[i].reshape((1,)*mesh[i].ndim+(len(data[i]),)))/h
                    for i in range(mesh.shape[0])]) # stucked arrays of diferences grid - explanatory data
    kernel_values=ker_NW(m_tmp)
    num = np.sum(np.multiply(value, kernel_values), axis=mesh.ndim-1)
    kappa = np.sum(kernel_values, axis=(mesh.ndim-1))
    NW_bound=calculate_bounds(mesh,kappa,h,delta,L,noise_var)
    return num, kappa, NW_bound



mesh= np.array(np.meshgrid(*[np.arange(l[0], l[1]+grid_step[i],grid_step[i]) for i,l in enumerate(limits.T) ]))


ed,v=generate_data(limits,n,noise_var,fun_main)


num,kappa,NW_bound=NW(ed,v,mesh,h,0.01,L,noise_var,Kernel_fun)

NW_est=num/kappa

plt.figure(1)
plt.plot(mesh[0],fun2d(mesh[0]),color="C0")
plt.plot(mesh[0],NW_est,color="C1")
plt.fill_between(mesh[0],NW_est-NW_bound,NW_est+NW_bound,alpha=0.4,color="C0")
plt.fill_between(mesh[0],NW_est-L*h,NW_est+L*h,alpha=0.4,color="C2")
plt.scatter(ed,v)



