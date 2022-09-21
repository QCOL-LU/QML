from linear_system_solvers import *


#from ipm.problem_generators.convert_stad_problem_to_conl_lo import *
#from ipm.problem_generators.convert_conl_problem_to_stad_lo import *
# from ipm.print_methods.print_IPM import *

import numpy as np
from time import time
from copy import deepcopy
import sys






def IR_QLSS (X, y, Params):
    
    nabla             = 1                    # Scaling factor 
    IRprecision         = Params.IR_Precision
    #Params.LS_Noise					=  1e-1
    M=np.matmul(X.T,X)
    matrix=M/np.linalg.norm(M)
    con=[]
    vector=np.dot(X.T,y)/np.linalg.norm(M)
    iteration         = 1
    d=len(matrix)
    z   =   np.zeros(d)
    r   =   vector
    res=[]
    while (np.linalg.norm(r)>IRprecision):
        #noise=np.random.rand(d)
        #noise=(QLSAprecision/(np.linalg.norm(noise)))*noise
        nabla = 1/(np.linalg.norm(r))
        MM=matrix
        ss=nabla*r
        t1=time()
        c, norm_of_residual, is_sign_changed 	= linear_system_solver(MM, ss, Params)
        t2=time()
        #c = np.linalg.solve(matrix, nabla*r)+noise
        con.append(t2-t1)
        z = z + (1/nabla)*c
        lam = np.dot(matrix, z)
        r = vector - lam
        res.append(np.linalg.norm(r))
        
        iteration=iteration+1

    return (z, res, iteration, con)

def AD1_IR_QLSS (X, y, Params):
    C=10
    nabla             = 1                    # Scaling factor 
    IRprecision         = Params.IR_Precision
    #Params.LS_Noise					=  1e-1
    M=np.matmul(X.T,X)
    matrix=M/np.linalg.norm(M)
    con=[]
    vector=np.dot(X.T,y)/np.linalg.norm(M)
    iteration         = 1
    d=len(matrix)
    z   =   np.zeros(d)
    r   =   vector
    res=[]
    while (np.linalg.norm(r)>IRprecision):
        #noise=np.random.rand(d)
        #noise=(QLSAprecision/(np.linalg.norm(noise)))*noise
        nabla = 1/(np.linalg.norm(r))
        regParam=C/iteration
        MM=matrix+(regParam)*np.eye(d)
        ss=nabla*r
        t1=time()
        c, norm_of_residual, is_sign_changed 	= linear_system_solver(MM, ss, Params)
        t2=time()
        #c = np.linalg.solve(matrix+(regParam)*np.eye(d), nabla*r)+noise
        con.append(t2-t1)
        z = z + (1/nabla)*c
        lam = np.dot(matrix, z)
        r = vector - lam
        res.append(np.linalg.norm(r))
        iteration=iteration+1

    return (z, res, iteration, con)
def AD2_IR_QLSS (X, y, Params):
    C=0.1
    nabla             = 1                    # Scaling factor 
    IRprecision         = Params.IR_Precision
    #Params.LS_Noise					=  1e-1
    M=np.matmul(X.T,X)
    matrix=M/np.linalg.norm(M)
    con=[]
    vector=np.dot(X.T,y)/np.linalg.norm(M)
    iteration         = 1
    d=len(matrix)
    z   =   np.zeros(d)
    r   =   vector
    res=[]
    while (np.linalg.norm(r)>IRprecision):
        #noise=np.random.rand(d)
        #noise=(QLSAprecision/(np.linalg.norm(noise)))*noise
        nabla = 1/(np.linalg.norm(r))
        regParam=C/(2**iteration)
        MM=matrix+(regParam)*np.eye(d)
        ss=nabla*r
        t1=time()
        c, norm_of_residual, is_sign_changed 	= linear_system_solver(MM, ss, Params)
        t2=time()
        #c = np.linalg.solve(matrix+(regParam)*np.eye(d), nabla*r)+noise
        con.append(t2-t1)
        z = z + (1/nabla)*c
        lam = np.dot(matrix, z)
        r = vector - lam
        res.append(np.linalg.norm(r))
        iteration=iteration+1

    return (z, res, iteration, con)


