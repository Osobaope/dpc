import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
#import espresso
from qpsolvers import solve_qp
import quadprog
import qpsolvers
import time
import scipy.io
import cvxpy as cp
from cvxopt import matrix, solvers
from scipy.linalg import eig


def Transform_f(f, lambda_y, p, N_p):
    # Compute f_transpose from OCP first, then transpose to get F for quadprog
    # f_transpose = [f, lambda_y * ones(1, p * N_p)]

    f_transpose = np.hstack((f, lambda_y * np.ones((1, p * N_p))))
    Ft = f_transpose.T
    return Ft

def blockHankelMat(dataMat, depth):
    # Uses 'data' to construct a hankel matrix of depth 'depth'.
    # 'data' has to be specified as a matrix, where each column represents a data sample.
    vecSize, nSamples = dataMat.shape
    if nSamples < depth:
        raise ValueError("Too few data samples to construct a hankel matrix of the given depth.")
    #dataMat = dataMat.reshape(-1, 1) this is the wrong approach, this affects the ud matrix in that
    #it first reshapes the first row under one another, then reshaping the second row
    dataMat = dataMat.flatten(order='F')
    dataMat = dataMat.reshape(-1, 1)
    print(dataMat.shape)
    #print(dataMat[100:110,:])
    HH = np.zeros((vecSize * depth, nSamples - depth + 1))
    for i in range(nSamples - depth + 1):
        HH[:, i] = dataMat[(i * vecSize):((i + depth) * vecSize), 0]
    return HH

def Transform_Equality(HuP, HyP, Y_0, U_0, p, N_p):
    # Equality constraint for the deterministic case was formerly:
    # [HuP]       [U_0]
    # [HyP] * a = [Y_0]
    # but now, (HyP + sigma) * a_tilde = Y_0 is given implicitly in the inequality constraints,
    # due to substitution of this term in the inequality constraint
    # therefore only [HuP] * a_tilde = [U_0] holds as an equality constraint.

    # Compute Gt_eq_u
    rows_G_u = HuP.shape[0]
    Gt_eq_u = np.hstack((HuP, np.zeros((rows_G_u, p * N_p))))

    # Compute Gt_eq_y
    rows_G_y = HyP.shape[0]
    Gt_eq_y = np.hstack((HyP, (-1) * np.ones((rows_G_y, p * N_p))))

    # Combine Gt_eq
    Gt_eq = np.vstack((Gt_eq_u, Gt_eq_y))

    # Allocate e_eq
    et_eq = np.vstack((U_0, Y_0))

    return Gt_eq, et_eq

def Transform_Inequality(G, e, HyP, Y_0, p, N_p):
    # Compute Gt
    # Extend columns of G matrix
    row_G = G.shape[0]
        
    print("debug trafo ineq 0")
    Gt = np.hstack((G, np.zeros((row_G, p * N_p))))
    print("debug trafo ineq 1")
    # Compute Gt for second to last row and last row
    G_inequality_constraint_1 = np.hstack((HyP, (-1) * np.ones((p * N_p, p * N_p))))
        
    print("debug trafo ineq 2")
    G_inequality_constraint_2 = np.hstack(((-1) * HyP, (-1) * np.ones((p * N_p, p * N_p))))

    # Update Gt with the second to last row
    Gt = np.vstack((Gt, G_inequality_constraint_1))
    print("debug trafo ineq 3")
    # Update Gt with the last row
    Gt = np.vstack((Gt, G_inequality_constraint_2))
    print("debug trafo ineq 4")
    # Compute et
    #et = np.vstack((e, Y_0,np.multiply(Y_0, -1)))
    print(f"e:{e.shape}")
    print(f"Y_0:{Y_0.shape}")
    et = np.vstack((e, Y_0))
      
    print("debug trafo ineq 5")
    et = np.vstack((et,np.multiply(Y_0, -1)))
    print("debug trafo ineq 6")
    return Gt, et

def TransformOCP(H, f, lambda_2, lambda_y, p, N_p):
    # Compute Ht
    L = H.shape[1]
    H = H + (2 * lambda_2 * np.eye(L))  # Update H
    row_H, _ = H.shape
    # Determine the number of rows in H
    row_H, col_H = H.shape

    # Create a block of zeros
    zeros_block = np.ones((row_H, p * N_p))

    # Concatenate H with the zeros block to the right
    Ht = np.hstack((H, zeros_block))

    #Ht = np.vstack(np.hstack((H, np.zeros((row_H, p * N_p)))), np.zeros((p * N_p, row_H + p * N_p)))
    col_Ht = Ht.shape[1]
    #Create a block of zeros
    zeros_block = np.ones((p * N_p, col_Ht))

    # Stack Ht and the zeros block vertically
    Ht = np.vstack((Ht, zeros_block))

    # Compute Ft
    f_transpose = np.hstack((f, lambda_y * np.ones((1, p * N_p))))
    Ft = f_transpose.T

    return Ht, Ft


# Use the Python module object to store controller state
this = sys.modules[__name__]


# define system parameters of Coffee machine
N_p = 10 # number of looks into the past(data)
m, p = 2, 1 # inputs, outputs
N = 40 # prediction horizon

this.m = m
this.N = N
this.p = p
this.N_p = N_p

# set reference temperature
setpoint = 95
y_ref = setpoint #where do we get setpoint from?
print(f"y_set was set to: {y_ref}")
#y_ref = 95
#setpoint = y_ref

# Import coffee data csv file
#espresso.WWWROOT/newCoffeeData.csv
#newCoffeeData = espresso.WWWROOT + "/newCoffeeData.csv"

coffee_data = pd.read_csv("newCoffeeData.csv")

# choose data around operating point, skip data at coffee machine warm-up
coffee_data = coffee_data.iloc[0:4655, 2:6].values

# create array with column 1: brew group temperature
# column 2: input 1, heating element one
# column 3: input 2, heating element two
coffee_data = coffee_data[:, [1, 2, 3]]
coffee_row, coffee_col = coffee_data.shape

# Augment input data (u1,u2) into one full column matrix
ud = np.zeros((m, coffee_row))
ud[0, :] = coffee_data[:, 1]  # fill u1
ud[1, :] = coffee_data[:, 2]  # fill u2
yd = coffee_data[:, 0].reshape(1, coffee_row)

# Implementation of controller
L = (m + N_p) * (p + N)
this.L = L
data_length = L + N + N_p - 1

# Slice input and output matrices to get a square D_tilde
ud = ud[:, 0:data_length]
yd = yd[:, 0:data_length]
ud = np.array(ud, dtype=np.float128)
yd = np.array(yd, dtype=np.float128)

depth = N + N_p

#debug ud
print(ud)
# Construct (block-)hankel matrices and partition them for future/past
Hu = blockHankelMat(ud, depth)  # Hankel matrix of inputs
Hu = np.array(Hu, dtype=np.float128)
HuP = Hu[0:m * N_p, :]  # Block matrix for "past" inputs
HuF = Hu[m * N_p:, :]  # Block matrix for "future" inputs
Hy = blockHankelMat(yd, depth)  # Hankel matrix of outputs
Hy = np.array(Hy, dtype=np.float128)
HyP = Hy[0:N_p * p, :]  # Block matrix for "past" outputs
HyF = Hy[N_p * p:, :]  # Block matrix for "future" outputs

D_Ntilde = np.vstack((Hu, Hy))

'''print(f" rANK DnTILDE {np.linalg.matrix_rank(D_Ntilde)}")
print(f"(N + N_p) * (m + p): {(N + N_p) * (m + p)}")
# Check GPE condition
if np.linalg.matrix_rank(D_Ntilde) == (N + N_p) * (m + p):
    print('GPE condition satisfied.')
else:
    print(f'rank(D_Ntilde) = {np.linalg.matrix_rank(D_Ntilde)} < {(N + N_p) * (m + p)} = (N + p) * m + p')
    raise ValueError('GPE condition not satisfied.')'''

# Specify DPC design parameters
Qtilde = np.eye(p)
R = np.eye(m)
uLb, uUb = 0, 100
yLb, yUb = 0, 100

# Compute augmented matrices
q_tilde_row, q_tilde_col = HyF.shape
QQtilde = np.kron(np.eye(q_tilde_row), Qtilde)
RR = np.kron(np.eye(q_tilde_row), R)

ULb = np.kron(np.ones((q_tilde_row * m, 1)), uLb)
UUb = np.kron(np.ones((q_tilde_row * m, 1)), uUb)
YLb = np.kron(np.ones((q_tilde_row, 1)), yLb)
YUb = np.kron(np.ones((q_tilde_row, 1)), yUb)

#compute Y_ref for N-time-steps
dim_a = QQtilde.shape[1]  # Dimension of optimization variable z = a
Y_ref_N = np.full((dim_a, 1), y_ref)

# set tuning parameters
lambda_2 = 5
lambda_y = 100000


#initialize step
coffee_data = pd.read_csv("coffee_data.csv")

# choose data around operating point, skip data at coffee machine warm-up
coffee_data = coffee_data.iloc[0:6655, 2:6].values

# create array with column 1: brew group temperature
# column 2: input 1, heating element one
# column 3: input 2, heating element two
coffee_data = coffee_data[:, [1, 2, 3]]

start_data_pt = 4689
#N_p = 3  # Assuming N_p is defined somewhere in your code
#coffee_data = np.array(...)  # Define your coffee_data array

# Extract the 'kick_starter' portion of 'coffee_data'
kick_starter = coffee_data[start_data_pt - N_p:start_data_pt, :]

# Extract 'y_pre' and 'u_pre'
y_pre = kick_starter[:, 0]
u_pre = kick_starter[:, 1:3]

# Reshape 'u_pre' to 'U_0'
U_0 = u_pre.reshape(-1, 1)

# Set 'Y_0' to 'y_pre'
Y_0 = y_pre.reshape(-1,1)

# Extract the last values of 'Y_0' and 'U_0' as 'y_0' and 'u_0'
y_0 = Y_0[-1]
u_0 = U_0[-1]


# initialize u-k-minus-1 with zeros
this.u_k_minus_one = np.zeros(m*N)
this.u_k_minus_one = this.u_k_minus_one.reshape(m*N,1)
#initialize Y_0 and U_0 with zeros
#U_0 = np.zeros(m*N_p)
#Y_0 = np.zeros(p*N_p)
#U_0 = U_0.reshape(m*N_p, 1)
#Y_0 = Y_0.reshape(p*N_p, 1)
print(f"U_O: {U_0.shape}")
this.U_0 = U_0
this.Y_0 = Y_0



# Compute time-invariant QP parameters
H = 2 * (HyF.T @ QQtilde @ HyF + HuF.T @ RR @ HuF)
H_np = np.array(H, dtype=np.float128)
f = (-2 * Y_ref_N.T @ QQtilde @ HyF - 2 * this.u_k_minus_one.T @ RR @ HuF).astype(np.float256)
G = np.vstack((HuF, -HuF, HyF, -HyF))
e = np.vstack((UUb, -ULb, YUb, -YLb))

print(f"Size of time-invariant-H: {H.shape}")
print(f"Size of time-invariant-f: {f.shape}")
print(f"Size of time-invariant-G: {G.shape}")
print(f"Size of time-invariant-e: {e.shape}")

#if np.any(H):
    #raise ValueError("H-Vector contains only null values. Change variables to global variables")

this.G = G
this.e = e
# Compute transformed matrices
# Assuming you have defined the TransformOCP, Transform_Inequality, and Transform_Equality functions.

[Ht, Ft] = TransformOCP(H, f, lambda_2, lambda_y, p, N_p)
[Gt, et] = Transform_Inequality(G, e, HyP, Y_0, p, N_p)
[Gt_eq, et_eq] = Transform_Equality(HuP, HyP, Y_0, U_0, p, N_p)

print(f"Size of Ht: {Ht.shape}")
print(f"Size of Ft: {Ft.shape}")
print(f"Size of Gt: {Gt.shape}")
print(f"Size of et: {et.shape}")
print(f"Size of Gt_eq: {Gt_eq.shape}")
print(f"Size of et_eq: {et_eq.shape}")

this.Ht = Ht
this.Ft = Ft
this.Gt = Gt
this.et = et
this.Gt_eq = Gt_eq
this.et_eq = et_eq

#update Y_0
t_brew_group = 90
'''this.Y_0 = this.Y_0[this.p:] #slice Y_0 and pop out first p elements
for i in range(p):
    print(f"i am p: {p}")
    print(t_brew_group)
    #this.Y_0.append(t_brew_group) #append boiler pmw command for current m inputs
    this.Y_0 = np.append(this.Y_0, t_brew_group)
    this.Y_0 = this.Y_0.reshape(p*N_p, 1)
    print(this.Y_0.shape)
    print("kurz vor start 2")
print("kurz vor start 2")'''
#update f from OCP
f = -2 * Y_ref_N.T @ QQtilde @ HyF - 2 * this.u_k_minus_one.T @ RR @ HuF
Ft = Transform_f(f, lambda_y, p, N_p)
print("kurz vor start 3")
#update Gt_eq and et_eq
[this.Gt, this.et] = Transform_Inequality(this.G, this.e, HyP, this.Y_0, p, N_p)
print("kurz vor start 3")
[this.Gt_eq, this.et_eq] = Transform_Equality(HuP, HyP, this.Y_0, this.U_0, p, N_p)

print("kurz vor start 1")

# solve updated QP
print(this.Gt.shape)
print(this.et.shape)
print(this.Ft.shape)
print(this.Ht.shape)
print(this.Gt_eq.shape)
print(this.et_eq.shape)

# Load the variable from the MATLAB .mat file
matlab_data = scipy.io.loadmat('workspace_variables.mat')
#matlab_variable = matlab_data['variable_name']

# Load the variables from the MATLAB .mat file
matlab_data = scipy.io.loadmat('workspace_variables.mat')

# Create a dictionary to store Python variables
python_variables = {
    'Hu': Hu,
    'Hu': Hu,
    # Add more variables as needed
}

# Load corresponding MATLAB variables
matlab_variable1 = matlab_data['Hu']
matlab_variable2 = matlab_data['Hu']
# Load more variables as needed

# Compare variables
for name, py_var in python_variables.items():
    matlab_var = matlab_data[name]  # Assumes MATLAB variable names match Python variable names
    #comparison_result = (matlab_var == py_var).all()
    #print(f"{name}: Variables are equal: {comparison_result}")


#end_time = time.time()
#execution_time = end_time - start_time
#print(f"exec time: {execution_time}")
#this.Ht.reshape(622,)

#this.Ft = this.Ft.reshape((622,))
#this.et = this.et.reshape((320,))
#this.et_eq = this.et_eq.reshape((30,))
#zTildeOpt= quadprog.solve_qp(this.Ht, this.Ft)
#zTildeOpt = solve_qp(this.Ht, this.Ft, this.Gt, this.et,this.Gt_eq, this.et_eq, solver="quadprog")


# Define the size of the problem
n = this.Ht.shape[1]  # Assuming Ht is a 2D matrix
#n = n.reshape(-1,1)

# Define the optimization variables
x = cp.Variable((n,1))
#x = x.reshape(-1,1)

# Define the objective function (quadratic cost)
objective = cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(this.Ht)) + this.Ft.T @ x)

# Define inequality constraints (Gx <= et)
inequality_constraints = [this.Gt @ x <= this.et]

# Define equality constraints (G_eq x == et_eq)
equality_constraints = [this.Gt_eq @ x == this.et_eq]

# Create the optimization problem with objective and constraints
problem = cp.Problem(objective, inequality_constraints + equality_constraints)


zTildeOpt = solve_qp(this.Ht, this.Ft, this.Gt, this.et, this.Gt_eq, this.et_eq,solver="quadprog")

start_time = time.time()
# Solve the problem
problem.solve()

end_time = time.time()

# Check the status of the solver
if problem.status == cp.OPTIMAL:
    print("Optimal value =", problem.value)
    print("Optimal solution =", x.value)
else:
    print("Solver did not converge.")

# Calculate the elapsed time
elapsed_time = end_time - start_time

zTildeOpt = x.value
#zTildeOpt = solvers.qp(this.Ht, this.Ft, this.Gt, this.et,this.Gt_eq, this.et_eq)
print(f"nach solving: {zTildeOpt}")
zOpt = zTildeOpt[:L]
#if(str(zOpt) == "None"):
#        raise Exception("No QP solution")

# Get optimal Input sequence
UOpt = HuF@zOpt
uPred = UOpt.reshape(this.m, -1)
u_k_minus_one = uPred.flatten()  # update u_k_minus_1
this.u_k_minus_one = u_k_minus_one
u1 = uPred[:, -1][0]
u2 = uPred[:, -1][1]
#print(f"u1: {u1}")
#print(f"u2: {u2}")


plt.figure()
plt.title('Brew group temperature trajectory')
plt.plot(Y_ref_N)  # Plot reference temperature

y_Pred = np.reshape(np.dot(HyF, zOpt), (p, -1))
print(y_Pred)  # See initial condition
#plt.step(np.arange(len(y_Pred[0,:])+1).reshape(1,-1), np.insert(y_Pred, 0, y_0, axis=1), where='post')
plt.step(np.arange(len(y_Pred[0])), y_Pred[0], where='post')
#plt.step(np.arange(len(uPred[1])+1), np.concatenate((uPred[1], [uPred[1][-1]])), where='post')
plt.step(np.arange(len(uPred[1])+1), np.concatenate((uPred[1], [uPred[1][-1]])), where='post')
plt.step(np.arange(len(uPred[0])+1), np.concatenate((uPred[0], [uPred[0][-1]])), where='post')
plt.show()