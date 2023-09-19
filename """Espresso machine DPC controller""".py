"""Espresso machine DPC controller"""

# Simple DPC controller demo for RCSPresso
# version: 1
# authors: Opeyemi Emmanuel Osoba <opeyemi.osoba@tu-dortmund.de>, Sagar Suman Karna <sagar.karna@tu-dortmund.de>

# date: 04/Aug/2023

# This script demonstrates the implementation of a dataDriven predictive controller program for the
# 'RCSPresso' espresso machine. We have also re-used some parts of the original code for the 
# implementation of the PID-COntroller by the initial author of the controller programs for the
# espresso coffee machine Jan Weigelt <jan.weigelt@tu-dortmund.de> 


# Exception handling:
# - If a syntax error is observed while parsing the script, loading of the
#   controller is stopped and the controller is disabled.
# - If a runtime error is detected running initialize(), further execution of the
#   script is halted and the controller is unloaded.
# - If a runtime error is observed during a controller update, the
#   controller will be unloaded.
# - If the controller is unloaded following an exception, it must be re-enabled
#   manually to continue operation.
#   The exception can be viewed on the "Controllers" page of the web panel.
# - The espresso module performs sanity-checks on all supplied arguments.
#   If an invalid argument is observed, an exception is thrown.
# - Python exceptions can be used to trigger runtime errors and halt the controller.
# - If the controller is halted, the hardware is automatically steered to a safe
#   standby state.
# - If initialize() or run() take longer than 1/f_update to compute, execution
#   is halted and the controller is disabled.

# Device IO is mapped using virtual id constants. Device IO accessible via
# the espresso Python module is listed below.

# Output:
# IO_LAMP_MAINS - green panel indicator
# IO_LAMP_STEAM - red panel indicator
# IO_LAMP_HEATER - amber panel indicator
# IO_PUMP - water pump

# PWM controlled output:
# IO_HEATER_BOILER - boiler heating element
# IO_HEATER_BREWGROUP - brew group heating element

# Input:
# IO_SWITCH_MAINS - main breaker switch
# IO_SWITCH_COFFEE - front panel coffee switch
# IO_SWITCH_STEAM - front panel steam switch
# IO_SWITCH_TANK - tank switch (LOW if tank is removed)
# IO_THERMOSTAT_BOILER - 2-point boiler thermostat with hysteresis
# IO_THERMOSTAT_STEAM - 2-point boiler steam thermostat with hysteresis

import numpy as np
import pandas as pd
import sys
import espresso
from qpsolvers import solve_qp
import qpsolvers
import time




start_time = time.time()

csv_filename = espresso.WWWROOT + "/dpc_data.csv"


def log_init():
    f = open(csv_filename, "a")
    f.write("timestamp [ms],T_set [deg C],T_boiler [deg C], T_brew_group [deg C],pwm_boiler [%],pwm_brew_group[%],pump_on\r\n")
    f.close()

def str_fixed(value):
    return "{:.2f}".format(value)

def log_update(timestamp, setpoint):
    t_boiler = espresso.sensor(espresso.SENSOR_BOILER)
    t_brew_group = espresso.sensor(espresso.SENSOR_BREW_GROUP)
    pwm_boiler = espresso.pwm(espresso.PWM_BOILER)
    pwm_brew_group = espresso.pwm(espresso.PWM_BREW_GROUP)
    pump_on = espresso.io(espresso.IO_PUMP)
    f = open(csv_filename, "a")
    f.write(str(timestamp)+","+str_fixed(setpoint)+","+ str_fixed(t_boiler)+","+str_fixed(t_brew_group)+","+str_fixed(pwm_boiler)+","+str_fixed(pwm_brew_group)+","+str(pump_on)+"\r\n")
    f.close()

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
    dataMat = dataMat.reshape(-1, 1)
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
    H = H + 2 * lambda_2 * np.eye(L)  # Update H
    row_H, _ = H.shape
    Ht = np.vstack((np.hstack((H, np.zeros((row_H, p * N_p)))), np.zeros((p * N_p, row_H + p * N_p))))
    col_Ht = Ht.shape[1]

    # Compute Ft
    f_transpose = np.hstack((f, lambda_y * np.ones((1, p * N_p))))
    Ft = f_transpose.T

    return Ht, Ft


# Use the Python module object to store controller state
this = sys.modules[__name__]


# define system parameters of Coffee machine
N_p = 10 # number of looks into the past(data)
m, p = 2, 1 # inputs, outputs
N = 10 # prediction horizon

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
newCoffeeData = espresso.WWWROOT + "/newCoffeeData.csv"

coffee_data = pd.read_csv(newCoffeeData)

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
depth = N + N_p

#debug ud
print(ud)
# Construct (block-)hankel matrices and partition them for future/past
Hu = blockHankelMat(ud, depth)  # Hankel matrix of inputs
HuP = Hu[0:m * N_p, :]  # Block matrix for "past" inputs
HuF = Hu[m * N_p:, :]  # Block matrix for "future" inputs
Hy = blockHankelMat(yd, depth)  # Hankel matrix of outputs
HyP = Hy[0:N_p * p, :]  # Block matrix for "past" outputs
HyF = Hy[N_p * p:, :]  # Block matrix for "future" outputs
D_Ntilde = np.vstack((Hu, Hy))

# Check GPE condition
if np.linalg.matrix_rank(D_Ntilde) == (N + N_p) * (m + p):
    print('GPE condition satisfied.')
else:
    print(f'rank(D_Ntilde) = {np.linalg.matrix_rank(D_Ntilde)} < {(N + N_p) * (m + p)} = (N + p) * m + p')
    raise ValueError('GPE condition not satisfied.')

# Specify DPC design parameters
Qtilde = np.eye(p)
R = np.eye(m)
uLb, uUb = 0, 100
yLb, yUb = 0, 150

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

# initialize u-k-minus-1 with zeros
this.u_k_minus_one = np.zeros(m*N)
this.u_k_minus_one = this.u_k_minus_one.reshape(m*N,1)
#initialize Y_0 and U_0 with zeros
U_0 = np.zeros(m*N_p)
Y_0 = np.zeros(p*N_p)
U_0 = U_0.reshape(m*N_p, 1)
Y_0 = Y_0.reshape(p*N_p, 1)
print(f"U_O: {U_0.shape}")
this.U_0 = U_0
this.Y_0 = Y_0



# Compute time-invariant QP parameters
H = 2 * (HyF.T @ QQtilde @ HyF + HuF.T @ RR @ HuF)
f = -2 * Y_ref_N.T @ QQtilde @ HyF - 2 * this.u_k_minus_one.T @ RR @ HuF
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



# The initialize() function is called when rcspresso finished parsing the
# controller script. initialize() is always called before the first control
# loop iteration. It is a good place to initialize controller
# and io state.
def initialize():
    print("Initializing DeePC.......")
    # stdout & stderr:
    # - All output to stdout is redirected to the system log, that be can seen on
    #   the web panel's "Log" page or in the code editor
    # - Output to stdout (for example using print()) is logged with event level
    #   "info"
    # - Output to stderr is logged with event level "error"
    # - Writing to stderr does not raise an exception, thus the controller is
    #   not halted.

    # Set controller update rate to 1.0Hz using espresso. 
    # Frequency is given in Hz [1/s] and must be in the interval between
    # [espresso.F_MIN, espresso.F_MAX] [0.1Hz, 20Hz]. The controller
    # is halted if an out-of-range frequency is supplied.
    # espresso.frequency() always returns the current update rate.
    espresso.frequency(0.01)

    # number of samples both controllers have been in the acceptable temperature range
    this.nEpsilon = 0
    this.epsilon = 2
    this.nBrew = 0
    # Turn on "mains" panel indicator (green lamp) using the espresso.io()-function.
    # espresso.io() returns io state as seen during the last io update.
    this.d = 0
    espresso.io(espresso.IO_LAMP_MAINS, True)
    print("debug")

    print(m*N_p)
    
    

    print("Initialize done...............")

    log_init() #initialize Data logger

def update(timestamp, setpoint):

    # The update() function is called on every controller update.
    # The timestamp [integer] parameter holds the current system time in milliseconds.
    # The setpoint [floating point] parameter supplies the target coffee temperature
    # as set in the web panel.
    

    # Read current boiler & brew group temperature. Sensor temperatures
    # are returned in degrees Celsius.
    t_boiler = espresso.sensor(espresso.SENSOR_BOILER)
    t_brew_group = espresso.sensor(espresso.SENSOR_BREW_GROUP)
    
    print("kurz vor start 1")
    #update Y_0 
    this.Y_0 = this.Y_0[this.p:] #slice Y_0 and pop out first p elements
    for i in range(p):
        print(f"i am p: {p}")
        print(t_brew_group)
        #this.Y_0.append(t_brew_group) #append boiler pmw command for current m inputs
        this.Y_0 = np.append(this.Y_0, t_brew_group)
        this.Y_0 = this.Y_0.reshape(p*N_p, 1)
        print(this.Y_0.shape)
        print("kurz vor start 2")
    print("kurz vor start 2")
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
    
    #end_time = time.time()
    #execution_time = end_time - start_time
    #print(f"exec time: {execution_time}")
    #zTildeOpt = solve_qp(this.Ht, this.Ft, this.Gt, this.et, this.Gt_eq, this.et_eq,solver="quadprog")
    #zTildeOpt = solve_qp(this.Ht, this.Ft, this.Gt, this.et,this.Gt_eq, this.et_eq, solver="quadprog")
    print(f"nach solving: {zTildeOpt.shape}")
    zOpt = zTildeOpt[:L]
    if(str(zOpt) == "None"):
            raise Exception("No QP solution")

    # Get optimal Input sequence
    UOpt = HuF@zOpt
    uPred = UOpt.reshape(this.m, -1)
    u_k_minus_one = uPred.flatten()  # update u_k_minus_1
    this.u_k_minus_one = u_k_minus_one
    u1 = uPred[:, -1][0]
    u2 = uPred[:, -1][1]
    #print(f"u1: {u1}")
    #print(f"u2: {u2}")

    # Set both boiler input commands
    # We use espresso.pwm(<channel>, <value>) to assign the on-time per
    # modulation cycle. Duty cycle is given in percent [0~100%].
    # The controller is halted if an invalid PWM channel or violation
    # of duty cycle constraints is observed.
    # espresso.pwm() returns the current pwm duty cycle.
    pmw_boiler = u1
    pwm_brew_group = u2
    espresso.pwm(espresso.PWM_BREW_GROUP, pwm_brew_group)
    espresso.pwm(espresso.PWM_BOILER, pmw_boiler)
    
    #update U_0 and Y_0
    this.U_0 = this.U_0[this.m:] #slice U_0 and pop out first m elements
    for i in range(m):
        this.U_0.append(uPred[:, -1][i]) #append boiler pmw command for current m inputs



    # Control logic remains thesame as Jan's/Marius'
    
    # We increment nEpsilon if the absolute temperature error is below epsilon
    # Kelvin for both control loops
    if(abs(t_boiler - setpoint) < this.epsilon):
        this.nEpsilon += 1
    else:
        this.nEpsilon = 0
        
    # Turn off the heater lamp on the coffee machines front panel if both
    # control loops have been in the acceptable epsilon band
    # for at least 30 sample times
    if(this.nEpsilon > 30):
        if(espresso.io(espresso.IO_LAMP_HEATER)):
            espresso.io(espresso.IO_LAMP_HEATER, False)
    elif not espresso.io(espresso.IO_LAMP_HEATER):
        espresso.io(espresso.IO_LAMP_HEATER, True)
    

    # Check if the "coffee" switch on the front panel has been activated
    # Enable the pump while the switch is on
    # We also turn on the "steam" panel lamp for 30 samples
    if(espresso.io(espresso.IO_SWITCH_TANK)):
        if(not espresso.io(espresso.IO_SWITCH_COFFEE) and espresso.io(espresso.IO_LAMP_STEAM)):
            espresso.io(espresso.IO_LAMP_STEAM, False)
        
        if(espresso.io(espresso.IO_SWITCH_COFFEE)):
            this.nBrew += 1
            if(this.nBrew == 30):
                espresso.io(espresso.IO_LAMP_STEAM, False)
            
            if(not espresso.io(espresso.IO_PUMP)):
                print("Brewing coffee :)")
                espresso.io(espresso.IO_PUMP, True)
                espresso.io(espresso.IO_LAMP_STEAM, True)
        elif(espresso.io(espresso.IO_PUMP)):
                espresso.io(espresso.IO_PUMP, False)
                espresso.io(espresso.IO_LAMP_STEAM, False)
                this.nBrew = 0
        
    elif(espresso.io(espresso.IO_PUMP)):
        espresso.io(espresso.IO_PUMP, False)
        
    else:
        espresso.io(espresso.IO_LAMP_STEAM, not espresso.io(espresso.IO_LAMP_STEAM))



    # data logger
    log_update(timestamp, setpoint)