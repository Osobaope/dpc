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


coffee_data = pd.read_csv("newCoffeeData.csv")

# choose data around operating point, skip data at coffee machine warm-up
coffee_data = coffee_data.iloc[2:4655, 3:6].values
#coffee_data = coffee_data[:, [1, 2, 3]]
y = 2
