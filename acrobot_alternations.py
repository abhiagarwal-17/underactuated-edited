from pydrake.examples.acrobot import (AcrobotGeometry, AcrobotInput,
                                      AcrobotPlant, AcrobotState, AcrobotParams)
from pydrake.all import Linearize, LinearQuadraticRegulator, SymbolicVectorSystem, Variable, Saturation, \
WrapToSystem, Simulator, Polynomial

from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogVectorOutput

from IPython.display import SVG, display
import pydot
import numpy as np
import math
import control
import pydrake

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pydrake.forwarddiff as pf
import time

from IPython.display import display, Math, Markdown
from pydrake.examples.pendulum import PendulumPlant
from pydrake.examples.acrobot import AcrobotPlant
from pydrake.all import MultibodyPlant, Parser, SinCos, MakeVectorVariable, ToLatex, Substitute, Box
from underactuated import FindResource
from underactuated.scenarios import AddShape

from pydrake.all import (Jacobian, MathematicalProgram, Polynomial,
                         RegionOfAttraction, RegionOfAttractionOptions, Solve,
                         SymbolicVectorSystem, ToLatex, Variable, Variables,
                         plot_sublevelset_expression,  SolverOptions, CommonSolverOption, SolverOptions)

from pydrake.solvers.mosek import MosekSolver

def find_lambdas(V, Vdot, x): 
    print('finding lambda')
    prog = MathematicalProgram()
    prog.AddIndeterminates(x)

    rho = prog.NewContinuousVariables(1, 'rho')[0]

    l_deg = Vdot.TotalDegree()
    l_deg = 10

    lambda_, lambda_Q = prog.NewSosPolynomial(Variables(x), l_deg) 

    prog.AddSosConstraint(Polynomial(x.dot(x))*(V-rho) - lambda_*Vdot)

    prog.AddLinearCost(-rho)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintFileName, './check_solver_acrobot.txt')

    solver = MosekSolver()

    result = solver.Solve(prog, solver_options=options)

    assert result.is_success(), str(result.get_solver_details().solution_status)

    lambda_sol = result.GetSolution(lambda_).RemoveTermsWithSmallCoefficients(1e-5)
    rho_sol = result.GetSolution(rho)

    return lambda_sol, rho_sol

def find_V(lambda_, M, Mdot, f, rho, x): 
    print('Finding V')
    prog = MathematicalProgram()
    prog.AddIndeterminates(x)

    S = prog.NewSymmetricContinuousVariables(4, 'S')

    V = x.T@M.T@S@M@x 
    Vdot = f.T@S@M@x + x.T@Mdot.T@S@M@x + x.T@M.T@S@Mdot@x + x.T@M.T@S@f
    Vdot = Vdot[0]

    prog.AddSosConstraint(x.dot(x)*(V-rho) - lambda_.ToExpression()*Vdot)
    prog.AddCost(np.trace(S))
    prog.AddLinearConstraint(V.Substitute({x[0]:0, x[1]:0, x[2]:0, x[3]:0}) == 0)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintFileName, './check_solver_acrobot.txt')

    solver = MosekSolver()
    result = solver.Solve(prog, solver_options=options)

    assert result.is_success(), str(result.get_solver_details().solution_status)

    S_sol = result.GetSolution(S)

    return S_sol

def search_function(x, K, B, X, M, Mdot, P, f, max_itr, conv_tol): 
    print('Entering Search')
    #add the controller
    K2 = B.T@X@M
    u = -K@x 
    print(u)
    f[3] = f[3] + u[0]*(1+p1**2)*(1+p2**2)

    V = x.T@M.T@P@M@x
    V = Polynomial(V)
    Vdot = f.T@P@M@x + x.T@Mdot.T@P@M@x + x.T@M.T@P@Mdot@x + x.T@M.T@P@f
    Vdot = Polynomial(Vdot[0])

    S_old = P
    Ss = []
    rhos = []
    for i in range(max_itr): 
        lambda_, rho = find_lambdas(V, Vdot, x)
        S = find_V(lambda_, M, Mdot, f, rho, x)

        V = x.T@M.T@S@M@x
        V = Polynomial(V)
        Vdot = f.T@S@M@x + x.T@Mdot.T@S@M@x + x.T@M.T@S@Mdot@x + x.T@M.T@S@f
        Vdot = Polynomial(Vdot[0])

        if np.sum(np.abs((S_old-S)/S_old) < conv_tol) == 16:
            break

        print(i)
        print(S)  

        S_old = S
        Ss.append(S)
        rhos.append(rho)

    return Ss, rhos



    print('Reached here succeffuly. Suck it!')

if __name__ == '__main__': 
    p = AcrobotParams()

    m1 = p.m1()
    m2 = p.m2()
    l1 = p.l1()
    lc1 = p.lc1()
    lc2 = p.lc2()
    Ic1 = p.Ic1()
    Ic2 = p.Ic2()
    b1 = p.b1()
    b2 = p.b2()
    gravity = p.gravity()

    I1 = Ic1 + m1*lc1**2
    I2 = Ic2 + m2*lc2**2

    prog_create = MathematicalProgram()
    x = prog_create.NewIndeterminates(4, 'X')
    p1 = x[0]
    p2 = x[1]
    t1d = x[2]
    t2d = x[3]

    M = np.array([[1, 0, 0, 0], 
                  [0, 1, 0, 0], 
                  [0, 0, (1+p1**2)*(1+p2**2)*(I1+I2+m2*l1**2)+2*m2*l1*lc2*(1+p1**2)*(1-p2**2), \
                                           (1+p1**2)*(1+p2**2)*I2+m2*l1*lc2*(1-p2**2)*(1+p1**2)], 
                  [0, 0, I2*(1+p1**2)*(1+p2**2)+m2*l1*lc2*(1+p1**2)*(1-p2**2), I2*(1+p1**2)*(1+p2**2)]])

    f3 = -2*m2*l1*lc2*2*p2*t1d*t2d*(1+p1**2) - m2*l1*lc2*2*p2*t2d**2*(1+p1**2) - m1*gravity*lc1*2*p1*(1+p2**2) \
            +m2*gravity*(-l1*2*p1*(1+p2**2) + lc2*(-2*p1*(1-p2**2) - 2*p2*(1-p1**2))) + b1*t1d*(1+p1**2)*(1+p2**2)

    f4 = m2*l1*lc2*2*p2*t1d**2*(1+p1**2) + m2*gravity*lc2*(-2*p1*(1-p2**2) - 2*p2*(1-p1**2)) \
                                                                                + b2*t2d*(1+p1**2)*(1+p2**2)

    f = np.array([[0.5*(1+p1**2)*t1d], 
                  [0.5*(1+p2**2)*t2d], 
                  [-f3], 
                  [-f4]])

    p1d = 0.5*(1+p1**2)*t1d
    p2d = 0.5*(1+p2**2)*t2d

    Mdot = np.array([[0, 0, 0, 0], 
                     [0, 0, 0, 0], 
                     [0, 0, (2*p2*p2d+2*p1*p1d+2*p1*p2**2*p1d+2*p2*p1**2*p2d)*(I1+I2+m2*l1**2)\
                                     +2*m2*l1*lc2*(-2*p2*p2d+2*p1*p1d-2*p1*p2**2*p1d-2*p2*p1**2*p2d), \
                                         (2*p2*p2d+2*p1*p1d+2*p1*p2**2*p1d+2*p2*p1**2*p2d)*I2\
                                                +m2*l1*lc1*(-2*p2*p2d+2*p1*p1d-2*p1*p2**2*p1d-2*p2*p1**2*p2d)], \
                     [0, 0, I2*(2*p2*p2d+2*p1*p1d+2*p1*p2**2*p1d+2*p2*p1**2*p2d)\
                                                 +m2*l1*lc2*(-2*p2*p2d+2*p1*p1d-2*p1*p2**2*p1d-2*p2*p1**2*p2d), \
                                                     I2*(2*p2*p2d+2*p1*p1d+2*p1*p2**2*p1d+2*p2*p1**2*p2d)]])

    env = {p1:0, p2:0, t1d:0, t2d:0}
    A_general = []
    for exp in f: 
        exp_curr = exp[0].Jacobian(x)
        A_general.append(exp_curr)
    A_general = np.array(A_general)  

    A = np.zeros_like(A_general)
    for i, row in enumerate(A_general): 
        for j, elem in enumerate(row): 
            A[i, j] = elem.Evaluate(env)
            
    A = np.array(A, dtype=float)


    E = np.zeros_like(M)
    for i, row in enumerate(M): 
        for j, elem in enumerate(row): 
            if type(elem) == pydrake.symbolic.Expression:
                E[i, j] = elem.Evaluate(env)
            else: 
                E[i, j] = M[i, j]
            
    E = np.array(E, dtype=float)

    B = np.array([[0], 
                  [0], 
                  [0], 
                  [1]])

    Q = np.diag([5, 5, 1, 1])
    R = [1]

    X, L, K = control.care(A, B, Q, R, E=E)

    A_cloop = A-B@K
    Q_lyap = np.eye(4)

    P = control.lyap(A_cloop.T, Q_lyap, E = E.T)

    max_itr = 1
    conv_tol = 1e-2

    Ss, rhos = search_function(x, K, B, X, M, Mdot, P, f, max_itr, conv_tol)

    Ss = np.array(Ss)
    rhos = np.array(rhos)

    np.save('./acrobot_alternations_S.npy', Ss)
    np.save('./acrobot_alternations_rho.npy')

    print(P)
   