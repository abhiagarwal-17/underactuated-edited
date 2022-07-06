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
                         plot_sublevelset_expression)

from pydrake.solvers.mosek import MosekSolver


# finding the bound
negative_infinity = float('-inf')
coeff = 1e-2
positive_infinity = float('inf')

def find_implicit_lambdas(x, xd, V, Vdot, g, x_0, u):
    prog = MathematicalProgram()
    prog.AddIndeterminates(x)
    prog.AddIndeterminates(xd)
    rho = prog.NewContinuousVariables(1, 'rho')[0]
    
    y = np.hstack([x, xd])
    
    l_deg = math.ceil(Vdot.TotalDegree()/2) * 2
    l_deg = 4
    
    lambda_ = prog.NewSosPolynomial(Variables(y), l_deg)[0]
    lambda_g = []
    
    for i in range(g.size): 
        gi_deg = max(l_deg + Vdot.TotalDegree() - g[i].TotalDegree(), 0)
        lambda_g.append(prog.NewFreePolynomial(Variables(y), gi_deg))
        
    lambda_g = np.array(lambda_g)    
    
    s_deg = max(l_deg + Vdot.TotalDegree() - 2, 0)
    lambda_s = prog.NewFreePolynomial(Variables(y), s_deg)
    lambda_s2 = prog.NewFreePolynomial(Variables(y), s_deg)
    
    trig = Polynomial(x[0]**2 + x[1]**2 - 1)    
    trig2 = Polynomial(x[2]**2 + x[3]**2 - 1)

    prog.AddSosConstraint(Polynomial((x-x_0).dot(x-x_0))*(V - rho) -lambda_*Vdot \
                          + lambda_g.dot(g) + lambda_s*trig + lambda_s2*trig2)
    
#     prog.AddSosConstraint(Vdot + Polynomial((x-x_0).dot(x-x_0)) + lambda_*(V + 362) + \
#                           lambda_g.dot(g) + lambda_s*trig + lambda_s2*trig2)
    
    prog.AddLinearCost(-rho)
    prog.AddLinearConstraint(rho, 0, positive_infinity)
        
    solver = MosekSolver()
    print("HERE")
    result = solver.Solve(prog)
    
    k = result.get_solver_details().solution_status
                
    print("Lambda solution status: " + str(k))
    
    assert result.is_success()
    
    lambda_g_results = []
    
    for i in range(g.size):
        lambda_g_results.append(result.GetSolution(lambda_g[i]).RemoveTermsWithSmallCoefficients(coeff))
    
    return result.GetSolution(lambda_).RemoveTermsWithSmallCoefficients(coeff), np.array(lambda_g_results), \
result.GetSolution(lambda_s).RemoveTermsWithSmallCoefficients(coeff)\
, result.GetSolution(rho)


def problem_solver_implicit(V_degree, G):
    prog = MathematicalProgram()
    s1 = prog.NewIndeterminates(1, "s1")
    c1 = prog.NewIndeterminates(1, "c1")
    s2 = prog.NewIndeterminates(1, "s2")
    c2 = prog.NewIndeterminates(1, "c2")
    td1 = prog.NewIndeterminates(1, "\dot{t1}")
    td2 = prog.NewIndeterminates(1, "\dot{t2}")
    
    x = np.hstack([s1, c1, s2, c2, td1, td2])
    x_0 = np.array([0, 1, 0, 1, 0, 0])
    
    xd = prog.NewIndeterminates(6, 'xd')

    u = -G@(x - x_0)
            
    g0 = xd[0] - c1[0]*td1[0]
    g1 = xd[1] + s1[0]*td1[0]
    g2 = xd[2] - c2[0]*td2[0]
    g3 = xd[3] + s2[0]*td2[0]
    g4 = I1*xd[4] + I2*xd[4] + m2*l1**2*xd[4] * 2*m2*l1*lc2*c2[0]*xd[4] + \
            I2*xd[5] + m2*l1*lc2*c2[0]*xd[5] - 2*m2*l1*lc2*s2[0]*td1[0]*td2[0] - \
                m2*l1*lc2*s2[0]*td2[0]**2 - \
                    m1*gravity*lc1*s1[0] + m2*gravity*(-l1*s1[0] + lc2*(-s1[0]*c2[0] - c1[0]*s2[0]))
    g5 = I2*xd[4] + m2*l1*lc2*c2[0]*xd[4] + I2*xd[5] + m2*l1*lc2*s2[0]*td1[0]**2 + \
            m2*gravity*lc2*(-s1[0]*c2[0] - c1[0]*s2[0]) - u
        
    g = np.array([g0, g1, g2, g3, g4, g5])
    
    
    V = 1.6216589046514689*td2[0]**2 -4.3177741128219491*td1[0] * td2[0] + 16.84029326925371*td1[0]**2 - \
    11.754113053392956*s2[0] * td2[0] + 146.19694313913635*s2[0] * td1[0] + 327.48490678102974*(2-2*c2[0]) - \
    -28.958835082382414*s1[0] * td2[0] + 340.32411119816487*s1[0] * td1[0] + \
    1515.5320202469929*s1[0] * s2[0] + \
    1760.7194856732497*(2-2*c1[0])
    
    Vdot = V.Jacobian(x).dot(xd)
    
    V = Polynomial(V)
    Vdot = Polynomial(Vdot)
    
    g_poly = []
    for i in range(g.size):
        g_poly.append(Polynomial(g[i]))
        
    g_poly = np.array(g_poly)
    
    for i in range(1):
        lambda_, lambda_g, lambda_s, rho = find_implicit_lambdas(x, xd, V, Vdot, g_poly, x_0, Polynomial(u))
        
        print(rho)
        
        Vdot = V.Jacobian(x).dot(xd)
     
        display(Markdown("$ V(x)="+ToLatex(V.ToExpression(), 9)+"$"))
        
    return V, Q

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
    V_degree = 2
    K_casted = np.array([-278.44223126,    0.        , -112.29125985,    0.        , -119.72457377,  -56.82824017])
    V, Q = problem_solver_implicit(V_degree, K_casted)