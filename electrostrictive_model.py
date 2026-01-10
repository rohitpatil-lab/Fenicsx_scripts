from dolfinx import log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import pyvista
import numpy as np
import ufl
import matplotlib.pyplot as plt

from mpi4py import MPI
from dolfinx import fem, mesh, plot
import basix
from basix import CellType, ElementFamily, LagrangeVariant

# Creating mesh
L=0.1
W=0.02
domain = mesh.create_rectangle(
    comm=MPI.COMM_WORLD, 
    points=((0.0, 0.0), (L, W)), 
    n=[100, 50], 
    cell_type=mesh.CellType.quadrilateral
)

scalar_el = basix.ufl.element(
    "Lagrange", 
    domain.topology.cell_name(), 
    1, 
    basix.LagrangeVariant.equispaced
)

v_el = basix.ufl.blocked_element(scalar_el, shape=(domain.geometry.dim,))

s_el = scalar_el

m_el = basix.ufl.mixed_element([v_el, s_el])

V = fem.functionspace(domain, m_el)

#Functions and Trial/Test Spaces
sol = fem.Function(V)
u, phi = ufl.split(sol)
v, q = ufl.TestFunctions(V)
dw = ufl.TrialFunction(V)
du, dphi = ufl.split(dw)

#Material Parameters
lmbda = fem.Constant(domain, default_scalar_type(1e4))  # Lame lambda
mu = fem.Constant(domain, default_scalar_type(5e3))     # Lame mu
gamma = fem.Constant(domain, default_scalar_type(1e-2))  # Dielectric permittivity
beta1 = fem.Constant(domain, default_scalar_type(1e-3))  # Electrostrictive coeff 1
beta2 = fem.Constant(domain, default_scalar_type(1e-3))  # Electrostrictive coeff 2

#External Work/Source Terms (Variables from Eq. 13)
B_force = fem.Constant(domain, default_scalar_type((0.0, 0.0))) # Body force f_bar
Traction = fem.Constant(domain, default_scalar_type((0.0, 0.0))) # Traction t_bar
S_f = fem.Constant(domain, default_scalar_type(0.0))            # Free charge density S_f
w_f = fem.Constant(domain, default_scalar_type(0.0))            # Surface charge density w_f

# Kinematics and Electric Field
epsilon_var = ufl.variable(ufl.sym(ufl.grad(u))) # Small strain tensor
E_var = ufl.variable(-ufl.grad(phi))       # Electric field vector

# Invariants
I1 = ufl.tr(epsilon_var)
I2 = ufl.tr(epsilon_var * epsilon_var)
V1 = ufl.inner(E_var, E_var)
K1 = ufl.inner(epsilon_var, ufl.outer(E_var, E_var))

# Total Energy Density Function
psi = (lmbda/2 * I1**2 + mu * I2) - (gamma/2 * V1) - (beta1 * I1 * V1) - (beta2 * K1)

# Explicit Stresses, Displacements, and Moduli
sigma = ufl.diff(psi, epsilon_var)
D = -ufl.diff(psi, E_var)

A = ufl.diff(sigma, epsilon_var)
permittivity = ufl.diff(-D, E_var)
M = ufl.diff(sigma, E_var)

fdim = domain.topology.dim - 1
metadata = {"quadrature_degree": 4}
dx = ufl.Measure("dx", domain=domain, metadata=metadata)
ds = ufl.Measure("ds", domain=domain, metadata=metadata)

# Internal Virtual Work - External Work terms from Eq. (13)/(66) [cite: 66]
# Note: D = -dPsi/dE, sigma = dPsi/deps [cite: 103]
internal_work = ufl.derivative(psi * dx, sol, ufl.TestFunction(V))
external_work = (ufl.inner(v, B_force) * dx + ufl.inner(v, Traction) * ds 
                 - ufl.inner(q, S_f) * dx - ufl.inner(q, w_f) * ds)

residual = internal_work - external_work

J = ufl.derivative(residual, sol, ufl.TrialFunction(V))