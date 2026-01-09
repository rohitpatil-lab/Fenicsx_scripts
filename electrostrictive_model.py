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

#Material Parameters
lmbda = fem.Constant(domain, default_scalar_type(1e4))  # Lame lambda
mu = fem.Constant(domain, default_scalar_type(5e3))     # Lame mu
gamma = fem.Constant(domain, default_scalar_type(1e-2))  # Dielectric permittivity
beta1 = fem.Constant(domain, default_scalar_type(1e-3))  # Electrostrictive coeff 1
beta2 = fem.Constant(domain, default_scalar_type(1e-3))  # Electrostrictive coeff 2

# Kinematics and Electric Field
epsilon = ufl.sym(ufl.grad(u)) # Small strain tensor
E_field = -ufl.grad(phi)       # Electric field vector

# Invariants
I1 = ufl.tr(epsilon)
I2 = ufl.tr(epsilon * epsilon)
V1 = ufl.inner(E_field, E_field)
K1 = ufl.inner(epsilon, ufl.outer(E_field, E_field))

# Total Energy Density Function
psi = (lmbda/2 * I1**2 + mu * I2) - (gamma/2 * V1) - (beta1 * I1 * V1) - (beta2 * K1)