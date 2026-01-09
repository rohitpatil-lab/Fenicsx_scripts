from dolfinx import log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import pyvista
import numpy as np
import ufl
import matplotlib.pyplot as plt

from mpi4py import MPI
from dolfinx import fem, mesh, plot

# Creating mesh
L=0.1
W=0.02
domain = mesh.create_rectangle(
    comm=MPI.COMM_WORLD, 
    points=((0.0, 0.0), (L, W)), 
    n=[100, 50], 
    cell_type=mesh.CellType.quadrilateral
)

d = domain.geometry.dim

# Assigning the shape function
V = fem.functionspace(domain, ("Lagrange", 1 ,(domain.geometry.dim,)))