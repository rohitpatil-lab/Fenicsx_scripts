import dolfinx
from dolfinx import mesh, fem, plot
import dolfinx.fem.petsc
from mpi4py import MPI
import ufl
import numpy as np
import pyvista

L, H = 10.0, 1.0
domain = mesh.create_rectangle(
    MPI.COMM_WORLD, 
    [np.array([0, 0]), np.array([L, H])], 
    [50, 10], 
    cell_type=mesh.CellType.quadrilateral
)

#Define Function Space (2D Vector space)
V = fem.functionspace(domain, ("Lagrange", 1, (2,)))

#Material Properties (Plane Strain Assumption)
E, nu = 1.0e4, 0.3
mu = fem.Constant(domain, E / (2*(1 + nu)))
lmbda = fem.Constant(domain, E*nu / ((1 + nu)*(1 - 2*nu)))

#Kinematics and Constitutive Law
def epsilon(u): 
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lmbda * ufl.div(u) * ufl.Identity(2) + 2*mu*epsilon(u)

# Variational Form
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = fem.Constant(domain, np.array([0, -0.05], dtype=np.float64))

a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L_form = ufl.dot(f, v) * ufl.dx

# Boundary Conditions (Fixed on the left edge x=0)
def left_boundary(x): 
    return np.isclose(x[0], 0)

boundary_facets = mesh.locate_entities_boundary(domain, 1, left_boundary)
bc = fem.dirichletbc(np.array([0, 0], dtype=np.float64), 
                     fem.locate_dofs_topological(V, 1, boundary_facets), V)


problem = dolfinx.fem.petsc.LinearProblem(a, L_form, bcs=[bc], petsc_options_prefix=None)
uh = problem.solve()

#Compute Von Mises Stress in 2D
W_space = fem.functionspace(domain, ("Lagrange", 1))
s = sigma(uh) - (1./2) * ufl.tr(sigma(uh)) * ufl.Identity(2)
von_Mises = ufl.sqrt(2.0 * ufl.inner(s, s)) 

stress_expr = fem.Expression(von_Mises, W_space.element.interpolation_points)
stress_field = fem.Function(W_space)
stress_field.interpolate(stress_expr)


if MPI.COMM_WORLD.rank == 0:
    topology, cell_types, geometry = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    
    # Map 2D displacement to 3D for PyVista (adds a zero Z-coordinate)
    grid.point_data["Displacement"] = np.c_[uh.x.array.reshape(-1, 2), np.zeros(geometry.shape[0])]
    grid.point_data["VonMises"] = stress_field.x.array
    
    warped = grid.warp_by_vector("Displacement", factor=1.0)
    
    # Initialize Plotter
    plotter = pyvista.Plotter(off_screen=True)
    plotter.add_mesh(warped, scalars="VonMises", cmap="jet", show_edges=True)
    
    plotter.view_xy()       
    plotter.camera.zoom(1.2) 

    plotter.screenshot("front_view_beam.png")