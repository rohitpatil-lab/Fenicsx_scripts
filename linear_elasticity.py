import dolfinx
from dolfinx import mesh, fem, plot
import dolfinx.fem.petsc
from mpi4py import MPI
import ufl
import numpy as np
import pyvista

# 1. Create Mesh (Unit: meters)
L, H, W = 10.0, 1.0, 1.0
domain = mesh.create_box(MPI.COMM_WORLD, [[0,0,0], [L,H,W]], [30,6,6])

V = fem.functionspace(domain, ("Lagrange", 1, (3,)))

# 3. Material Properties
E, nu = 1.0e4, 0.3
mu = fem.Constant(domain, E / (2*(1 + nu)))
lmbda = fem.Constant(domain, E*nu / ((1 + nu)*(1 - 2*nu)))


def epsilon(u): 
    return ufl.sym(ufl.grad(u)) 

def sigma(u): 
    # Hooke's Law: σ = λ tr(ε) I + 2με
    return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(3) + 2*mu*epsilon(u)

# 5. Variational Form (Principle of Virtual Work)
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = fem.Constant(domain, np.array([0, 0, -0.2], dtype=np.float64))

a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L_form = ufl.dot(f, v) * ufl.dx

# 6. Boundary Conditions (Fixed on the Left face x=0)
def left_boundary(x): 
    return np.isclose(x[0], 0)

boundary_facets = mesh.locate_entities_boundary(domain, domain.topology.dim-1, left_boundary)
bc = fem.dirichletbc(np.array([0,0,0], dtype=np.float64), 
                     fem.locate_dofs_topological(V, domain.topology.dim-1, boundary_facets), V)

# 7. Solve using PETSc Linear Solver
problem = dolfinx.fem.petsc.LinearProblem(
    a, 
    L_form, 
    bcs=[bc], 
    petsc_options_prefix="beam_solver"
)
uh = problem.solve()

# --- POST-PROCESSING ---

# 8. Calculate Von Mises Stress 
W_space = fem.functionspace(domain, ("Lagrange", 1))
s = sigma(uh) - (1./3) * ufl.tr(sigma(uh)) * ufl.Identity(3) 
von_Mises = ufl.sqrt(3./2 * ufl.inner(s, s))

stress_expr = fem.Expression(von_Mises, W_space.element.interpolation_points)
stress_field = fem.Function(W_space)
stress_field.interpolate(stress_expr)

if MPI.COMM_WORLD.rank == 0:
    topology, cell_types, geometry = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    
    grid.point_data["Displacement"] = uh.x.array.reshape((geometry.shape[0], 3))
    grid.point_data["VonMises"] = stress_field.x.array
    grid.set_active_vectors("Displacement")
    
    warped = grid.warp_by_vector("Displacement", factor=1.5)
    
    plotter = pyvista.Plotter(off_screen=True)
    plotter.add_mesh(warped, scalars="VonMises", cmap="jet")
    plotter.screenshot("elasticity_result.png")
    print(f"Success! Max Displacement: {np.max(uh.x.array):.4e}")
    print("Check your folder for 'elasticity_result.png'")