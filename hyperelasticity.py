from dolfinx import log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import pyvista
import numpy as np
import ufl

from mpi4py import MPI
from dolfinx import fem, mesh, plot

# Creating mesh
L=0.1
W=0.02
domain = mesh.create_rectangle(
    comm=MPI.COMM_WORLD, 
    points=((0.0, 0.0), (L, W)), 
    n=[50, 25], 
    cell_type=mesh.CellType.quadrilateral
)

d = domain.geometry.dim

# Assigning the shape function
V = fem.functionspace(domain, ("Lagrange", 1 ,(domain.geometry.dim,)))
W_stress = fem.functionspace(domain, ("Discontinuous Lagrange", 0, (d, d)))
P_func = fem.Function(W_stress, name="Piola_Stress")

def left(x):
    return np.isclose(x[0], 0)


def right(x):
    return np.isclose(x[0], L)

fdim = domain.topology.dim - 1
left_facets = mesh.locate_entities_boundary(domain, fdim, left)
right_facets = mesh.locate_entities_boundary(domain, fdim, right)

marked_facets = np.hstack([left_facets, right_facets])
marked_values = np.hstack([np.full_like(left_facets, 1), np.full_like(right_facets, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(
    domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets]
)

u_bc = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)

left_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
bcs = [fem.dirichletbc(u_bc, left_dofs, V)]

B = fem.Constant(domain, default_scalar_type((0, 0)))
T = fem.Constant(domain, default_scalar_type((0, 0)))

v = ufl.TestFunction(V)
u = fem.Function(V)

# Spatial dimension
d = len(u)

# Identity tensor
I = ufl.variable(ufl.Identity(d))

# Deformation gradient
F = ufl.variable(I + ufl.grad(u))

# Right Cauchy-Green tensor
C = ufl.variable(F.T * F)

# Invariants of deformation tensors
Ic = ufl.variable(ufl.tr(C))
J = ufl.variable(ufl.det(F))

E = default_scalar_type(1.0e4)
nu = default_scalar_type(0.3)
mu = fem.Constant(domain, E / (2 * (1 + nu)))
lmbda = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))

# Neo-Hookean model
psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J)) ** 2

P = ufl.diff(psi, F)
P_magnitude_expr = ufl.sqrt(ufl.inner(P, P))

metadata = {"quadrature_degree": 4}
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

residual = (
    ufl.inner(ufl.grad(v), P) * dx - ufl.inner(v, B) * dx - ufl.inner(v, T) * ds(2)
)

petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_monitor": None,
    "snes_atol": 1e-8,
    "snes_rtol": 1e-8,
    "snes_stol": 1e-8,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
problem = NonlinearProblem(
    residual,
    u,
    bcs=bcs,
    petsc_options=petsc_options,
    petsc_options_prefix="hyperelasticity",
)

plotter = pyvista.Plotter(shape=(1, 2))
plotter.open_gif("deformation_and_stress.gif", fps=3)

topology, cells, geometry = plot.vtk_mesh(u.function_space)
function_grid = pyvista.UnstructuredGrid(topology, cells, geometry)

values = np.zeros((geometry.shape[0], 3))
values[:, : len(u)] = u.x.array.reshape(geometry.shape[0], len(u))
function_grid["u"] = values
function_grid.set_active_vectors("u")

# Warp mesh by deformation
warped = function_grid.warp_by_vector("u", factor=1)
warped.set_active_vectors("u")

# Add mesh to plotter and visualize
plotter.subplot(0, 0)
plotter.add_text("Displacement Magnitude", font_size=10)


warped["mag"] = np.zeros(geometry.shape[0])
actor = plotter.add_mesh(warped, scalars="mag", show_edges=True, lighting=False, clim=[0, 0.01])
plotter.view_xy()

plotter.subplot(0, 1)
plotter.add_text("Stress Magnitude (P)", font_size=10)
warped_stress = warped.copy()
warped_stress.cell_data["P_mag"] = np.zeros(function_grid.n_cells)
actor_p = plotter.add_mesh(warped_stress, scalars="P_mag", show_edges=True, lighting=False, clim=[0, 5000])
plotter.view_xy()

# Compute magnitude of displacement to visualize in GIF
Vs = fem.functionspace(domain, ("Lagrange", 1))
magnitude = fem.Function(Vs)
us = fem.Expression(
    ufl.sqrt(sum([u[i] ** 2 for i in range(len(u))])), Vs.element.interpolation_points
)
magnitude.interpolate(us)
warped["mag"] = magnitude.x.array
V_mag_stress = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
P_mag_func = fem.Function(V_mag_stress)
stress_expr = fem.Expression(P_magnitude_expr, V_mag_stress.element.interpolation_points)


log.set_log_level(log.LogLevel.INFO)
tval0 = -2
for n in range(1, 10):
    T.value[1] = n * tval0
    problem.solve()
    converged = problem.solver.getConvergedReason() > 0
    num_its = problem.solver.getIterationNumber()
    assert converged > 0, f"Solver did not converge with reason {converged}."

    print(f"Time step {n}, Number of iterations {num_its}, Load {T.value}")
    function_grid["u"][:, : len(u)] = u.x.array.reshape(geometry.shape[0], len(u))
    magnitude.interpolate(us)
    warped.set_active_scalars("mag")
    warped_n = function_grid.warp_by_vector(factor=1)
    
    # Update Displacement points and data
    warped.points[:, :] = warped_n.points
    current_mag = magnitude.x.array
    warped.point_data["mag"][:] = current_mag

    # Update Stress Visuals
    P_mag_func.interpolate(stress_expr)
    warped_stress.points[:, :] = warped_n.points
    warped_stress.cell_data["P_mag"][:] = P_mag_func.x.array

    plotter.subplot(0, 0)
    max_disp = np.max(current_mag)
    if max_disp > 0:
        plotter.update_scalar_bar_range([0, max_disp])
-
    plotter.subplot(0, 0)
    warped.points[:, :] = warped_n.points
    current_mag = magnitude.x.array
    warped.point_data["mag"][:] = current_mag
    warped.set_active_scalars("mag")
    max_disp = np.max(current_mag)
    if max_disp > 1e-10:
        plotter.update_scalar_bar_range([0, max_disp], name="mag")
-
    plotter.subplot(0, 1)
    warped_stress.points[:, :] = warped_n.points
    warped_stress.cell_data["P_mag"][:] = P_mag_func.x.array
    max_stress = np.max(P_mag_func.x.array)
    if max_stress > 1e-10:
        plotter.update_scalar_bar_range([0, max_stress], name="P_mag")
    
    plotter.render()
    plotter.write_frame()

plotter.close()