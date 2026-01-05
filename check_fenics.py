import dolfinx
from mpi4py import MPI
from dolfinx import mesh
import ufl

# 1. Print start message
print("--- Starting FEniCSx Verification ---")

# 2. Create a 2D mesh 
domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

# 3. Print the version and success message
print(f"FEniCSx Version: {dolfinx.__version__}")
print("Successfully connected VS Code to FEniCSx on WSL!")
print("--- Test Complete ---")