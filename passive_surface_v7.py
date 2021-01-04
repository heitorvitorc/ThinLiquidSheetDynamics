from dolfin import * # FEniCs library
from vedo.dolfin import ProgressBar 
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
from utils2 import curvature, readDomains, surface_displacement, update_mesh

# define mesh path
mesh_path = '/mnt/d/dissertation/free_surface/'
mesh_name = 'deformed_thin_sheet_v3'

# # Get subdomain indexes
BoundaryTags = readDomains(mesh_path, mesh_name)
print(BoundaryTags)

# # Load mesh and subdomains
mesh = Mesh(mesh_path + mesh_name + '.xml')
boundaries = MeshFunction('size_t', mesh, mesh_path + mesh_name + "_facet_region.xml")
markers = MeshFunction('size_t', mesh, mesh_path + mesh_name + '_physical_region.xml')

Time = 1  # Total time
num_steps = 100    # number of time steps
dt = Time / num_steps # time step size

# Bulk parameters
mu = 0.01 # kinematic viscosity
rho = 1000 # Density
sigma = 1 # Interfacial tension 
g = 0 # Gravity
P_ = 1.0

k   = Constant(dt) # time step
mu  = Constant(mu) # kinematic viscosity
rho = Constant(rho) # Density
sigma = Constant(sigma) # Surface tension
f   = rho*Constant((0, - g)) # Body force

# Rate-of-deformation tensor for bulk
def DD(u):
    return sym(nabla_grad(u))

# Cauchy stress tensor
def TT(u, p, mu, I):
    return 2*mu*DD(u) - p*I

# Interface stress tensor
def IST(sigma, kappa, n):
    return sigma*kappa*n 

def solve_NS_split(V, Q, mesh, boundaries, markers, BoundaryTags, u_n, p_n, P_, k, mu, rho, sigma, f):
 
    # define domain and subdomain measures
    dx = Measure('dx', domain = mesh)
    ds = Measure('ds', subdomain_data = boundaries)

    d = mesh.geometry().dim()
    I = Identity(d)

    # get normal vector and curvature of top boundary
    kappa, n = curvature(mesh, boundaries, ds, BoundaryTags['Top'])

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)

    # Define functions for solutions at current time steps
    u_  = Function(V)
    p_  = Function(Q)

    # Define expressions used in variational forms
    U   = 0.5*(u_n + u)

    # Pressure boundary on the interface
    P_left = DirichletBC(Q, Constant(P_), boundaries, BoundaryTags['Left'])
    P_right = DirichletBC(Q, Constant(P_), boundaries, BoundaryTags['Right'])
    bcp = [P_left, P_right]

    # Velocity boundary condition
    # Symmetry on bottom 
    symmetry  = DirichletBC(V.sub(1), Constant(0.0),  boundaries, BoundaryTags['Bottom']) 
    bcu = [symmetry]
    
    F1 = rho*dot((u - u_n) / k, v)*dx \
        + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
        + inner(TT(U, p_n, mu, I), DD(v))*dx \
        - dot(mu*nabla_grad(U)*n, v)*ds \
        - dot(f, v)*dx \
        + dot(p_n*n, v)*ds

    # Boundary integral term 
            # stress balance on interface       # pressure term
    b_int = inner(IST(sigma, kappa, n), v)*ds(BoundaryTags['Top'])

    F1 = F1 - b_int

    a1 = lhs(F1)
    L1 = rhs(F1)

    # Define variational problem for step 2
    a2 = dot(nabla_grad(p), nabla_grad(q))*dx
    L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

    # Define variational problem for step 3
    a3 = dot(u, v)*dx
    L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

    # Assemble matrices
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)

    # Apply boundary conditions to matrices
    [bc.apply(A1) for bc in bcu]
    [bc.apply(A2) for bc in bcp]

    # Solve the system ################################

    # Step 1: Tentative velocity
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1)

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2)

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3)
    ################################

    return u_, p_

# Initialize function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)
u_n = Function(V)
p_n = Function(Q)

# Saving files
u_pvd = File("/mnt/d/dissertation/free_surface/results1/u.pvd")
p_pvd = File("/mnt/d/dissertation/free_surface/results1/p.pvd")
m_pvd = File("/mnt/d/dissertation/free_surface/results1/m.pvd")

pb = ProgressBar(0, num_steps, 1, c='green')
for it in pb.range():
 
    u_, p_ = solve_NS_split(V, Q, mesh, boundaries, markers, BoundaryTags, u_n, p_n, P_, k, mu, rho, sigma, f)

    displacement = surface_displacement(V, u_, dt, boundaries, BoundaryTags)

    # mesh, boundaries, markers, dx, ds = update_mesh(mesh, displacement, boundaries, markers)
    mesh, boundaries, markers = update_mesh(mesh, displacement, boundaries, markers)

    
    V = VectorFunctionSpace(mesh, 'P', 2)
    Q = FunctionSpace(mesh, 'P', 1)
    u_n = Function(V)
    p_n = Function(Q)

    u_n.set_allow_extrapolation(True)
    p_n.set_allow_extrapolation(True)
    
    u_n.assign(u_)
    p_n.assign(p_)

    m_pvd << mesh
    u_pvd << u_
    p_pvd << p_


    pb.print()
    

