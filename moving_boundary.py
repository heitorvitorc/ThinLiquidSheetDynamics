from dolfin import * # FEniCs library
from vedo.dolfin import ProgressBar 
# from vedo.dolfin import plot# as vplt # Post-processing
from mshr import *
import numpy as np
# import random
# import matplotlib.pyplot as plt

print("FEniCs version: ", dolfin.__version__)

def noslip_pointwise(FS, bflag, bmesh, bmf):
    '''
    FS -> VectorFunctionSpace
    bflag = flag index related to the boundary
    bmesh -> boundary mesh
    bmf -> boundary marker mesh function
    '''
    topBC = []
    for i in range(bmesh.num_vertices()):
        if bmf[i] == bflag:
            xbc = bmesh.coordinates()[i,0]
            ybc = bmesh.coordinates()[i,1]

            topBC.append(DirichletBC(FS, Constant((0.0, 0.0)), 
            "near(x[0],"+str(xbc)+") && near(x[1],"+str(ybc)+")", "pointwise"))

    return topBC

def update_bc_movement(FS, bflag, bmesh, bmf):
    '''
    FS -> VectorFunctionSpace
    bflag = flag index related to the boundary
    bmesh -> boundary mesh
    bmf -> boundary marker mesh function
    '''
    k = 2
    Omega = 5
    topBC = []
    for i in range(bmesh.num_vertices()):
        if (bmf[i] == bflag):
            # store previous coordinates
            x_previous = bmesh.coordinates()[i,0]
            y_previous = bmesh.coordinates()[i,1]

            # move mesh
            bmesh.coordinates()[i,1] -= 0.001*np.sin((np.pi/L)*bmesh.coordinates()[i,0])
            # bmesh.coordinates()[i,1] -= np.sin(k*bmesh.coordinates()[i,0] + Omega*t)/200

            #store new coordinates
            x_current = bmesh.coordinates()[i,0]
            y_current = bmesh.coordinates()[i,1]

            # Calculate point velocity
            vx = (x_current-x_previous)/dt
            vy = (y_current-y_previous)/dt
            topBC.append(DirichletBC(FS, Constant((vx, vy)), 
            "near(x[0],"+str(x_current)+") && near(x[1],"+str(y_current)+")", "pointwise"))
            
    return topBC

class Left(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return near(x[0], 0.0)# and on_boundary

class Right(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return near(x[0], L)# and on_boundary

class Bottom(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return near(x[1], 0.0)# and on_boundary

class Top(SubDomain): # This will later on become a free surface
    def __init__(self):
        SubDomain.__init__(self)
    
    def inside(self, x, on_boundary):
        return near(x[1], H)


L = 2*np.pi  # Length of channel
H = 1       # Height of channel

Time = 1  # Total time
num_steps = 100    # number of time steps
dt = Time / num_steps # time step size

print("time step: ", dt)
print("Number of steps: ", num_steps)


# Fluid parameters
mu = 1 # kinematic viscosity
rho = 1 # Density

channel = Rectangle(Point(0,0), Point(L, H))
mesh = generate_mesh(channel, 50)

left = Left()
right = Right()
top = Top()
bottom = Bottom()

# Extract boundary mesh
bmesh = BoundaryMesh(mesh, "exterior", True)

# Pass 0 as last argument to define a function on vertices,
#  i.e., mesh entities of topological dimension 0. 
bmf = MeshFunction("size_t", bmesh, 0)
bmf.set_all(0)

top.mark(bmf, 1) # Mark boundary with flag

V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Impose zero pressure on left and right boundaries
P_left = DirichletBC(Q, Constant(0.0), left)
P_right = DirichletBC(Q, Constant(0.0), right)
bcp = [P_left, P_right]

# Velocity boundary condition
symmetry  = DirichletBC(V.sub(1), Constant(0.0), bottom)   
bcu = [symmetry]#, Uleft]

# Top boundary
topBC = noslip_pointwise(V, 1, bmesh, bmf)
bcU = bcu + topBC

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u_n = Function(V)
u_  = Function(V)
p_n = Function(Q)
p_  = Function(Q)

# Define expressions used in variational forms
U   = 0.5*(u_n + u)
n   = FacetNormal(mesh)
f   = Constant((0, 0))
k   = Constant(dt)
mu  = Constant(mu)
rho = Constant(rho)

# Define strain-rate tensor
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# Define variational problem for step 1
F1 = rho*dot((u - u_n) / k, v)*dx + \
     rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + inner(sigma(U, p_n), epsilon(v))*dx \
   + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
   - dot(f, v)*dx
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
[bc.apply(A1) for bc in bcU]
[bc.apply(A2) for bc in bcp]

# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"
# Use nonzero guesses - essential for CG with non-symmetric BC
parameters['krylov_solver']['nonzero_initial_guess'] = True

mesh_pvd = File("./results5/mesh.pvd")
u_pvd = File("./results5/u.pvd")
p_pvd = File("./results5/p.pvd")

t = 0

print("Starting simulation")
pb = ProgressBar(0, num_steps, 1, c='green')
for n in pb.range(): 
# for n in range(num_steps):    
    
    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcU]
    # [top.apply(b1) for top in topBC]
    solve(A1, u_.vector(), b1)#, "bicgstab", "default")

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2)#, "bicgstab", prec)

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3)#, "bicgstab", "default")

    # Update bc
    topBC = update_bc_movement(V, 1, bmesh, bmf)
    # Move bc
    ALE.move(mesh, bmesh)
    mesh.bounding_box_tree().build(mesh)

    # Update boundary condition
    bcU = bcu + topBC

    # Assign values for the next iteration
    u_n.assign(u_)
    p_n.assign(p_)

    t += dt
    # print(t)

    # save solution
    mesh_pvd << mesh
    u_pvd << u_
    p_pvd << p_
    pb.print()

print("Simulation finished.")