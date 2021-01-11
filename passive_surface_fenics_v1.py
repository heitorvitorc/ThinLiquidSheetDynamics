import numpy as np
import matplotlib.pyplot as plt
from vedo.dolfin import ProgressBar 
import matplotlib.pyplot as plt
from dolfin import *
from mshr import *

def surface_displacement(V, u, dt, bmf, surface_marker, right_marker, left_marker):
    '''
    This function obtains the surface displacement as
               displacement = u*dt
    '''
    # Calculate displacement 
    # displacement = u_*dt
    displacement = Function(V)
    # displacement.interpolate(u_*dt)
    b_ = DirichletBC(V, dt*u, bmf, surface_marker)
    v_left = DirichletBC(V.sub(0), Constant(0),bmf, left_marker)
    v_right = DirichletBC(V.sub(0), Constant(0), bmf, right_marker)

    b_.apply(displacement.vector())
    v_left.apply(displacement.vector())
    v_right.apply(displacement.vector())

    return displacement

def surface_displacement2(D, u, dt, bmf, surface_marker, right_marker, left_marker):
    '''
    This function obtains the surface displacement as
               displacement = u*dt
    '''
    # Calculate displacement 
    # displacement = u_*dt
    displacement = Function(D)
    dummy = Function(D)
    dummy.interpolate(u)
    # displacement.interpolate(u_*dt)
    b_ = DirichletBC(D, dt*dummy, bmf, surface_marker)
    v_left = DirichletBC(D.sub(0), Constant(0),bmf, left_marker)
    v_right = DirichletBC(D.sub(0), Constant(0), bmf, right_marker)

    b_.apply(displacement.vector())
    v_left.apply(displacement.vector())
    v_right.apply(displacement.vector())

    return displacement

def gaussian_pulse_surface(D, mesh, boundary_mesh, boundary_markers, surface_marker):
    
    initial_deform = Function(D)
    ampl = 0.02
    sigma_ = .2
    mu_ = L/2
    # sin_shape = Expression(("0", "0.05*sin(4*(pi/L)*x[0])"), L=L, degree=2)
    pulse = Expression(("0", "ampl*(1/(sigma*sqrt(2*pi)))*exp(-((x[0]-mu)*(x[0]-mu))/(2*sigma*sigma))"), ampl=ampl, mu = mu_, sigma = sigma_, degree=2)

    bc = DirichletBC(D, pulse, boundary_markers, surface_marker)
    bc.apply(initial_deform.vector())

    ALE.move(boundary_mesh, initial_deform)
    ALE.move(mesh, boundary_mesh)

    mesh.bounding_box_tree().build(mesh)

# Rate-of-deformation tensor for bulk
def DD(u):
    return sym(nabla_grad(u))

# Cauchy stress tensor
def TT(u, p, mu, I):
    return 2*mu*DD(u) - p*I

# Interface stress tensor
def IST(sigma, kappa, n):
    return sigma*kappa*n 

class InterfaceTension(Expression):
    def __init__(self, T, **kwargs):
        super().__init__(kwargs)
        self.kappa = kappa
        self.n = n

    def eval(self, values, x):
        values = sigma*kappa*n 
    
     def update(self, t, u):
        self.kappa = kappa
        self.n = n

    def value_shape(self):
        return (2,)

def solve_NS_split(V, Q, mesh, bmf, surface_marker, left_marker, right_marker, bottom_marker, u_n, p_n, P_, k, mu, rho, sigma, f):
 
    # define domain and subdomain measures
    # dx = Measure('dx', domain = mesh)
    # ds = Measure('ds', subdomain_data = boundaries)
    ds = Measure('ds', domain = mesh, subdomain_data = bmf) 

    d = mesh.geometry().dim()
    I = Identity(d)

    # get normal vector and curvature of top boundary
    kappa, n = curvature(mesh, ds, surface_marker)

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
    P_left = DirichletBC(Q, Constant(P_), bmf, left_marker)
    P_right = DirichletBC(Q, Constant(P_), bmf, right_marker)
    bcp = [P_left, P_right]

    # Velocity boundary condition
    # Symmetry on bottom 
    symmetry  = DirichletBC(V.sub(1), Constant(0.0), bmf, bottom_marker) 
    bcu = [symmetry]
    
    F1 = rho*dot((u - u_n) / k, v)*dx \
        + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
        + inner(TT(U, p_n, mu, I), DD(v))*dx \
        - dot(mu*nabla_grad(U)*n, v)*ds \
        - dot(f, v)*dx \
        + dot(p_n*n, v)*ds

    # Boundary integral term 
            # stress balance on interface       # pressure term
    b_int = inner(IST(sigma, kappa, n), v)*ds(surface_marker)

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


def divK(nN):
    # div(n)
    div_k = inner(Identity(3),as_tensor([[nN[0].dx(0), 0, 0],
                                         [0, 0, 0],
                                         [0, 0, nN[1].dx(1)]]))
    return div_k 


def curvature(mesh, ds, marker):
    '''
    This function takes the mesh, boundary and boundary marker as inputs and returns 
    the curvature (kappa) and normal unit vector of the respective boundary.
    '''

    V = VectorFunctionSpace(mesh, "CG", 2)
    C = FunctionSpace(mesh, "CG", 1)

    # Projection of the normal vector on P2 space
    u = TrialFunction(V)
    v = TestFunction(V)
    l = TestFunction(C)
    n = FacetNormal(mesh)

    a = inner(u, v('+'))*ds(marker)
    L = inner(n, v('+'))*ds(marker)

    # Solve system
    A = assemble(a, keep_diagonal=True)
    b = assemble(L)
    A.ident_zeros()
    nNorm = Function(V)

    solve(A, nNorm.vector(), b)

    kappa = - divK(nNorm/sqrt(dot(nNorm,nNorm)))

    return kappa, nNorm

# Define classes
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

L = 5*np.pi  # Length of channel
H = 1       # Height of channel

# Create mesh
channel = Rectangle(Point(0,0), Point(L, H))
mesh = generate_mesh(channel, 75)

# d = mesh.geometry().dim()
# I = Identity(d)

# For the variational problem
bmf = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
bmf.set_all(0)

surface_marker = 1
right_marker = 2
bottom_marker = 3
left_marker = 4

Top().mark(bmf, surface_marker)
Right().mark(bmf, right_marker)
Bottom().mark(bmf, bottom_marker)
Left().mark(bmf, left_marker)

ds = Measure('ds', domain = mesh, subdomain_data = bmf) 

# For the deformation of the mesh
boundary_mesh = BoundaryMesh(mesh, "exterior", True)
boundary_markers = MeshFunction("size_t", boundary_mesh, 0)
boundary_markers.set_all(0)

Top().mark(boundary_markers, surface_marker)
Right().mark(boundary_markers, right_marker)
Bottom().mark(boundary_markers, bottom_marker)
Left().mark(boundary_markers, left_marker)

D= VectorFunctionSpace(boundary_mesh, "CG", 2)

# initialize mesh deformation
gaussian_pulse_surface(D, mesh, boundary_mesh, boundary_markers, surface_marker)

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



# Saving files
u_pvd = File("/mnt/d/dissertation/free_surface/results1/u.pvd")
p_pvd = File("/mnt/d/dissertation/free_surface/results1/p.pvd")
m_pvd = File("/mnt/d/dissertation/free_surface/results1/m.pvd")


V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)
u_n = Function(V)
p_n = Function(Q)

pb = ProgressBar(0, num_steps, 1, c='green')
for it in pb.range():
 
    u_, p_ = solve_NS_split(V, Q, mesh, bmf, surface_marker, left_marker, right_marker, bottom_marker, u_n, p_n, P_, k, mu, rho, sigma, f)

    # displacement = surface_displacement(V, u_, k, bmf, surface_marker, right_marker, left_marker)
    displacement = surface_displacement2(D, u_, k, boundary_markers, surface_marker, right_marker, left_marker)

    ALE.move(boundary_mesh, displacement)
    ALE.move(mesh, boundary_mesh)

    # ALE.move(mesh, displacement)
    mesh.bounding_box_tree().build(mesh)

    u_n.set_allow_extrapolation(True)
    p_n.set_allow_extrapolation(True)
    
    u_n.assign(u_)
    p_n.assign(p_)

    m_pvd << mesh
    u_pvd << u_
    p_pvd << p_




    pb.print()
