import numpy as np
import timeit
import matplotlib.pyplot as plt
from vedo.dolfin import ProgressBar 
import matplotlib.pyplot as plt
from dolfin import *
from mshr import *

# Calculate Auto-time step
def autoTimestep(no_iterations, dt_prev, dt_min = 1e-6, dt_max = 1, min_iter = 4, dt_dt = 3, increment = 5):

    # Check if 
    if no_iterations < min_iter:
        # Increase timestep if possible
        dt = min(increment*dt_prev, dt_max)

    elif no_iterations > min_iter + dt_dt:
        # reduce timestep if necessary
        dt = max(dt_min, (1/increment)*dt_prev)

    else:
        # Keep the timestep - Passing equal dtMin==dtMax, auto-timestep is turned off.
        dt = dt_prev
    
    return dt


def _mesh(L, H, Nx, Ny):
    '''
    Returns a mesh refined near top boundary according to the reference below.
    Reference: Mortensen and Valen-Sendstad (2016), arXiv:1602.03643v1
    '''
    m = RectangleMesh(Point(0, 0), Point(L, H), Nx, Ny)
    x = m.coordinates()
    x[:,1] = np.arctan(np.pi*(x[:, 1])) / np.arctan(np.pi)

    return m


def surface_displacement(D, W, u, dt, bmf, domain_bmf, surface_marker, right_marker, left_marker):
    '''
    This function obtains the surface displacement as
               displacement = u*dt
    '''
    # Calculate displacement 
    # displacement = u_*dt
    b1 = DirichletBC(W.sub(0).sub(0), Constant(0.0), domain_bmf, surface_marker)
    b1.apply(u.vector())

    displacement = Function(D)
    displacement.set_allow_extrapolation(True)
    dummy = Function(D)
    dummy.set_allow_extrapolation(True)
    dummy.interpolate(u)
    # displacement.interpolate(u_*dt)
    b_ = DirichletBC(D, dt*dummy, bmf, surface_marker)
    v_left = DirichletBC(D.sub(0), Constant(0),bmf, left_marker)
    v_right = DirichletBC(D.sub(0), Constant(0), bmf, right_marker)

    b_.apply(displacement.vector())
    v_left.apply(displacement.vector())
    v_right.apply(displacement.vector())

    return displacement

def gaussian_pulse_surface(D, mesh, boundary_mesh, boundary_markers, surface_marker, amplitude=0.02):
    
    initial_deform = Function(D)
    ampl = amplitude
    sigma_ = .2
    mu_ = L/2
    # sin_shape = Expression(("0", "0.05*sin(4*(pi/L)*x[0])"), L=L, degree=2)
    pulse = Expression(("0", "ampl*(1/(sigma*sqrt(2*pi)))*exp(-((x[0]-mu)*(x[0]-mu))/(2*sigma*sigma))"), ampl=ampl, mu = mu_, sigma = sigma_, degree=2)

    bc = DirichletBC(D, pulse, boundary_markers, surface_marker)
    bc.apply(initial_deform.vector())

    ALE.move(boundary_mesh, initial_deform)
    ALE.move(mesh, boundary_mesh)

    mesh.bounding_box_tree().build(mesh)


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

def solve_NS_monolithic(mesh, bmf, surface_marker, left_marker, right_marker, bottom_marker, w0, k, mu, rho, sigma, f):
    
    # Solver Parameters
    absTol = 1e-10          # absolute tolerance: residual value
    relTol = 1e-12          # relative tolerance: change with respect to previous
    maxIter = 15           # Maximum iterations for non-linear solver
    nlinSolver = 'newton'   # Non-Linear Solver(Coupled Pressure/Velocity)
    linSolver = 'mumps'     # Linear Solver(Concentration)
    alpha = 0.9             # relaxation

    P_ = Constant(0.0)

    dx = Measure('dx', domain=mesh)
    ds = Measure('ds', domain=mesh, subdomain_data = bmf)

    d = mesh.geometry().dim()
    I = Identity(d)
    n = FacetNormal(mesh)
    kappa, n_surf = curvature(mesh, ds, surface_marker)

    Uel = VectorElement('Lagrange', mesh.ufl_cell(), 2)
    Pel = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    UPel = MixedElement([Uel,Pel])
   
    # Function Space
    W = FunctionSpace(mesh, UPel)
    (U, P) = W.split()

    # Trial and Test functions
    dw = TrialFunction(W)
    (v,q) = TestFunctions(W)
    w = Function(W)
    (u, p) = (as_vector((w[0], w[1])), w[2])
    W0 = Function(W)

    # Symmetry boundary condition on bottom boundary
    symmetry = DirichletBC(W.sub(0).sub(1), Constant(0.0), bmf, bottom_marker)
    bcu = [symmetry]

    (u0, p0) = w0.leaf_node().split()

    a1 = rho*dot((u-u0)/k,v)*dx() + alpha *(rho*dot(dot(u ,grad(u) ),v) + inner(TT(u,p,mu, I),DD(v)))*dx() + \
                                                (1-alpha)*(rho*dot(dot(u0,grad(u0)),v) + inner(TT(u0,p0,mu, I),DD(v)))*dx()  # Relaxation

            # Inlet Pressure                                    # Outlet Pressure                                      # Gravity
    L1 = - (P_)*dot(n,v)*ds(left_marker) - (P_)*dot(n,v)*ds(right_marker) # + inner(rho*g,v)*dx()

    # Boundary integral term 
            # stress balance on interface       
    b_int = inner(IST(sigma, kappa, n_surf), v)*ds(surface_marker)


    ## Mass Conservation(Continuity)
    a2 = (q*div(u))*dx()
    L2 = 0

    ## Complete Weak Form
    F = (a1 + a2) - (L1 + b_int + L2)
    ## Jacobian Matrix
    J = derivative(F,w,dw)
    
    # Problem and Solver definitions
    problemU = NonlinearVariationalProblem(F,w,bcu,J)
    solverU = NonlinearVariationalSolver(problemU)

    # Solver Parameters
    prmU = solverU.parameters
    # #info(prmU,True)  # get full info of the parameters
    prmU['nonlinear_solver'] = nlinSolver
    prmU['newton_solver']['absolute_tolerance'] = absTol
    prmU['newton_solver']['relative_tolerance'] = relTol
    prmU['newton_solver']['maximum_iterations'] = maxIter
    prmU['newton_solver']['linear_solver'] = linSolver

    # Solve problem
    try:
        (no_iterations,converged) = solverU.solve()
    except:
        converged = False
        no_iterations = maxIter
        print("Convergence failed")
        w = w0
        
    
    return w, no_iterations


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


# Rate-of-deformation tensor for bulk
def DD(u):
    return sym(nabla_grad(u))

# Cauchy stress tensor
def TT(u, p, mu, I):
    return 2*mu*DD(u) - p*I

# Interface stress tensor
def IST(sigma, kappa, n):
    return sigma*kappa*n 


L = 5*np.pi  # Length of channel
H = 1       # Height of channel
Nx = 300
Ny = 50



mesh = _mesh(L, H, Nx, Ny)
# Create mesh
# channel = Rectangle(Point(0,0), Point(L, H))
# mesh = generate_mesh(channel, 175)

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

# ds = Measure('ds', domain = mesh, subdomain_data = bmf) 

# For the deformation of the mesh
boundary_mesh = BoundaryMesh(mesh, "exterior", True)
boundary_markers = MeshFunction("size_t", boundary_mesh, 0)
boundary_markers.set_all(0)

Top().mark(boundary_markers, surface_marker)
Right().mark(boundary_markers, right_marker)
Bottom().mark(boundary_markers, bottom_marker)
Left().mark(boundary_markers, left_marker)

D= VectorFunctionSpace(boundary_mesh, "CG", 1)

# initialize mesh deformation
gaussian_pulse_surface(D, mesh, boundary_mesh, boundary_markers, surface_marker, amplitude=0.05)


# Bulk parameters
mu = 0.01 # kinematic viscosity
rho = 1000 # Density
sigma = 1 # Interfacial tension (water/air)
g = 0 # Gravity
P_ = 0.0

# k   = Constant(dt) # time step
mu  = Constant(mu) # kinematic viscosity
rho = Constant(rho) # Density
sigma = Constant(sigma) # Surface tension
f   = rho*Constant((0, - g)) # Body force

# Saving files
u_pvd = File("/mnt/d/dissertation/free_surface/results5/u.pvd")
p_pvd = File("/mnt/d/dissertation/free_surface/results5/p.pvd")
m_pvd = File("/mnt/d/dissertation/free_surface/results5/m.pvd")

Uel = VectorElement('Lagrange', mesh.ufl_cell(), 2)
Pel = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
UPel = MixedElement([Uel,Pel])
W = FunctionSpace(mesh, UPel)
w_n = Function(W)

Time = 1  # Total time
t = 0
dt = 0.005
dt_save = 0.01 # Number of results to save

# Time execution time
start = timeit.default_timer()

while t <= Time and dt > 0.0:
    print("t = ", t)
    print("dt = ", dt)

    k = Constant(dt)
    w_, no_iterations = solve_NS_monolithic(mesh, bmf, surface_marker, left_marker, right_marker, bottom_marker, w_n, k, mu, rho, sigma, f)

    (u_, p_) = w_.leaf_node().split()
    # displacement = surface_displacement(D, u_, k, boundary_markers, surface_marker, right_marker, left_marker)
    displacement = surface_displacement(D, W, u_, dt, boundary_markers, bmf, surface_marker, right_marker, left_marker)
    
    ALE.move(boundary_mesh, displacement)
    ALE.move(mesh, boundary_mesh)

    # ALE.move(mesh, displacement)
    mesh.bounding_box_tree().build(mesh)

    w_n.assign(w_)

    # Saving results
    if t==0 or t >= t_sav or t>= Time:
        print("------------ Saving results --------------")
        print("t = ", t)
        print("dt = ", dt)
        print("------------------------------------------")
        t_sav = t + dt_save

        u_.rename("u", "Velocity Field")
        p_.rename("p", "Pressure Field")

        m_pvd << (mesh, t)
        u_pvd << (u_, t)
        p_pvd << (p_, t)


    dt = autoTimestep(no_iterations, dt) #, dt_min = 1e-3, dt_max = 1, min_iter = 3, dt_dt = 3, increment = 2)
    
    t += dt

# u_xdmf.close()
# p_xdmf.close()

stop = timeit.default_timer()
total_time = stop - start
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)

print("Simulation time %dh:%dmin:%ds \n" % (hours, mins, secs))