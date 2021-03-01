import numpy as np
import timeit
import matplotlib.pyplot as plt
from dolfin import *
from mshr import *

'''
Comments:

Testing different boundary conditions for the lateral boundaries. 

Imposition of pressure difference condition: dp = vdw(x=left) - vdw (x=right) = 0

'''

# Calculate Auto-time step
def autoTimestep(no_iterations, dt_prev, dt_min = 1e-7, dt_max = 0.5, min_iter = 3, dt_dt = 2, increment = 2):

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
    # Define scale
    scale = min(L, H)
    # Create rectangle mesh
    m = RectangleMesh(Point(0, 0), Point(L, H), Nx, Ny)
    # Get coordinates
    x = m.coordinates()
    # refine near boundary
    x[:,1] = np.arctan(np.pi*(x[:, 1])) / np.arctan(np.pi)

    # downscale coordinate
    # x[:,0] = x[:,0]*scale
    # x[:,1] = x[:,1]*scale  

    return m

def shift_mesh(mesh, L):
    '''
    Shifts mesh to the left
    '''
    x = mesh.coordinates()
    x[:,0] =  x[:,0] - L/2

    mesh.bounding_box_tree().build(mesh)

    # return mesh

def dimensionless_mesh(mesh, L, H, scale):
    '''
    Returns dimensionless mesh/boundary mesh
    '''
    mesh1 = Mesh(mesh)
    x = mesh1.coordinates()
    
    x[:,0] =  x[:,0]/(((L+2)/2)*scale)
    # x[:,0] =  x[:,0]*(scale*2) # scale up mesh
    x[:,1] =  x[:,1]/(scale*(H/0.6))

    mesh1.bounding_box_tree().build(mesh1)

    return mesh1


def surface_displacement(D, W, u, dt, bmf, domain_bmf, surface_marker, right_marker, left_marker):
    '''
    This function obtains the surface displacement as
               displacement = u*dt
    '''
    # Calculate displacement 
    # displacement = u_*dt
    # b1 = DirichletBC(W.sub(0).sub(0), Constant(0.0), domain_bmf, surface_marker)
    # b1.apply(u.vector())

    displacement = Function(D)
    displacement.set_allow_extrapolation(True)
    dummy = Function(D)
    dummy.set_allow_extrapolation(True)
    dummy.interpolate(u)
    # displacement.interpolate(u_*dt)
    b_ = DirichletBC(D, dt*dummy, bmf, surface_marker)
    # v_left = DirichletBC(D.sub(0), Constant(0),bmf, left_marker)
    # v_right = DirichletBC(D.sub(0), Constant(0), bmf, right_marker)

    b_.apply(displacement.vector())
    # v_left.apply(displacement.vector())
    # v_right.apply(displacement.vector())

    return displacement

def gaussian_pulse_surface(D, mesh, boundary_mesh, boundary_markers, surface_marker, sigma = .2, amplitude= - 0.05):

    initial_deform = Function(D)
    ampl = amplitude
    sigma_ = sigma
    mu_ = L/2

    # cos_shape = Expression(("0", "1 - *cos(pi*x[0])"), L=L, degree=2)
    pulse = Expression(("0", "ampl*(1/(sigma*sqrt(2*pi)))*exp(-((x[0]-mu)*(x[0]-mu))/(2*sigma*sigma))"), ampl=ampl, mu = mu_, sigma = sigma_, degree=2)

    bc = DirichletBC(D, pulse, boundary_markers, surface_marker)
    bc.apply(initial_deform.vector())

    ALE.move(boundary_mesh, initial_deform)
    ALE.move(mesh, boundary_mesh)

    mesh.bounding_box_tree().build(mesh)

def sin_surface(D, mesh, boundary_mesh, boundary_markers, surface_marker, ampl=0.1):
    
    initial_deform = Function(D)
    sin_shape = Expression(("0", "ampl*sin(x[0]*0.32*pi + pi/2) - ampl"), ampl=ampl, degree=2)

    bc = DirichletBC(D, sin_shape, boundary_markers, surface_marker)
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

def solve_NS_monolithic_dimensionless(mesh, bmf, surface_marker, left_marker, right_marker, bottom_marker, w0, k, mu, rho, sigma, A, f, scale):
    
    # Solver Parameters
    absTol = 1e-9          # absolute tolerance: residual value
    relTol = 1e-10          # relative tolerance: change with respect to previous
    maxIter =   20         # Maximum iterations for non-linear solver
    nlinSolver = 'newton'   # Non-Linear Solver(Coupled Pressure/Velocity)
    linSolver = 'mumps'     # Linear Solver(Concentration)
    alpha = 0.9             # relaxation

    dx = Measure('dx', domain=mesh)
    ds = Measure('ds', domain=mesh, subdomain_data = bmf)

    d = mesh.geometry().dim()
    I = Identity(d)
    n = FacetNormal(mesh)
    x = SpatialCoordinate(mesh)


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

    
    '''
    Dimensionless variables ( _):
    x_, y_ = x/h0, y/h0
    t_ = t/(h0²/nu)
    u_ = u/(nu/h0)
    p_ = p/((rho*nu²)/h0²)

    
    '''

    a1 = dot((u-u0)/k,v)*dx() + alpha *(dot(dot(u ,grad(u) ),v) + inner(TT(u,p,mu, I),DD(v)))*dx() + \
                                                (1-alpha)*(rho*dot(dot(u0,grad(u0)),v) + inner(TT(u0,p0,mu, I),DD(v)))*dx()  # Relaxation

            # Inlet Pressure                                    # Outlet Pressure                                      # Gravity
    L1 = - (VdW_macete_ND(A, scale))*dot(n,v)*ds(left_marker) - (VdW_macete(A, scale))*dot(n,v)*ds(right_marker) # + inner(rho*g,v)*dx()
    # L1 = - (P_)*dot(n,v)*ds(right_marker) # + inner(rho*g,v)*dx()

    # Boundary integral term 
            # stress balance on interface       
    b_int = inner(IST(sigma, kappa, n_surf), v)*ds(surface_marker)

    # Van der Waals forces
    vdw = - VdW(A, x)*dot(n, v)*ds(surface_marker)

    L1 += vdw

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
        
    
    return w, no_iterations, converged, kappa, n_surf



def solve_NS_monolithic(mesh, bmf, surface_marker, left_marker, right_marker, bottom_marker, w0, k, mu, rho, sigma, A, f, scale):
    
    # Solver Parameters
    absTol = 1e-9          # absolute tolerance: residual value
    relTol = 1e-10          # relative tolerance: change with respect to previous
    maxIter =   20         # Maximum iterations for non-linear solver
    nlinSolver = 'newton'   # Non-Linear Solver(Coupled Pressure/Velocity)
    linSolver = 'mumps'     # Linear Solver(Concentration)
    alpha = 0.9             # relaxation

    dx = Measure('dx', domain=mesh)
    ds = Measure('ds', domain=mesh, subdomain_data = bmf)

    d = mesh.geometry().dim()
    I = Identity(d)
    n = FacetNormal(mesh)
    x = SpatialCoordinate(mesh)


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
    L1 = - (VdW_macete(A, scale))*dot(n,v)*ds(left_marker) - (VdW_macete(A, scale))*dot(n,v)*ds(right_marker) # + inner(rho*g,v)*dx()
    # L1 = - (P_)*dot(n,v)*ds(right_marker) # + inner(rho*g,v)*dx()

    # Boundary integral term 
            # stress balance on interface       
    b_int = inner(IST(sigma, kappa, n_surf), v)*ds(surface_marker)

    # Van der Waals forces
    vdw = - VdW(A, x)*dot(n, v)*ds(surface_marker)

    L1 += vdw

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
        
    
    return w, no_iterations, converged, kappa, n_surf


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

#! Non-Dimensional (ND) form
# Rate-of-deformation tensor for bulk
def DD(u):
    return sym(nabla_grad(u))

# Cauchy stress tensor
def TT_ND(u, p, rho, I):
    return (2/rho)*DD(u) - p*I

# Interface stress tensor
def IST_ND(sigma, kappa, rho, mu, h0, n):

    S = (3*(mu**3)*sigma/(rho*h0**2))

    return S*kappa*n

def VdW_ND(A, x):
    '''
    Van der Waals forces on surface
    A = Hamacker constant for the fluid
    '''
    return A/(2*np.pi*pow(2*x[1],3))

def VdW_macete_ND(A, scale):
    '''
    Van der Waals forces on surface
    A = Hamacker constant for the fluid
    '''
    return A/(2*np.pi*pow(2*scale,3))



# Rate-of-deformation tensor for bulk
def DD(u):
    return sym(nabla_grad(u))

# Cauchy stress tensor
def TT(u, p, mu, I):
    return 2*mu*DD(u) - p*I

# Interface stress tensor
def IST(sigma, kappa, n):
    return sigma*kappa*n 

def VdW(A, x):
    '''
    Van der Waals forces on surface
    A = Hamacker constant for the fluid
    '''
    return A/(2*np.pi*pow(2*x[1],3))

def VdW_macete(A, scale):
    '''
    Van der Waals forces on surface
    A = Hamacker constant for the fluid
    '''
    return A/(2*np.pi*pow(2*scale,3))

# ------------------------------------------------------------------------------#
sim_attempt = "microscale_vdw_lateral_frustrated_attempt_1/"

H = 1     # Initial height of channel
L = 2*np.pi*H + 5 # Initial length of channel

scale = 0.5e-6
L_problem = 0.407173*2*scale # Half wavelength perturbation

Nx = 650
Ny = 70

mesh = _mesh(L, H, Nx, Ny)

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
sigma_wave = .8
amplitude = - H*0.65
gaussian_pulse_surface(D, mesh, boundary_mesh, boundary_markers, surface_marker, sigma=sigma_wave, amplitude=amplitude)

# sin_surface(D, mesh, boundary_mesh, boundary_markers, surface_marker, ampl=0.1)

# Shifts mesh to the left so that the perturbation is given at x = 0
shift_mesh(mesh, L)
shift_mesh(boundary_mesh, L)

mesh.scale(scale)
mesh.bounding_box_tree().build(mesh)
boundary_mesh.scale(scale)
boundary_mesh.bounding_box_tree().build(boundary_mesh)

# Bulk parameters
mu_ = 0.01 # kinematic viscosity
rho_ = 1000 # Density
nu = mu_/rho_
sigma_ = 0.06 # Interfacial tension (water/air)
A__ = 2.36e-19 # Hamaker constant - Ref: Prokopovich & Rahnejat (2010) - Surface phenomena in thin-film tribology
A_ = A__ / (6*np.pi*rho_*nu*scale)

g = 0 # Gravity

mu  = Constant(mu_) # kinematic viscosity
rho = Constant(rho_) # Density
sigma = Constant(sigma_) # Surface tension
f   = rho*Constant((0, - g)) # Body force

# Saving files
u_pvd = File("/mnt/d/dissertation/free_surface/" + sim_attempt +"u.pvd")
p_pvd = File("/mnt/d/dissertation/free_surface/"+ sim_attempt +"p.pvd")
m_pvd = File("/mnt/d/dissertation/free_surface/"+ sim_attempt +"m.pvd")
displacement_pvd = File("/mnt/d/dissertation/free_surface/"+ sim_attempt +"displacement.pvd")

u_dim_pvd = File("/mnt/d/dissertation/free_surface/" + sim_attempt +"/dimensionless/u_dim.pvd")
m_dim_pvd = File("/mnt/d/dissertation/free_surface/" + sim_attempt +"/dimensionless/m_dim.pvd")
b_m_dim_pvd = File("/mnt/d/dissertation/free_surface/" + sim_attempt +"/dimensionless/bm_dim.pvd")

Uel = VectorElement('Lagrange', mesh.ufl_cell(), 2)
Pel = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
UPel = MixedElement([Uel,Pel])
W = FunctionSpace(mesh, UPel)
w_n = Function(W)

Time = 1e-2 # Total time
t = 0
dt = 5e-11

save = 0 # Dummy counter to save results
dt_save = 5# Save results after dt_save number of iterations

# Time execution time
start = timeit.default_timer()
print("------------------------------------------")
print("Case: " + sim_attempt)
print("------------------------------------------")

while t <= Time and dt > 0.0:
    print("t = ", t)
    print("dt = ", dt)
    

    k = Constant(dt)
    w_, no_iterations, converged, kappa, n_surf = solve_NS_monolithic(mesh, bmf, surface_marker, left_marker, right_marker, bottom_marker, w_n, k, mu, rho, sigma, A_, f, scale)

    (u_, p_) = w_.leaf_node().split()

    # print Courant number
    print("Courant Number = ", (u_.vector().max()*dt)/mesh.hmin())

    # CFL condition: Courant < 1 -> dt < 1*dx/u
    # dt = mesh.hmin()/u_.vector().max() * 100

    # displacement = surface_displacement(D, u_, k, boundary_markers, surface_marker, right_marker, left_marker)
    displacement = surface_displacement(D, W, u_, dt, boundary_markers, bmf, surface_marker, right_marker, left_marker)
    
    ALE.move(boundary_mesh, displacement)
    ALE.move(mesh, boundary_mesh)

    # ALE.move(mesh, displacement)
    mesh.bounding_box_tree().build(mesh)

    w_n.assign(w_)

    # coord = boundary_mesh.coordinates()
    # h_rupture = coord[np.where(coord[:,0]== 0.0)][1][1]

    # Saving results
    # if t==0 or t >= t_sav or t>= Time:
    if save % dt_save == 0:

        # Dimensionless time: h0 = 2*scale (remember the symmetry condition)
        tau_ = t/(pow(L_problem, 2)/nu)

        # Dimensionless velocity
        u_dim = u_ / (mu_/(rho_*L*scale))
        print("------------ Saving results --------------")
        # print("Dimensional h(x=0) = ", h_rupture)
        print("Dimensional time t = ", t)
        print("dt = ", dt)
        print("\n")
        # print("Dimensionless h(x=0) = ", h_rupture/scale)
        print("Dimensionless time tau = ", tau_)
        print("------------------------------------------")
            # t_sav = t + dt_save

        u_.rename("u", "Velocity Field")
        p_.rename("p", "Pressure Field")
        displacement.rename("displacement", "Displacement")
        
        m_pvd << (mesh, t)
        u_pvd << (u_, t)
        p_pvd << (p_, t)
        displacement_pvd << (displacement, t)

        m_dim_pvd << (dimensionless_mesh(mesh, L, H, scale), tau_)
        # u_dim_pvd << (u_dim, tau_)
        b_m_dim_pvd << (dimensionless_mesh(boundary_mesh, L, H, scale), tau_)

        # dim_boundary_mesh << (dimensionless_mesh(boundary_mesh, L, H), t)

    # if not converged:
    #     '''
    #     Diminishing time step if necessary
    #     '''
    #     dt = autoTimestep(no_iterations, dt) #, dt_min = 1e-3, dt_max = 1, min_iter = 3, dt_dt = 3, increment = 2)
    
    # if no_iterations < 5:
    #     '''
    #     Increase time step if possible
    #     '''
    #     dt = autoTimestep(no_iterations, dt)
        
    t += dt
    

    save += 1


stop = timeit.default_timer()
total_time = stop - start
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)

print("Simulation time %dh:%dmin:%ds \n" % (hours, mins, secs))
