import numpy as np
import timeit
import matplotlib.pyplot as plt
from dolfin import *
from mshr import *

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
    channel = Rectangle(Point(0,0), Point(L, H))
    m = generate_mesh(channel, Nx)
    # Create rectangle mesh
    # m = RectangleMesh(Point(0, 0), Point(L, H), Nx, Ny)
    # Get coordinates
    x = m.coordinates()
    # refine near boundary
    x[:,1] = np.arctan(np.pi*(x[:, 1])) / np.arctan(np.pi)

    # downscale coordinate
    # x[:,0] = x[:,0]*scale
    # x[:,1] = x[:,1]*scale  

    return m

def mesh_polygon(H, L, l_extra, N_surface, N_domain, amplitude):
    '''
    Domain creation from boundary prescribed by a polygon. 
    Sinoidal shape on top boundary (free surface). 
    mshr library requires a polygon list ordered in counter-clockwise direction
    '''
    
    domain = [Point(L,0), Point(L+l_extra, 0)]
    
    x0 = L
    dx = L/N_surface

    for i in range(N_surface +1):
        
        ys = amplitude*np.cos(x0) + H # Create free surface points

        # Check if it is the first run to append the coordinates
        # of the right corner in the list
        if i == 0:
            domain.append(Point(L+l_extra, ys))
        
        domain.append(Point(x0,ys)) # append free surface points
        # print((x0, ys))
        x0 -= dx # update x0

    # Append coordinates of top and bottom left corners
    domain.append(Point(-l_extra,ys))
    domain.append(Point(-l_extra,0))

    # Draw polygon
    domain = Polygon(domain)
    # Generate mesh
    mesh = generate_mesh(domain,int(N_domain))

    return mesh

def shift_mesh(mesh, L):
    '''
    Shifts mesh to the left
    '''
    x = mesh.coordinates()
    x[:,0] =  x[:,0] - L/2

    mesh.bounding_box_tree().build(mesh)

def center_mesh(mesh):
    '''
    Centers mesh in x = 0. 
    '''
    # Extract moordinate points
    x = mesh.coordinates()
    # assure the mesh starts at zero
    x[:,0] =  x[:,0] + abs(min(x[:,0]))
    # center mesh
    x[:,0] =  x[:,0] - abs(max(x[:,0]))/2
    
    # Rebuild  tree
    mesh.bounding_box_tree().build(mesh)

def dimensionless_mesh(mesh, L, H, scale):
    '''
    Returns dimensionless mesh/boundary mesh
    '''
    mesh1 = Mesh(mesh)
    x = mesh1.coordinates()
    
    x[:,0] =  x[:,0]/(((L)/2)*scale)
    # x[:,0] =  x[:,0]*(scale*2) # scale up mesh
    x[:,1] =  x[:,1]/(scale)

    mesh1.bounding_box_tree().build(mesh1)

    return mesh1

def surface_displacement(D, u, n, n_surf, dt, bmf, domain_bmf, surface_marker, bottom_marker, right_marker, left_marker):
    '''
    This function obtains the surface displacement as
               displacement = u_normal*dt
    u_normal is obtained by applying the kinematic boundary condition on the surface
    u_surface . n = u . n = u_normal
    '''
    # Define displacement 
    displacement = Function(D)
    displacement.set_allow_extrapolation(True)

    ## Get normal velocity vector
    # First project normal vector on mesh dofs
    # nh = normal_dofs(mesh, n)

    # Then, interpolate normal velocity vector on boundary using UserExpression
    u_surf = Function(D)
    u_surf.set_allow_extrapolation(True)
    u_surf.interpolate(normal_u()) # Using a UserExpression

    # Create DirichletBC for surface displacement
    s_ = DirichletBC(D, dt*u_surf, bmf, surface_marker)

    u_boundary = Function(D)
    u_boundary.set_allow_extrapolation(True)
    u_boundary.interpolate(u) # Using a UserExpression
        
    # Apply zero to velocity x-component 
    ux, uy = u_boundary.split(deepcopy=True)
    ux.vector()[:] = 0
    assign(u_boundary.sub(0), ux) # assign modified values

    # Create DirichletBC for left/right displacements
    l_ = DirichletBC(D, dt*u_boundary, bmf, left_marker)
    r_ = DirichletBC(D, dt*u_boundary, bmf, right_marker)
    
    s_.apply(displacement.vector()) 
    l_.apply(displacement.vector())
    r_.apply(displacement.vector())

    return displacement

def gaussian_pulse_surface(D, mesh, boundary_mesh, boundary_markers, surface_marker, ls, sigma = .2, amplitude= - 0.05):

    initial_deform = Function(D)
    ampl = amplitude
    sigma_ = sigma
    mu_ = L/2
    L1 = ls
    L2 = L1 + 2*np.pi
    ymax = 0
    ymin = -0.1 # initial thicknes - Ida and Miksis (1996)
    sin_shape = Expression(("0", "(ymax - ymin)*cos(x[0])+ ymin"), ymax = ymax, ymin = ymin, degree=2)
    # cos_shape = Expression(("0", "1 - *cos(pi*x[0])"), L=L, degree=2)

    # pulse = Expression(("0", "ampl*(1/(sigma*sqrt(2*pi)))*exp(-((x[0]-mu)*(x[0]-mu))/(2*sigma*sigma))"), ampl=ampl, mu = mu_, sigma = sigma_, degree=2)
    
    pulse = Expression(("0", "x[0] > L1 ? x[0] < L2 ? (ymax - ymin)*cos(x[0]-L1)+ ymin: ymax: ymax"), ymax = ymax, ymin = ymin, L1 = L1, L2 = L2, degree=2)
   

    # x[0] >= L1 ? x[0] <= L2 ? (ymax - ymin)*cos(x[0])+ ymin: ymax: ymax 
    # x[0]-x[1]-t >= 0 ? -2*(exp(x[0]-x[1]-t)-1)/(exp(x[0]-x[1]-t)-1): -(x[0]-x[1]-t)'



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

def sin_surface1(D, mesh, boundary_mesh, boundary_markers, surface_marker, ampl=0.1):
    
    initial_deform = Function(D)
    ymax = 0
    ymin = -0.1 # initial thicknes - Ida and Miksis (1996)
    sin_shape = Expression(("0", "(ymax - ymin)*cos(x[0] -2)+ ymin"), ymax = ymax, ymin = ymin, degree=2)

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

def normal_dofs(mesh, n):
    '''
    This function approximates the normal vector values on the degrees of freedom of the
    mesh.
    '''
    # Create suitable function space to project the normal vector
    V = VectorFunctionSpace(mesh, "CG", 2)

    ua_ = TrialFunction(V)
    va_ = TestFunction(V)
    a = inner(ua_,va_)*ds
    l = inner(n, va_)*ds
    A = assemble(a, keep_diagonal=True)
    L = assemble(l)

    A.ident_zeros()
    nh = Function(V)

    # solve system 
    solve(A, nh.vector(), L) # nh is the normal vector approximated on dofs

    return nh

def solve_NS_monolithic(mesh, bmf, surface_marker, left_marker, right_marker, bottom_marker, w0, k, mu, rho, sigma, A, f, scale):
    
    # Solver Parameters
    absTol = 1e-9          # absolute tolerance: residual value
    relTol = 1e-10          # relative tolerance: change with respect to previous
    maxIter =   20         # Maximum iterations for non-linear solver
    nlinSolver = 'newton'   # Non-Linear Solver(Coupled Pressure/Velocity)
    linSolver = 'mumps'     # Linear Solver(Concentration)
    alpha = 0.9             # relaxation

    # metadata = {"quadrature_degree": 3, "quadrature_scheme": "default"}
    # dx = Measure('dx', domain=mesh, metadata = metadata)
    # ds = Measure('ds', domain=mesh, subdomain_data = bmf, metadata = metadata)
    dx = Measure('dx', domain=mesh)
    ds = Measure('ds', domain=mesh, subdomain_data = bmf)

    d = mesh.geometry().dim()
    I = Identity(d)
    n = FacetNormal(mesh)
    x = SpatialCoordinate(mesh)
    coords = mesh.coordinates()


    kappa, n_surf = curvature(mesh, ds, surface_marker)

    Uel = VectorElement('Lagrange', mesh.ufl_cell(), 2)
    Pel = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    # Uel = VectorElement('Quadrature', mesh.ufl_cell(), degree = 2, quad_scheme = 'default')
    # Pel = FiniteElement('Quadrature', mesh.ufl_cell(), degree = 1, quad_scheme = 'default')

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
    L1 = - (VdW_sides(A, coords))*dot(n,v)*ds(left_marker) - (VdW_sides(A, coords))*dot(n,v)*ds(right_marker) # + inner(rho*g,v)*dx()
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
        
    
    return w, no_iterations, converged, kappa, n_surf, n

# Define classes
class Left(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return near(x[0], -(L+2*l_extra)/2) # and on_boundary

class Right(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return near(x[0], (L+2*l_extra)/2) # and on_boundary

class Bottom(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return near(x[1], 0.0)# and on_boundary

# class Top(SubDomain): # This will later on become a free surface
#     def __init__(self):
#         SubDomain.__init__(self)
    
#     def inside(self, x, on_boundary):
#         return near(x[1], H)

class normal_u(UserExpression):
    '''
    UserExpression to obtain normal velocity vector from normal vector.
    As n = FacetFunction(mesh) is defined on the facets, one must first 
    approximate the normal vector on the degrees of freedom of the mesh.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def eval(self, values, x):
        n_eval = n_surf(x)
        u_eval = u_(x)
        un = (u_eval[0]*n_eval[0] + u_eval[1]*n_eval[1]) # magnitude
        values[0] = un * n_eval[0] # x component of normal vector
        values[1] = un * n_eval[1] # y component of normal vector

    def value_shape(self):
        return (2,)

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
    return A/(2*np.pi*pow(x[1],3))

def VdW_sides(A, x):
    '''
    Van der Waals forces on surface
    A = Hamacker constant for the fluid
    '''
    

    return A/(2*np.pi*pow(max(x[:,1]),3))

# ------------------------------------------------------------------------------#

'''
===== Validation test 1 ===========

Considering a senoidal shape as initial condition for the interface;
The validation cases are Ida and Miksis (1994), Bazzi and Carvalho (2019), as:
    
    S/A = 1/pi²

    S = (sigma*rho*H) / (3*mu²)
    A = (Ã*rho*Lc²) / (6*pi*H³*mu²)


'''
# =============================================================================================
sim_attempt = "mesh_test_8/" 
save_index = 8
'''
Mesh test:
0 -> N_surf = 50   N = 50
1 -> N_surf = 100   N = 80
2 -> N_surf = 200   N = 80
3 -> N_surf = 200   N = 100
4 -> N_surf = 200   N = 150
5 -> N_surf = 200   N = 200
6 -> N_surf = 250   N = 200
7 -> N_surf = 250   N = 250
8 -> N_surf = 300   N = 250
9 -> N_surf = 300   N = 300
'''



sigma_ = 0.030396 # Interfacial tension
A_ = 4.77e-14 # Hamaker constant 
Courant = 800 # Fixed Courant to speed up simulation (counting on convergence with more iterations)

# Domain parameters
H = .5 # Initial height
L = 2*np.pi # Initial Length
l_extra = 2 # Extra length
amplitude = 0.1 # Initial perturbation amplitude
N_surface = 300 # Surface discretization points
N_domain = 250 # Mesh resolution / number of points on the surface

# Scale parameters
scale = 1e-6 # problem scale
L_problem = L*scale # Initial perturbation characteristic length

# Saving parameters
dt_save = 50 # Save results after dt_save number of iterations
MaxSave = 10000 # Max number of saved data

## =============================================================================================
## Create polygon mesh
mesh = mesh_polygon(H, L, l_extra, N_surface, N_domain, amplitude)
center_mesh(mesh) # Center domain in x = 0

# Create mesh function to mark the boundaries
bmf = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

# Assure the free surface is marked
surface_marker = 1 # Surface tag
bmf.set_all(surface_marker)

# Boundary tags
right_marker = 2
bottom_marker = 3
left_marker = 4

# Top().mark(bmf, surface_marker)
Right().mark(bmf, right_marker)
Bottom().mark(bmf, bottom_marker)
Left().mark(bmf, left_marker)

# Create boundary mesh from mesh. Used to get the displacement vector
boundary_mesh = BoundaryMesh(mesh, "exterior", True)
boundary_markers = MeshFunction("size_t", boundary_mesh, 0)
# Assure the free surface is marked
boundary_markers.set_all(surface_marker)

# Top().mark(boundary_markers, surface_marker)
Right().mark(boundary_markers, right_marker)
Bottom().mark(boundary_markers, bottom_marker)
Left().mark(boundary_markers, left_marker)

# Rescale mesh and boundary_mesh with problem scale
mesh.scale(scale)
mesh.bounding_box_tree().build(mesh)
boundary_mesh.scale(scale)
boundary_mesh.bounding_box_tree().build(boundary_mesh)

# Create function space for the displacement
D= VectorFunctionSpace(boundary_mesh, "CG", 1)

# Bulk parameters
mu_ = 0.01 # kinematic viscosity
rho_ = 1000 # Density
nu = mu_/rho_

g = 0 # Gravity

mu  = Constant(mu_) # kinematic viscosity
rho = Constant(rho_) # Density
sigma = Constant(sigma_) # Surface tension
f   = rho*Constant((0, - g)) # Body force

local_path = "/media/heitorvc/Simulations/Mestrado/results/passive_surface/"

# Saving files
u_pvd = File(local_path + sim_attempt +"u_"+str(save_index)+".pvd")
p_pvd = File(local_path + sim_attempt +"p_"+str(save_index)+".pvd")
m_pvd = File(local_path + sim_attempt +"m_"+str(save_index)+".pvd")
displacement_pvd = File(local_path + sim_attempt + "displacement_"+str(save_index)+".pvd")

# u_dim_pvd = File(local_path + sim_attempt +"/dimensionless/u_dim.pvd")
m_dim_pvd = File(local_path + sim_attempt +"/dimensionless/m_dim_"+str(save_index)+".pvd")
b_m_dim_pvd = File(local_path + sim_attempt +"/dimensionless/bm_dim_"+str(save_index)+".pvd")

Uel = VectorElement('Lagrange', mesh.ufl_cell(), 2)
Pel = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
UPel = MixedElement([Uel,Pel])
W = FunctionSpace(mesh, UPel)
w_n = Function(W)


Time = 1e-2 # Total time
t = 0 # initial time
dt = 5e-8 # initial dt

save = 0 # Dummy counter to save results

# Time execution time
start = timeit.default_timer()
print("------------------------------------------")
print("Case: " + sim_attempt)
print("------------------------------------------")

while t <= Time and dt > 0.0 and save < MaxSave:
    print("t = ", t)
    print("dt = ", dt)
    print("Iterations: ", save)
    

    k = Constant(dt)
    w_, no_iterations, converged, kappa, n_surf, n = solve_NS_monolithic(mesh, bmf, surface_marker, left_marker, right_marker, bottom_marker, w_n, k, mu, rho, sigma, A_, f, scale)

    (u_, p_) = w_.leaf_node().split()

    if not converged:
        # print("Breakpoint")
        break
 
    displacement = surface_displacement(D, u_, n, n_surf, dt, boundary_markers, bmf, surface_marker, bottom_marker, right_marker, left_marker)
    
    ALE.move(boundary_mesh, displacement)
    ALE.move(mesh, boundary_mesh)

    # ALE.move(mesh, displacement)
    mesh.bounding_box_tree().build(mesh)

    # Fix Courant number in 1000 
    dt = (Courant*mesh.hmin())/u_.vector().max()

    w_n.assign(w_)

    # Saving results

    if save % dt_save == 0:

        # Dimensionless time: h0 = 2*scale (remember the symmetry condition)
        tau_ = t/(pow(L_problem, 2)/nu)

        # dimensionless mesh
        dim_mesh = dimensionless_mesh(mesh, L, H, scale)
        dim_bmesh = dimensionless_mesh(boundary_mesh, L, H, scale)

        # Dimensionless velocity
        # W_Dimensionless = FunctionSpace(dim_mesh, UPel)
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

        m_dim_pvd << (dim_mesh, tau_)
        # u_dim_pvd << (u_dim, tau_)
        b_m_dim_pvd << (dim_bmesh, tau_)

    t += dt
    
    save += 1


stop = timeit.default_timer()
total_time = stop - start
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)

print("Simulation time %dh:%dmin:%ds \n" % (hours, mins, secs))
