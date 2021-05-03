import numpy as np
import timeit
import matplotlib.pyplot as plt
import plotext as tplt
from dolfin import *
from mshr import *

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

def get_height(mesh, x0, which_side=0):
    '''
    Return y-coordinate with highest value given x = x0.
    which_side = 0 -> any given location
    which_side = 1 -> returns max height for outermost right x value
    which_side = -1 -> returns max height for outermost left x value
    '''
    # x0 = 0.0 # Constriction x coordinate
    # Get mesh coordinates
    x = mesh.coordinates()

    if which_side == 1: # x coordinate of right boundary
        x0 = max(x[:,0])
    
    if which_side == -1: # x coordinate of left boundary
        x0 = min(x[:,0])

    # Get indices that match coordinate such as x(mesh) = x0 from mesh.coordinates()
    indices = [idx for idx, value in enumerate(x) if np.isclose(value[0], x0, atol=1.e-8)]
    ymax = 0 # initialize counter for y value
    idx = 0 # initialize counter for index
    for index in indices: # get index of value with max y value
        
        if x[index][1] >= ymax:
            ymax = x[index][1]
            idx = index
    
    return x[idx][1]

def save_results(u_, p_, displacement, mesh, boundary_mesh, L, H, L_problem, scale, rho_, mu_, nu, t, dt):

    tau_ = t/(pow(L_problem, 2)/nu) # dimensionless time    

    # dimensionless mesh and boundary mesh
    dim_mesh = dimensionless_mesh(mesh, L, H, scale)
    dim_bmesh = dimensionless_mesh(boundary_mesh, L, H, scale)
    
    # dimensionless velocity and pressure
    # u_dim = u_ / (mu_/(rho_*L*scale))


    print("------------ Saving pvd files --------------")
    print("Dimensional: t = {}     dt = {}".format(t, dt))
    print("Dimensionless t = {}".format(tau_))

    # Rename variables
    u_.rename("u", "Velocity Field")
    p_.rename("p", "Pressure Field")
    displacement.rename("displacement", "Displacement")

    # Save dimensional variables
    m_pvd << (mesh, t)
    u_pvd << (u_, t)
    p_pvd << (p_, t)
    displacement_pvd << (displacement, t)

    # Save dimensionless variables
    m_dim_pvd << (dim_mesh, tau_)
    # u_dim_pvd << (u_dim, tau_)
    b_m_dim_pvd << (dim_bmesh, tau_)

def sim_log(path, sim_attempt, H, L, l_extra, scale, L_problem, amplitude, N_surface, N_domain, sigma_, A_, mu_, rho_, g, THICKNESS_CRITERION):

    # Create dictionary
    d = {
        "Case ": sim_attempt,
        "Path ": path,
        "Mean thickness [m]": H,
        "Scale of the problem [m]": scale,
        "Characteristic length [m]": L_problem,
        "Auxiliar length ": l_extra,
        "Initial perturbation amplitude ": amplitude,
        "Initial thickness [m] ": (H-amplitude)*scale,
        "N_surface ": N_surface,
        "N_domain ": N_domain,
        "Simulation minimum thickness (end criterion) ": THICKNESS_CRITERION,
        "Interface tension [N/m] ": sigma_,
        "Hamaker constant [J] ": A_,
        "Dynamic viscosity [Pa.s] ": mu_,
        "Density [Kg/m³] ": rho_,
        "Gravity [m/s²] ": g,
    }
    # print values
    print('\n')
    string = " ============ Simulation setup =========== \n"

    for key, value in d.items():
        print('{0} = {1}'.format(key, value))
        string += key +': '+ str(value) + '\n'
        
    print('\n')
    
    return string

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
    absTol = 1e-13          # absolute tolerance: residual value
    relTol = 1e-16          # relative tolerance: change with respect to previous
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
    coords = mesh.coordinates()


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
    L1 = - (VdW_sides(A, mesh, side = -1))*dot(n,v)*ds(left_marker) - (VdW_sides(A, mesh, side = 1))*dot(n,v)*ds(right_marker) # + inner(rho*g,v)*dx()
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

def VdW_sides(A, mesh, side):
    '''
    Van der Waals forces on surface
    A = Hamacker constant for the fluid
    
    side = 1 -> right boundary
    side = -1 -> left boundary
    '''
    
    y_side = get_height(mesh, x0 = 0.0, which_side=side)
    

    return A/(2*np.pi*pow(y_side,3))
    # return A/(2*np.pi*pow(max(mesh.coordinates()[:,1]),3))

# ------------------------------------------------------------------------------#

'''
===== Validation test 1 ===========

Considering a senoidal shape as initial condition for the interface;
The validation cases are Ida and Miksis (1994), Bazzi and Carvalho (2019), as:
    
    S/A = 1/pi²

    S = (sigma*rho*H) / (3*mu²)
    A = (Ã*rho*Lc²) / (6*pi*H³*mu²)

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
# =============================================================================================
# local_path = "/media/heitorvc/Simulations/Mestrado/results/passive_surface/"
local_path = "/mnt/d/dissertation/passive_surface_results/"

sim_attempt = "ida_miksis_setup_update_vdw_bc_Laux_2/" 
save_index = 3

# ========== Domain parameters ==============
H = .5 # Initial heights
L = 2*np.pi # Initial Length
l_extra = 2 # Extra length
amplitude = 0.1 # Initial perturbation amplitude
N_surface = 200 # Surface discretization points
N_domain = 300 # Mesh resolution / number of points on the surface

# ========== Interface parameters ==============
sigma_ = 3.08e-5 # Interfacial tension
A_ = 4.77e-16# Hamaker constant 
Courant = 500 # Fixed Courant to speed up simulation (counting on convergence with more iterations)

# ========== Bulk parameters ==============
mu_ = 0.001 # kinematic viscosity
rho_ = 1000 # Density
nu = mu_/rho_

g = 0 # Gravity
mu  = Constant(mu_) # kinematic viscosity
rho = Constant(rho_) # Density
sigma = Constant(sigma_) # Surface tension
f   = rho*Constant((0, - g)) # Body force

# Scale parameters
scale = 1e-6 # problem scale // Mean thickness for Erneux and Davis

'''
Erneux and Davis (1993) states that the characteristic length for the dimensionless time 
equals the mean thickness, whereas Ida and Miksis (1994) uses the perturbation wavelength
to obtain the dimensionless time.
'''
L_problem = L*scale # Initial perturbation characteristic length

# final thin sheet thickness to end simulation
THICKNESS_CRITERION = 0.05*scale 

# Saving parameters
dt_save = 20 # Save results after dt_save number of iterations
MaxSave = 10000 # Max number of saved data

# Save log
string = sim_log(local_path, sim_attempt, H, L, l_extra, scale, L_problem,
        amplitude, N_surface, N_domain,
        sigma_, A_, mu_, rho_, g,
        THICKNESS_CRITERION)

# with open(local_path+sim_attempt+"log.txt","w+") as file:
#     file.write(string)
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
dt = 5e-9 # initial dt

save = 0 # Dummy counter to save results

# Store constriction thickness evolution
thickness = []

# Time execution time
start = timeit.default_timer()
print("------------------------------------------")
print("Case: " + sim_attempt)
print("------------------------------------------")

while t <= Time and dt > 0.0 and save < MaxSave:
    print("t = ", t)
    print("dt = ", dt)
    print("Iterations: ", save)
    
    # ================ Solve system  ============================================== #
    k = Constant(dt) # Time-step

    # Solve system 
    w_, no_iterations, converged, kappa, n_surf, n = solve_NS_monolithic(mesh, bmf, 
                                                                        surface_marker, left_marker, right_marker, bottom_marker, 
                                                                        w_n, k, mu, rho, sigma, A_, f, scale)

    (u_, p_) = w_.leaf_node().split() # split variables

    # Calculate displacement vector
    displacement = surface_displacement(D, u_, n, n_surf, dt, boundary_markers, bmf, 
                                        surface_marker, bottom_marker, right_marker, left_marker)
    
    # ================ Check Simulation status =================================== #

    h_ = get_height(mesh, x0 = 0.0, which_side=0) # Get constriction thickness

    # Print values
    print("\n")
    print("thin sheet thickness: {}".format(h_))
    print("\n")

    # Simulation end criterion
    if not converged or h_ < THICKNESS_CRITERION: 
        # Save last time step
        save_results(u_, p_, displacement,
                     mesh, boundary_mesh, 
                     L, H, L_problem, scale, 
                     rho_, mu_, nu, t, dt)  
        break

    # ================ Save results =================================== #
    if save % dt_save == 0:

        # Save thickness profile
        thickness.append([t, h_])

        # Save pvd files     
        save_results(u_, p_, displacement,
                     mesh, boundary_mesh, 
                     L, H, L_problem, scale, 
                     rho_, mu_, nu, t, dt)    

    # ================ Update surface position ===================================#
    
    ALE.move(boundary_mesh, displacement) # Move boundary mesh
    ALE.move(mesh, boundary_mesh) # Move mesh

    mesh.bounding_box_tree().build(mesh) # Update mesh bounding box for next iteration

    # ================ Assign values for next iteration ===================================#
    
    dt = (Courant*mesh.hmin())/u_.vector().max() # Update time step: Fixed Courant number to control time resolution near rupture
    w_n.assign(w_) # Assign state variables for next iteration

    t += dt # update time    
    save += 1 # update save counter

# ================ Simulation ended ===================================#
print("-- Saving thickness evolution timeseries --")
np.save(local_path + sim_attempt + "thickness.npy", np.asarray(thickness))

stop = timeit.default_timer()
total_time = stop - start
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)

print("Simulation time %dh:%dmin:%ds \n" % (hours, mins, secs))
