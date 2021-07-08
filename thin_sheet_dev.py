import numpy as np
import timeit
import matplotlib.pyplot as plt
from dolfin import *
from mshr import *

import ufl

# ===================== CLASSES ==================
class Left(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        # return near(x[0], -(L+2*l_extra)/2) # and on_boundary
        return near(x[0], 0.0) # and on_boundary

class Right(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        # return near(x[0], (L+2*l_extra)/2) # and on_boundary
        return near(x[0], (L+2*l_extra)) # and on_boundary

class Bottom(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return near(x[1], 0.0)# and on_boundary

class Top(SubDomain): # This will later on become a free surface
    def __init__(self):
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return near(x[1], H*(1+ampl))


class SurfL(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        # return near(x[0], -(L+2*l_extra)/2) and near(x[1], ysl)
        # return near(x[0], xsl) and near(x[1], ysl)
        return near(x[1], ysl) and between(x[0], (0, xsl))
        # return x[0] >= xsl and near(x[1], ysl)


class SurfR(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        # return near(x[0], (L+2*l_extra)/2) and near(x[1], ysr)
        # return near(x[0], xsr) and near(x[1], ysr)
        return near(x[1], ysr) and between(x[0], (xsr, (L+2*l_extra)))
        # return x[0] <= xsr and near(x[1], ysr)



class normal_u(UserExpression):
    '''
    UserExpression to obtain normal velocity vector from normal vector.
    As n = FacetFunction(mesh) is defined on the facets, one must first 
    approximate the normal vector on the degrees of freedom of the mesh.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def eval(self, values, x):
        n_eval = n_surf_dof(x)
        u_eval = u0(x)
        un = (u_eval[0]*n_eval[0] + u_eval[1]*n_eval[1]) # magnitude
        values[0] = un * n_eval[0] # x component of normal vector
        values[1] = un * n_eval[1] # y component of normal vector

    def value_shape(self):
        return (2,)

class tangent_u(UserExpression):
    '''
    UserExpression to obtain normal velocity vector from normal vector.
    As n = FacetFunction(mesh) is defined on the facets, one must first 
    approximate the normal vector on the degrees of freedom of the mesh.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def eval(self, values, x):
        t_eval = t_surf_dof(x)
        u_eval = u0(x)
        ut = (u_eval[0]*t_eval[0] + u_eval[1]*t_eval[1]) # magnitude
        values[0] = ut * t_eval[0] # x component of normal vector
        values[1] = ut * t_eval[1] # y component of normal vector

    def value_shape(self):
        return (2,)

# ===================== FUNCTIONS ==================

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
    x[:,1] =  x[:,1]/((2*H)*scale)

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

def save_results(u_, p_, displacement, mesh, boundary_mesh, n_surf, t_surf, L, H, L_problem, scale, rho_, mu_, nu, t, dt):

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
    n_surf.rename("n", "Normal")
    t_surf.rename("t", "Tangent")
    displacement.rename("displacement", "Displacement")

    # Save dimensional variables
    m_pvd << (mesh, t)
    u_pvd << (u_, t)
    p_pvd << (p_, t)
    n_pvd << (n_surf, t)
    t_pvd << (t_surf, t)
    displacement_pvd << (displacement, t)

    # Save dimensionless variables
    m_dim_pvd << (dim_mesh, tau_)
    # u_dim_pvd << (u_dim, tau_)
    b_m_dim_pvd << (dim_bmesh, tau_)

def sim_log(path, sim_attempt, H, L, l_extra, scale, L_problem, CASE_FACTOR, amplitude, N_surface, N_domain, N_elem, sigma_, A_, mu_, rho_, g, THICKNESS_CRITERION):

    # Create dictionary
    d = {
        "Case ": sim_attempt,
        "Path ": path,
        "Scale of the problem [m]": scale,
        "Mean thickness [m]": H*scale,
        "Perturbation length [m]": L*scale,
        "S/A ": CASE_FACTOR,
        "Characteristic length [m]": L_problem,
        "Auxiliar length ": l_extra,
        "Initial perturbation amplitude ": amplitude,
        "Initial thickness [m] ": (H-amplitude)*scale,
        "Number of elements": N_elem,
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

def divN(nN):
    # div(n)
    div_n = inner(Identity(2),as_tensor([[nN[0].dx(0), 0],
                                         [0, nN[1].dx(1)]]))
    return div_n 

def curvature(mesh, ds, marker):
    '''
    This function takes the mesh, boundary and boundary marker as inputs and returns 
    the curvature (kappa) and normal unit vector of the respective boundary.
    '''

    V = VectorFunctionSpace(mesh, "CG", 2)
    # C = FunctionSpace(mesh, "CG", 1)

    # Projection of the normal vector on P2 space
    u = TrialFunction(V)
    v = TestFunction(V)
    # l = TestFunction(C)
    n = FacetNormal(mesh)

    # a = inner(u, v('+'))*ds(marker)
    # L = inner(n, v('+'))*ds(marker)
    a = inner(u, v)*ds(marker)
    L = inner(n, v)*ds(marker)

    # Solve system
    A = assemble(a, keep_diagonal=True)
    b = assemble(L)
    A.ident_zeros()
    nNorm = Function(V)

    solve(A, nNorm.vector(), b)

    kappa = - divN(nNorm/sqrt(dot(nNorm,nNorm)))

    return kappa, nNorm

def interpolate_nt_dofs(mesh, nt):
    '''
    This function approximates the normal/tangent vector values on the degrees of freedom of the
    mesh.
    '''
    # Create suitable function space to project the normal vector
    V = VectorFunctionSpace(mesh, "CG", 2)

    ua_ = TrialFunction(V)
    va_ = TestFunction(V)
    a = inner(ua_,va_)*ds
    l = inner(nt, va_)*ds
    A = assemble(a, keep_diagonal=True)
    L = assemble(l)

    A.ident_zeros()
    nt_dof = Function(V)

    # solve system 
    solve(A, nt_dof.vector(), L) # nh is the normal vector approximated on dofs

    return nt_dof

def DD(u):
    '''
    Rate-of-deformation tensor for bulk
    '''
    return sym(nabla_grad(u))

def TT(u, p, mu, I):
    '''
    Cauchy stress tensor
    '''
    return 2*mu*DD(u) - p*I

def IST(sigma, kappa, n):
    '''
    Interface stress tensor
    '''
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

def SurfProj(u, n):
    '''
    Projects entity u into the surface.
    '''
    return (Identity(u.ufl_shape[0]) - outer(n, n)) * u

def projected_velocity(u, nt):


    ux, uy = u.split(deepcopy=True)
    ntx, nty = nt.split(deepcopy=True)

    un = ux.vector()[:]*ntx.vector()[:] + uy.vector()[:]*nty.vector()[:]

    projected_u = Function(VS)
    pux, puy = projected_u.split(deepcopy=True)
    pux.vector()[:] = un*(ntx.vector()[:])
    puy.vector()[:] = un*(nty.vector()[:])

    assign(projected_u.sub(0), pux)
    assign(projected_u.sub(1), puy)

    return projected_u
# ===================== MAIN ==================

# ===================== LOCAL PATH =====================================
# local_path = "/media/heitorvc/Simulations/Mestrado/results/reformulated/viscous_surface3/"
local_path = "/mnt/d/dissertation/viscous_surface_results/order_reduction/"
# local_path = "/home/heitorvc/thin_sheets/passive/"


# ==================== CONTROL PARAMETERS ===========================

H = 5 # Initial height
scale = 1e-6 # problem scale
ampl = 0.2 # Amplitude factor of perturbation in terms of H
l_coeff = 10 # twice this value

# CASE_FACTOR = S/A -> S/A = 1/pi² equals Ida and Miksis
CASE_FACTOR = 1/pow(np.pi, 2)

Courant = 100 # Fixed Courant to speed up simulation (counting on convergence with more iterations)

sigma_ = 5e-2 # Interfacial tension

THICKNESS_CRITERION = 0.1*H*scale # final thin sheet thickness to end simulation

# ==================== DOMAIN PARAMETERS ===========================

L = 2*l_coeff*H# Initial Length
l_extra = 20*L/(2*l_coeff) # Extra length
L_problem = L*scale # Initial perturbation characteristic length

amplitude = ampl*H # Initial perturbation amplitude

N_domain = 500 # Mesh resolution / number of points on the surface

# ==================== INTERMOLECULAR PARAMETERS ===========================

# A_ = (2*sigma_ * np.pi * (H*scale)**4) / (CASE_FACTOR * ((L*scale)**2) ) # Ida and Miksis (1995)
# A_ = (sigma_ * np.pi * (H*scale)**4) / (CASE_FACTOR * ((L*scale)**2) ) # Bazzi and Carvalho (2019)
A_ = 0
# ==================== INTERFACIA VISCOSITY PARAMETERS ===========================
k_s = Constant(1e-3) # interfacial dilatational viscosity
mu_s = Constant(1e-3) # interfacial shear viscosity

# ==================== SOLVER PARAMETERS ===========================
# parameters["form_compiler"]["quadrature_degree"] = 4 # Quadrature integration points
# plt.ylim(-5*H*scale, 10*H*scale ) # Plotting limit

absTol = 1e-14          # absolute tolerance: residual value
relTol = 1e-16          # relative tolerance: change with respect to previous
maxIter =   40          # Maximum iterations for non-linear solver
nlinSolver = 'newton'   # Non-Linear Solver(Coupled Pressure/Velocity)
linSolver = 'mumps'     # Linear Solver(Concentration)
alpha = 0.9             # relaxation

# ==================== SAVING PARAMETERS ===========================

dt_save = 1 # Save results after dt_save number of iterations
MaxSave = 50000 # Max number of saved data

# ==================== BULK PARAMETERS ===========================

mu_ = 0.1 # kinematic viscosity
rho_ = 1000 # Density
nu = mu_/rho_

g = 0 # Gravity
mu  = Constant(mu_) # kinematic viscosity
rho = Constant(rho_) # Density
sigma = Constant(sigma_) # Surface tension
f   = rho*Constant((0, - g)) # Body force

# ==================== CASE PATH ===========================

sim_attempt = "viscous_H_"+str(H)+"_"+str(scale)+"_L_"+str(2*l_coeff)+"H_l_extra_"+str(l_extra)+\
    "_sigma_"+str(sigma_)+"_Atild_"+str(A_)+"_Co_"+str(Courant)+"_alpha_"+str(alpha)+\
    "_SA_"+str(CASE_FACTOR)+"_rho_"+str(rho_)+"_mu_bulk_"+str(mu_)+\
    "_order_reduced" + "/"#\
    # "_N_elem_"+str(N_domain)+"_"+ str(mesh.num_cells()) + "_/"

save_index = 'Lc_'+str(2*l_coeff)+'_Hc'

## ==================== CREATE MESH AND BOUNDARY MESH ===========================

channel = Rectangle(Point(0.0, 0.0), Point(L+2*l_extra, H*(1+ampl)))
mesh = generate_mesh(channel, N_domain)

bmf = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

boundary_mesh = BoundaryMesh(mesh, "exterior", True)
boundary_markers = MeshFunction("size_t", boundary_mesh, 0)

## ==================== MARK MESH AND BOUNDARY MESH ===========================

# Assure the free surface is marked
surface_marker = 1 # Surface tag

bmf.set_all(0)
boundary_markers.set_all(0)

# Boundary tags
right_marker = 2
bottom_marker = 3
left_marker = 4

Right().mark(bmf, right_marker)
Bottom().mark(bmf, bottom_marker)
Left().mark(bmf, left_marker)
Top().mark(bmf, surface_marker)

Right().mark(boundary_markers, right_marker)
Bottom().mark(boundary_markers, bottom_marker)
Left().mark(boundary_markers, left_marker)
Top().mark(boundary_markers, surface_marker)

# # get surface extremities coordinates
ysl = get_height(mesh, 0.0, which_side=-1) 
ysr = get_height(mesh, 0.0, which_side= 1) 

xsl = (L+2*l_extra)*0.001
xsr = (L+2*l_extra) - xsl

# Surface extremities
surf_left = 11
surf_right = 12

SurfL().mark(bmf, surf_left)
SurfR().mark(bmf, surf_right)

## ==================== APPLY SURFACE INITIAL SHAPE ===========================

D = VectorFunctionSpace(boundary_mesh, 'Lagrange', 1)
ic = Function(D)

# Smooth sine wave
initial_shape = Expression(("0", "(1/(1+ exp(-k*(x[0]-(le)))))*(amp*(1*cos(2*pi*x[0]/L) -1)*(1- 1/(1+ exp(-k*(x[0] -(L+le))))) )"), L = L, amp = amplitude, k=.08, le = l_extra, degree = 1)

# Aplly bc in displacement vector
bc0 = DirichletBC(D, initial_shape, boundary_markers, surface_marker)
bc0.apply(ic.vector())

# Move boundary mesh and mesh according to initial displacement
ALE.move(boundary_mesh, ic)
ALE.move(mesh, boundary_mesh)

# Update dofs in bounding box tree
boundary_mesh.bounding_box_tree().build(boundary_mesh)
mesh.bounding_box_tree().build(mesh)

center_mesh(mesh) # Center domain in x = 0
center_mesh(boundary_mesh)

# Update dofs in bounding box tree
boundary_mesh.bounding_box_tree().build(boundary_mesh)
mesh.bounding_box_tree().build(mesh)

# Rescale mesh and boundary_mesh to problem scale
mesh.scale(scale)
boundary_mesh.scale(scale)

# Update dofs in bounding box tree
boundary_mesh.bounding_box_tree().build(boundary_mesh)
mesh.bounding_box_tree().build(mesh)

# ==================== SAVE LOG ===========================

string = sim_log(local_path, sim_attempt, H, L, l_extra, scale, L_problem,
        CASE_FACTOR, amplitude, 100 , N_domain, mesh.num_cells(),
        sigma_, A_, mu_, rho_, g,
        THICKNESS_CRITERION)

## ==================== CREATE PARAVIEW FILES ===========================

# Dimensional
u_pvd = File(local_path + sim_attempt +"u_"+str(save_index)+".pvd")
p_pvd = File(local_path + sim_attempt +"p_"+str(save_index)+".pvd")
m_pvd = File(local_path + sim_attempt +"m_"+str(save_index)+".pvd")
displacement_pvd = File(local_path + sim_attempt + "displacement_"+str(save_index)+".pvd")

n_pvd = File(local_path + sim_attempt +"n_"+str(save_index)+".pvd")
t_pvd = File(local_path + sim_attempt +"t_"+str(save_index)+".pvd")

# Dimensionless
m_dim_pvd = File(local_path + sim_attempt +"/dimensionless/m_dim_"+str(save_index)+".pvd")
b_m_dim_pvd = File(local_path + sim_attempt +"/dimensionless/bm_dim_"+str(save_index)+".pvd")

## ==================== FUNCTION SPACES ===========================

# Create function space for tangent, normal and velocity
VS = VectorFunctionSpace(mesh, "CG", 2)

Uel = VectorElement('Lagrange', mesh.ufl_cell(), 2)
Pel = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
UPel = MixedElement([Uel,Pel])
W = FunctionSpace(mesh, UPel)

w_n = Function(W)

## ==================== TIME EVOLUTION LOOP ===========================

Time = 1e-2 # Total time
t = 0 # initial time
dt = 5e-10 # initial dt

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

    k = Constant(dt) # Time-step

    dx = Measure('dx', domain=mesh)
    ds = Measure('ds', domain=mesh, subdomain_data = bmf)

    Uel = VectorElement('Lagrange', mesh.ufl_cell(), 2)
    Pel = FiniteElement('Lagrange', mesh.ufl_cell(), 1)

    UPel = MixedElement([Uel,Pel])

    W = FunctionSpace(mesh, UPel)

    # Trial and Test functions
    (dw) = TrialFunction(W)
    (v,q) = TestFunctions(W)
    w = Function(W)
    (u, p) = (as_vector((w[0], w[1])), w[2])

    d = mesh.geometry().dim()
    I = Identity(d)
    n = FacetNormal(mesh)
    x = SpatialCoordinate(mesh)

    t_facet = ufl.perp(n)

    ############## GET CURVATURE ##############

    uu = TrialFunction(VS)
    vv = TestFunction(VS)

    aa = inner(uu, vv)*ds(surface_marker) 
    LL = inner(n, vv)*ds(surface_marker) 
    AA = assemble(aa, keep_diagonal=True)
    bb = assemble(LL)
    AA.ident_zeros()
    n_surf_dof = Function(VS)

    solve(AA, n_surf_dof.vector(), bb)

    # Get tangent vector evaluated on the free surface
    t_surf_dof = interpolate_nt_dofs(mesh, 
                   ufl.as_vector([n_surf_dof[1], - n_surf_dof[0]]))

    # Normal and tangent magnitudes
    n_mag = sqrt(dot(n_surf_dof,n_surf_dof))
    t_mag = sqrt(dot(t_surf_dof,t_surf_dof))

    kappa = - div(n_surf_dof/n_mag)

    # Introduce surfa values in the tangent vector  
    leftbcx = DirichletBC(VS, Constant((1,0)), bmf, surf_left)
    rightbcx = DirichletBC(VS, Constant((1,0)), bmf, surf_right)
    leftbcy = DirichletBC(VS, Constant((0,1)), bmf, surf_left)
    rightbcy = DirichletBC(VS, Constant((0,1)), bmf, surf_right)

    leftbcx.apply(t_surf_dof.vector()); rightbcx.apply(t_surf_dof.vector())
    leftbcy.apply(n_surf_dof.vector()); rightbcy.apply(n_surf_dof.vector())

    # Define Projection tensor
    I_S = I - outer(n_surf_dof, n_surf_dof)
    # I_S = I - outer(n, n)

    # Symmetry boundary condition on bottom boundary
    symmetry = DirichletBC(W.sub(0).sub(1), Constant(0.0), bmf, bottom_marker)
    bcu = [symmetry]

    # Weak form of the momentum equation
    (u0, p0) = w_n.leaf_node().split()

    a1 = rho*dot((u-u0)/k,v)*dx() + alpha *(rho*dot(dot(u ,grad(u) ),v) + inner(TT(u,p,mu, I),DD(v)))*dx() + \
                                                (1-alpha)*(rho*dot(dot(u0,grad(u0)),v) + inner(TT(u0,p0,mu, I),DD(v)))*dx()  # Relaxation

    # Side pressure - naturally imposed
    L1 = - (VdW_sides(A_, mesh, side = -1))*dot(n,v)*ds(left_marker) - (VdW_sides(A_, mesh, side = 1))*dot(n,v)*ds(right_marker) # + inner(rho*g,v)*dx()
    # L1 = 0

    # Tangent velocity and its gradient
    ut = projected_velocity(u0, t_surf_dof)

    # Define normalized tangent
    t_hat = t_surf_dof/t_mag

    gradsut = I_S*grad(ut)*t_hat

    # Normalized gradient of test function projected on the surface tangent vector
    gradsv = I_S*grad(v)*t_hat

    ################# PASSIVE COMPONENTS

    ##### Order reduction
    # integral over the surface
    b_int = - sigma*inner(gradsv, t_hat)*ds(surface_marker) 

    # integral evaluated on surface extremities
    b_int += sigma*(inner(t_hat,v)*ds(surf_right) - inner(t_hat,v)*ds(surf_left))

    # Van der Waals forces
    # vdw = - VdW(A_, x)*dot(n, v)*ds(surface_marker) -VdW(A_, x)*dot(n, v)*ds(surf_left) -VdW(A_, x)*dot(n, v)*ds(surf_right)
    vdw = - VdW(A_, x)*dot(n_surf_dof, v)*ds(surface_marker) +\
          - VdW(A_, x)*dot(n_surf_dof, v)*ds(surf_left) +\
          - VdW(A_, x)*dot(n_surf_dof, v)*ds(surf_right)

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

    prmU['nonlinear_solver'] = nlinSolver
    prmU['newton_solver']['absolute_tolerance'] = absTol
    prmU['newton_solver']['relative_tolerance'] = relTol
    prmU['newton_solver']['maximum_iterations'] = maxIter
    prmU['newton_solver']['linear_solver'] = linSolver


    # ================ Solve system  ============================================== #
    try:
        (no_iterations,converged) = solverU.solve()
    except:
        converged = False
        no_iterations = maxIter
        print("Convergence failed")
        w = w_n
    
    (u_, p_) = w.leaf_node().split() # split variables

    # Calculate displacement vector
    displacement = surface_displacement(D, u_, n, n_surf_dof, dt, boundary_markers, bmf, 
                                        surface_marker, bottom_marker, right_marker, left_marker)
    
    w_n.assign(w) # Assign state variables for next iteration
    (u_n, p_n) = w_n.leaf_node().split() # split variables

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
                     mesh, boundary_mesh, n_surf_dof, t_surf_dof, 
                     L, H, L_problem, scale, 
                     rho_, mu_, nu, t, dt)  
        
        print("-- Saving thickness evolution timeseries --")
        np.save(local_path + sim_attempt + "thickness.npy", 
                np.asarray(thickness))

        break

    # ================ Save results =================================== #
    if save % dt_save == 0:

        # Save thickness profile
        thickness.append([t, h_])

        # Save pvd files     
        save_results(u_, p_, displacement,
                     mesh, boundary_mesh, n_surf_dof, t_surf_dof,
                     L, H, L_problem, scale, 
                     rho_, mu_, nu, t, dt)    


    # ================ Update surface position ===================================#
    
    ALE.move(boundary_mesh, displacement) # Move boundary mesh
    ALE.move(mesh, boundary_mesh) # Move mesh

    mesh.bounding_box_tree().build(mesh) # Update mesh bounding box for next iteration

    # ================ Assign values for next iteration ===================================#
    
    dt = (Courant*mesh.hmin())/u_.vector().max() # Update time step: Fixed Courant number to control time resolution near rupture
    
    t += dt # update time    
    save += 1 # update save counter

# ================ Simulation ended ===================================#
print("===== Simulation finished ====== \n")
# np.save(local_path + sim_attempt + "thickness.npy", np.asarray(thickness))

stop = timeit.default_timer()
total_time = stop - start
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)

print("Simulation time %dh:%dmin:%ds \n" % (hours, mins, secs))