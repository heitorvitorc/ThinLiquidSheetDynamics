import numpy as np
import matplotlib.pyplot as plt
import dolfin
from dolfin import *
from mshr import *

def init_surface(mesh, bmesh, bmf, L, bflag):
    '''
    Move specified boundary in a senoidal manner.
    '''
    for i in range(bmesh.num_vertices()):
        if (bmf[i] == bflag):
            bmesh.coordinates()[i,1] -= 0.05*np.sin(4*(np.pi/L)*bmesh.coordinates()[i,0])

    ALE.move(mesh, bmesh)
    mesh.bounding_box_tree().build(mesh)


def readDomains(inPath,inFile):
    # Read .msh File
    fid = open(inPath+inFile+'.msh', 'r')
    # Initialize variables
    found = 0
    finished = 0
    physicalNames = {}
    # Loop througn .msh lines
    for line in fid:
        if '$EndPhysicalNames' in line:
            finished == 1
            break
        elif '$PhysicalNames' in line:
            found = 1
        elif found==1 and finished == 0:
            word=line.split()
            if len(word)==3:
                physicalNames[word[2][1:len(word[2])-1]] = int(word[1])

    return physicalNames

def divK(nN):
    # div(n)
    div_k = inner(Identity(3),as_tensor([[nN[0].dx(0), 0, 0],
                                         [0, 0, 0],
                                         [0, 0, nN[1].dx(1)]]))
    return div_k 


def curvature(mesh, boundaries, ds, marker):
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

def surface_displacement(V, u, dt, boundaries, subdomains):
    '''
    This function obtains the surface displacement as
               displacement = u*dt
    '''
    # Calculate displacement 
    # displacement = u_*dt
    displacement = Function(V)
    # displacement.interpolate(u_*dt)
    b_ = DirichletBC(V, dt*u, boundaries, subdomains['Top'])
    v_left = DirichletBC(V.sub(0), Constant(0), boundaries, subdomains['Left'])
    v_right = DirichletBC(V.sub(0), Constant(0), boundaries, subdomains['Right'])

    b_.apply(displacement.vector())
    v_left.apply(displacement.vector())
    v_right.apply(displacement.vector())

    return displacement

def update_mesh(mesh, displacement, boundaries, subdomains):
    '''
    Updates mesh geometry according to provided displacement. 
    '''
    # Move mesh
    new_mesh = Mesh(mesh) 
    displacement.set_allow_extrapolation(True)
    ALE.move(new_mesh, displacement)
    new_mesh.bounding_box_tree().build(new_mesh)

    # Update subdomains and boundaries
    new_subdomains = MeshFunction('size_t', new_mesh, new_mesh.topology().dim(), new_mesh.domains())
    new_subdomains.set_values(subdomains.array())
    new_boundaries = MeshFunction("size_t", new_mesh, new_mesh.topology().dim()-1)
    new_boundaries.set_values(boundaries.array())

    # Update measures
    # new_dx = Measure('dx', domain = new_mesh)
    # new_ds = Measure('ds', subdomain_data = new_boundaries)

    return new_mesh, new_boundaries, new_subdomains#, new_dx, new_ds







if __name__=='__main__':
    
    # ============================= MESHING FROM GMSH ======================================= #
    # define mesh path
    mesh_path = '/mnt/d/dissertation/free_surface/'
    mesh_name = 'deformed_thin_sheet'

    # # Get subdomain indexes
    Subdomains = readDomains(mesh_path, mesh_name)
    print(Subdomains)
    
    # # Load mesh and subdomains
    mesh = Mesh(mesh_path + mesh_name + '.xml')
    boundaries = MeshFunction('size_t', mesh, mesh_path + mesh_name + "_facet_region.xml")
    markers = MeshFunction('size_t', mesh, mesh_path + mesh_name + '_physical_region.xml')

    ds=Measure('ds',subdomain_data=boundaries)

    # # get normal and curvature before deformation marker -> top boundary
    kappa, n = curvature(mesh, boundaries, ds, 4)

    print(assemble(kappa*ds(Subdomains['Top'])))

    




    