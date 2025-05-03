import numpy as np
import os
import ctypes

libpolygrav = ctypes.CDLL(os.path.join(os.path.dirname(os.path.realpath(__file__)),'libpolygrav.so'))

# Define function argument and return types
libpolygrav.polyGrav.restype = None
libpolygrav.polyGrav.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), # r_field
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'), # verts
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags='C_CONTIGUOUS'), # faces
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags='C_CONTIGUOUS'), # edges
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags='C_CONTIGUOUS'), # faces_on_edge
    ctypes.c_int, # num_faces
    ctypes.c_int, # num_edges
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), # dU 
    np.ctypeslib.ndpointer(dtype=np.float64,ndim=1, flags='C_CONTIGUOUS') #U
]

def calc_polygrav(r_field, verts, faces, edges, faces_on_edge):
    # add some error checking here.
    dU = np.zeros(3, dtype=np.float64)
    U=np.zeros(1,dtype=np.float64)
    libpolygrav.polyGrav(r_field.astype(np.float64), 
                        verts.astype(np.float64), 
                        faces.astype(np.int32), 
                        edges.astype(np.int32), 
                        faces_on_edge.astype(np.int32), 
                        len(faces), 
                        len(edges),
                        dU, U)
    return dU, U[0]



#some utilities:
def obj2array(obj_file): 
    obj = np.genfromtxt(obj_file, dtype=[('vf', 'S1'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
    facesInd = np.where(obj['vf'] == b'f')[0]
    vertsInd = np.where(obj['vf'] == b'v')[0]


    verts = np.column_stack([
                            obj['x'][vertsInd], 
                            obj['y'][vertsInd], 
                            obj['z'][vertsInd]]).astype(np.float64)

    faces = np.column_stack([
                            obj['x'][facesInd], 
                            obj['y'][facesInd], 
                            obj['z'][facesInd]]).astype(np.int32)
    faces -= 1 #python = zero indexing

    return verts, faces

def obj2array_porter(obj_file):
    #deal with Simon Porter's odd arrokoth .obj file


    obj = np.genfromtxt(obj_file, dtype=[('vf', 'S2'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
    # facesInd = np.where(obj['vf'] == b'f')[0]
    vertsInd = np.where(obj['vf'] == b'v')[0]
    normalsInd = np.where(obj['vf'] == b'vn')[0]

    verts = np.column_stack([
                            obj['x'][vertsInd], 
                            obj['y'][vertsInd], 
                            obj['z'][vertsInd]]).astype(np.float64)

    normals = np.column_stack([
                            obj['x'][normalsInd], 
                            obj['y'][normalsInd], 
                            obj['z'][normalsInd]]).astype(np.float64)

    with open(obj_file, "r") as file:
        lines = [line.strip() for line in file if line.startswith('f')]
    
    faces = np.zeros((len(lines),3)).astype(np.int32)
    for i, line in enumerate(lines):
        faces[i] = [int(x.split("/")[0]) for x in line.split()[1:]]

    faces -= 1 #python = zero indexing

    # quit()

    return verts, faces, normals


def obj2array_spencer(obj_file):
    #deal with Spencer 2020 .obj arrokoth file


    obj = np.genfromtxt(obj_file, dtype=[('vf', 'S2'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
    # facesInd = np.where(obj['vf'] == b'f')[0]
    vertsInd = np.where(obj['vf'] == b'v')[0]
    normalsInd = np.where(obj['vf'] == b'vn')[0]

    verts = np.column_stack([
                            obj['x'][vertsInd], 
                            obj['y'][vertsInd], 
                            obj['z'][vertsInd]]).astype(np.float64)

    normals = np.column_stack([
                            obj['x'][normalsInd], 
                            obj['y'][normalsInd], 
                            obj['z'][normalsInd]]).astype(np.float64)

    with open(obj_file, "r") as file:
        lines = [line.strip() for line in file if line.startswith('f')]
    
    faces = np.zeros((len(lines),3)).astype(np.int32)
    for i, line in enumerate(lines):
        faces[i] = [int(x.split("//")[0]) for x in line.split()[1:]]

    faces -= 1 #python = zero indexing

    # quit()

    return verts, faces, normals

def get_face_midpoints(verts,faces):
    return np.mean(verts[faces], axis=1)


def reorder_faces(verts, faces):
    #change ordering of faces so all surface normals will point outwards

    #get direction to center of each face:
    face_directions = get_face_midpoints(verts,faces)
    face_directions /= np.linalg.norm(face_directions, axis=1)[:,np.newaxis]

    # Compute edge vectors for each face
    edge1 = verts[faces[:,1]] - verts[faces[:,0]]
    edge2 = verts[faces[:,2]] - verts[faces[:,0]]

    # Compute face normals
    face_normals = np.cross(edge1, edge2)
    face_normals /= np.linalg.norm(face_normals, axis=1)[:, np.newaxis]

    # Ensure normal points outward
    dot_products = np.sum(face_directions*face_normals, axis=1)

    # Check and adjust normals if needed
    dot_products = np.sum(face_directions * face_normals, axis=1)
    faces_to_flip = np.where(dot_products < 0)[0]
    faces[faces_to_flip, [1, 2]] = faces[faces_to_flip, [2, 1]]
    return faces

def get_face_normals(verts, faces):
    #get direction to center of each face:
    face_directions = get_face_midpoints(verts,faces)
    face_directions /= np.linalg.norm(face_directions, axis=1)[:,np.newaxis]

    # Compute edge vectors for each face
    edge1 = verts[faces[:,1]] - verts[faces[:,0]]
    edge2 = verts[faces[:,2]] - verts[faces[:,0]]

    # Compute face normals
    face_normals = np.cross(edge1, edge2)
    face_normals /= np.linalg.norm(face_normals, axis=1)[:, np.newaxis]

    # Ensure normal points outward
    dot_products = np.sum(face_directions*face_normals, axis=1)
    # assert(np.all(dot_products>=0.0)) #assert all normals pointed outwards

    return face_normals


def get_face_normals_porter(verts, faces, vertex_normals):
    #get direction to center of each face:
    face_directions = get_face_midpoints(verts,faces)
    face_directions /= np.linalg.norm(face_directions, axis=1)[:,np.newaxis]

    # Compute edge vectors for each face
    edge1 = verts[faces[:,1]] - verts[faces[:,0]]
    edge2 = verts[faces[:,2]] - verts[faces[:,0]]

    # Compute face normals
    face_normals = np.cross(edge1, edge2)
    face_normals /= np.linalg.norm(face_normals, axis=1)[:, np.newaxis]

    reference_normals = vertex_normals[faces[:,0]] #grab the vertex normal from the first of the three vertices
    # print(reference_normals)
    # print(np.shape(verts), np.shape(faces), np.shape(vertex_normals))
    # print(np.shape(faces), np.shape(faces[:,0]))
    # print(np.shape(reference_normals))

    dot_products = np.sum(face_normals*reference_normals, axis=1)
    normals_to_flip = np.where(dot_products < 0.0)[0]
    print(len(normals_to_flip))
    face_normals[normals_to_flip] *= -1

    # Ensure normal points outward
    dot_products = np.sum(reference_normals*face_normals, axis=1)
    assert(np.all(dot_products>=0.0)) #assert all normals pointed outwards

    return face_normals


def get_edges(faces):
    #get all the edges
    edges_all = np.row_stack([faces[:,0:2], faces[:,1:], faces[:,[2,0]]])
    edges_ordered =np.sort(edges_all, axis=1)
    edges = np.unique(edges_ordered, axis=0)
    return edges

def get_faces_on_edge(faces, edges):
     #get the two faces on each edge
    faces_on_edge = np.empty(np.shape(edges), dtype=int)

    for i, (v1_ind, v2_ind) in enumerate(edges):
        # Find all faces that contain v1_ind or v2_ind
        contains_v1 = np.any(faces == v1_ind, axis=1)
        contains_v2 = np.any(faces == v2_ind, axis=1)

        # Find faces that contain both v1_ind and v2_ind
        common_faces = np.where(contains_v1 & contains_v2)[0]

        # Ensure there are exactly two common faces
        assert(len(common_faces==2))

        faces_on_edge[i] = common_faces

    return faces_on_edge

def recenter_shape(verts,faces, align_principal_axes=True):
    #oftentimes .obj files are provided in a non-principal axis aligned and centered format
    #this may not be what we want for dynamics simulations
    #but use this function at your own risk!


    rho = 1 #doesn't actually matter, density can be any constant
    P = np.zeros((3,3))
    for j in range(0,3):
        for k in range(0,3):
            V = 0.0
            R = np.zeros(3)
            for i in range(0,len(faces[:,0])):
                D = verts[faces[i,0]]
                E = verts[faces[i,1]]
                F = verts[faces[i,2]]

                G = E-D
                H = F-D
                N = np.cross(G,H)
                dV = np.abs(np.dot(D/3.0,N/2.0))

                dR = (D+E+F)/4.0
                V += dV
                R += dV*dR

                P[j,k] += (rho*dV/20.) * (2.*D[j]*D[k] + 2.*E[j]*E[k] + 2.*F[j]*F[k] 
                                         + D[j]*E[k] + D[k]*E[j] 
                                         + D[j]*F[k] + D[k]*F[j]
                                         + E[j]*F[k] + E[k]*F[j])
    R = R/V
    # print('com offset:', R)
    verts -= R #center on center of mass

    if align_principal_axes:
        M = rho*V
        X, Y, Z = R
        #make inertia tensor:
        Ixx = np.sum(np.diag(P)) - P[0,0]
        Iyy = np.sum(np.diag(P)) - P[1,1]
        Izz = np.sum(np.diag(P)) - P[2,2]
        Ixy = -P[0,1]
        Ixz = -P[0,2]
        Iyz = -P[1,2]

        I = np.array([[Ixx, Ixy, Ixz],
                      [Ixy, Iyy, Iyz],
                      [Ixz, Iyz, Izz]])

        I -= M*np.array([[Y**2+Z**2, -X*Y, -X*Z],
                         [-X*Y, X**2+Z**2, -Y*Z],\
                         [-X*Z, -Y*Z, X**2+Y**2]])

        #rotate verts to principal axis alignment
        eigVal, eigVec = np.linalg.eig(I)
        order = np.argsort(eigVal)
        eigVal = eigVal[order]
        eigVec = eigVec[:,order]
        verts = np.dot(verts, eigVec)

    return verts

def recenter_shape_porter(verts,faces, vertex_normals, align_principal_axes=True):
    #oftentimes .obj files are provided in a non-principal axis aligned and centered format
    #this may not be what we want for dynamics simulations
    #but use this function at your own risk!


    rho = 1 #doesn't actually matter, density can be any constant
    P = np.zeros((3,3))
    for j in range(0,3):
        for k in range(0,3):
            V = 0.0
            R = np.zeros(3)
            for i in range(0,len(faces[:,0])):
                D = verts[faces[i,0]]
                E = verts[faces[i,1]]
                F = verts[faces[i,2]]

                G = E-D
                H = F-D
                N = np.cross(G,H)
                dV = np.abs(np.dot(D/3.0,N/2.0))

                dR = (D+E+F)/4.0
                V += dV
                R += dV*dR

                P[j,k] += (rho*dV/20.) * (2.*D[j]*D[k] + 2.*E[j]*E[k] + 2.*F[j]*F[k] 
                                         + D[j]*E[k] + D[k]*E[j] 
                                         + D[j]*F[k] + D[k]*F[j]
                                         + E[j]*F[k] + E[k]*F[j])
    R = R/V
    # print('com offset:', R)
    verts -= R #center on center of mass

    if align_principal_axes:
        M = rho*V
        X, Y, Z = R
        #make inertia tensor:
        Ixx = np.sum(np.diag(P)) - P[0,0]
        Iyy = np.sum(np.diag(P)) - P[1,1]
        Izz = np.sum(np.diag(P)) - P[2,2]
        Ixy = -P[0,1]
        Ixz = -P[0,2]
        Iyz = -P[1,2]

        I = np.array([[Ixx, Ixy, Ixz],
                      [Ixy, Iyy, Iyz],
                      [Ixz, Iyz, Izz]])

        I -= M*np.array([[Y**2+Z**2, -X*Y, -X*Z],
                         [-X*Y, X**2+Z**2, -Y*Z],\
                         [-X*Z, -Y*Z, X**2+Y**2]])

        #rotate verts to principal axis alignment
        eigVal, eigVec = np.linalg.eig(I)
        order = np.argsort(eigVal)
        eigVal = eigVal[order]
        eigVec = eigVec[:,order]
        verts = np.dot(verts, eigVec)

        vertex_normals = np.dot(vertex_normals, eigVec)

    return verts, vertex_normals

def polygrav_slow(r_field, sigma, verts, faces, edges, faces_on_edge, G=6.67430e-11):
    #dont use this....
    # This function calculates the gravitational acceleration and potential around a
    # polyhedron shape model at an exterior point. The polyhedron is defined by
    # triangular facets. This algorithm is defined in Werner & Scheeres, 1996
    #
    # This function is adapted from the matlab version written by Alex Meyer
    #
    # INPUTS:
    #   r_field: field point exterior to the polyhedron at which to calculate
    #            the gravitational acceleration
    #   sigma: mass density of the polyhedron [mks units]
    #   verts: numpy array where row i corresponds to the coordinates of vertex i
    #   facet: numpy array where row i gives the three vertices that define a
    #          face of the polyhedron
    #   edge: numpy array where row i gives the two vertices that define
    #         an edge of a face
    #   faces_on_edge: numpy array where row i gives the two faces that form edge i
    #
    # OUTPUT:
    #   dU: 3x1 gravitational acceleration at point R_field

    # G = 6.67430e-11

    dU_face = np.zeros(3)
    U_face = 0.0
    nhat_faces = np.empty((len(faces[:,0]),3))
    #iterate over faces
    for i in range(0,len(faces[:,0])): #loop through faces
        #the three vertices making up the face:
        r1 = verts[faces[i,0]]
        r2 = verts[faces[i,1]]
        r3 = verts[faces[i,2]]
        
        #two vectors along the edges to define surface normal:
        e1, e2 = r2-r1, r3-r1
        nhat = np.cross(e1,e2) #surface normal
        nhat /= np.linalg.norm(nhat) #make unit vector

        #ensure surface normal points outwards
        if np.dot(nhat, r1) < 0.0: 
            nhat = -nhat

        nhat_faces[i,:] = nhat #store surface normal

        #compute wf coefficient:
        #coordinates of 3 vertices relative to field point
        a,b,c = r1-r_field, r2-r_field, r3-r_field
        a_mag, b_mag, c_mag = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
        wf = 2*np.arctan2(np.dot(a, np.cross(b,c)), \
            a_mag*b_mag*c_mag + a_mag*np.dot(b,c) + b_mag*np.dot(c,a) + c_mag*np.dot(a,b))

        #Ff dyad
        Ff = np.outer(nhat, nhat)

        #vector from field point to face
        rf = r1 - r_field

        #add acceleration
        dU_face += np.dot(Ff, rf)*wf
        U_face += np.dot(rf, np.dot(Ff, rf))*wf

    #iterate over edges:
    dU_edge = np.zeros(3)
    U_edge = 0.0
    for i in range(0, len(edges[:,0])):
        #get coordinates of the vertices of the edge
        r1 = verts[edges[i,0]]
        r2 = verts[edges[i,1]]


        #get edge length
        e_vec = r1-r2
        e = np.linalg.norm(e_vec)
        #get vector from field point to each vertex:
        a_vec = r1-r_field
        a = np.linalg.norm(a_vec)
        b_vec = r2-r_field
        b = np.linalg.norm(b_vec)

        #get Le coefficient:
        Le = np.log((a+b+e)/(a+b-e))

        #get Ee dyad
        nhat_a = nhat_faces[faces_on_edge[i,0]] #surface normals of faces a and b
        nhat_b = nhat_faces[faces_on_edge[i,1]]
        
        nhat_a_e = np.cross(e_vec, nhat_a) #normal to edge in face a plane
        nhat_a_e /= np.linalg.norm(nhat_a_e)

        nhat_b_e = np.cross(e_vec, nhat_b)
        nhat_b_e /= np.linalg.norm(nhat_b_e)
        #get vector normal to edge in face a to check vector direction
        other_vert_a = verts[faces[faces_on_edge[i,0],0]]
        #want to make sure we are selecting the 3rd vertex (not the two on the edge)
        if np.all(other_vert_a == r1) or np.all(other_vert_a == r2):
            other_vert_a = verts[faces[faces_on_edge[i,0],1]]
            if np.all(other_vert_a == r1) or np.all(other_vert_a == r2):
                other_vert_a = verts[faces[faces_on_edge[i,0],2]]
                if np.all(other_vert_a == r1) or np.all(other_vert_a == r2):
                    print('something is wrong')
                    quit()
        other_vert_b = verts[faces[faces_on_edge[i,1],0]]
        #want to make sure we are selecting the 3rd vertex (not the two on the edge)
        if np.all(other_vert_b == r1) or np.all(other_vert_b == r2):
            other_vert_b = verts[faces[faces_on_edge[i,1],1]]
            if np.all(other_vert_b == r1) or np.all(other_vert_b == r2):
                other_vert_b = verts[faces[faces_on_edge[i,1],2]]
                if np.all(other_vert_b == r1) or np.all(other_vert_b == r2):
                    print('something is wrong')
                    quit()
        e_vec_a = other_vert_a - r1 #vector poingint in face a
        e_vec_b = other_vert_b - r1 #vector pointing in face b

        if np.dot(nhat_a_e,e_vec_a) > 0.0:
            nhat_a_e = -nhat_a_e
        if np.dot(nhat_b_e, e_vec_b) > 0.0:
            nhat_b_e = -nhat_b_e

        Ee = np.outer(nhat_a, nhat_a_e) + np.outer(nhat_b, nhat_b_e)

        #re vector
        re = r1 - r_field

        #add acceleration:
        dU_edge += np.dot(Ee, re)*Le
        U_edge += np.dot(re, np.dot(Ee, re))*Le

    dU = -G*sigma*dU_edge + G*sigma*dU_face
    U = 0.5*G*sigma*U_edge - 0.5*G*sigma*U_face
    return dU, U


