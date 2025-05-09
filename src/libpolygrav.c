/*
    File: libpolygrav.c
    Author: Harrison Agrusa, hagrusa@oca.eu
    Description: compute the acceleration & potential due to uniform density polyhedron
    This is based on a python script I wrote a long time ago, 
    which was based on a matlab script from Alex Meyer,
    which I then converted to c with some help from chatgpt. 
    I'm not a c expert but this should work...
*/
// gcc -shared -fPIC -o libpolygrav.so polygravity.c

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N_DIM 3

typedef struct {
    double x;
    double y;
    double z;
} Vector3D;

double dot_product(Vector3D a, Vector3D b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

double norm(Vector3D v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

Vector3D scalar_multiply(double s, Vector3D v) {
    Vector3D result;
    result.x = s * v.x;
    result.y = s * v.y;
    result.z = s * v.z;
    return result;
}

Vector3D vector_add(Vector3D v1, Vector3D v2) {
    Vector3D result;
    result.x = v1.x+v2.x;
    result.y = v1.y+v2.y;
    result.z = v1.z+v2.z;
    return result;
}

Vector3D vector_subtract(Vector3D v1, Vector3D v2) {
    Vector3D result;
    result.x = v1.x-v2.x;
    result.y = v1.y-v2.y;
    result.z = v1.z-v2.z;
    return result;
}

Vector3D cross_product(Vector3D a, Vector3D b) {
    Vector3D result;
    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
    return result;
}

Vector3D matrix_multiply(double M[N_DIM][N_DIM], Vector3D v) {
    Vector3D result;
    result.x = M[0][0] * v.x + M[0][1] * v.y + M[0][2] * v.z;
    result.y = M[1][0] * v.x + M[1][1] * v.y + M[1][2] * v.z;
    result.z = M[2][0] * v.x + M[2][1] * v.y + M[2][2] * v.z;
    return result;
}

double quadratic_form(double M[3][3], Vector3D v) {
    // this function does v^T M v
    double result = 0.0;
    result += M[0][0] * v.x * v.x + M[0][1] * v.x * v.y + M[0][2] * v.x * v.z;
    result += M[1][0] * v.y * v.x + M[1][1] * v.y * v.y + M[1][2] * v.y * v.z;
    result += M[2][0] * v.z * v.x + M[2][1] * v.z * v.y + M[2][2] * v.z * v.z;
    return result;
}

void matrix_add(double mat1[N_DIM][N_DIM], double mat2[N_DIM][N_DIM], double result[N_DIM][N_DIM]) {
    for (int i = 0; i < N_DIM; i++) {
        for (int j = 0; j < N_DIM; j++) {
            result[i][j] = mat1[i][j] + mat2[i][j];
        }
    }
}

void outer_product(Vector3D a, Vector3D b, double result[3][3]) {
    result[0][0] = a.x * b.x;
    result[0][1] = a.x * b.y;
    result[0][2] = a.x * b.z;

    result[1][0] = a.y * b.x;
    result[1][1] = a.y * b.y;
    result[1][2] = a.y * b.z;

    result[2][0] = a.z * b.x;
    result[2][1] = a.z * b.y;
    result[2][2] = a.z * b.z;
}

void polyGrav(double r_field[3], double verts[][3], int faces[][3], int edges[][2], int faces_on_edge[][2], int num_faces, int num_edges, double dU[3], double U[1]) {
    // This function calculates the gravitational acceleration and potential around a
    // polyhedron shape model at an exterior point. The polyhedron is defined by
    // triangular facets. This algorithm is defined in Werner & Scheeres, 1996
    //
    //
    // We are assuming G and the density are both equal to one.
    // you can then multiply the answer by whatever units of G and density you want
    //
    // INPUTS:
    //   r_field: field point exterior to the polyhedron at which to calculate
    //            the gravitational acceleration. Assumed to be in the body-fixed frame 
    //   verts: array where row i corresponds to the coordinates of vertex i
    //   facet: array where row i gives the three vertices that define a triangular facet
    //   edges: array where row i gives the two vertices that define an edge of a face
    //   faces_on_edge: array where row i gives the two face indices that form edge i
    //
    // OUTPUT:
    //   dU: 3x1 gravitational acceleration at point R_field
    //    U: 1x1 gravitational potential 

    // lots of variable names may not have good names. I'm just basing this off the symbols used in the paper

    // First, declare a bunch of stuff now:
    const double sigma = 1.0; // Assuming G and sigma equal to one
    const double G = 1.0; 
    double dU_face[3] = {0.0, 0.0, 0.0};
    double U_face = 0.0;
    double dU_edge[3] = {0.0, 0.0, 0.0};
    double U_edge = 0.0;
    Vector3D dU_toAdd;
    Vector3D nhat_faces[num_faces]; // normal vector at each face
    
    int i;

    // iterate over faces
    for (i = 0; i < num_faces; i++) {
        int v1_idx = faces[i][0];
        int v2_idx = faces[i][1];
        int v3_idx = faces[i][2];

        Vector3D v1 = {verts[v1_idx][0], verts[v1_idx][1], verts[v1_idx][2]}; //vertex 1
        Vector3D v2 = {verts[v2_idx][0], verts[v2_idx][1], verts[v2_idx][2]}; //vertex 2
        Vector3D v3 = {verts[v3_idx][0], verts[v3_idx][1], verts[v3_idx][2]}; //vertex 3
      
        Vector3D e1 = {v2.x - v1.x, v2.y - v1.y, v2.z - v1.z}; //edge 1
        Vector3D e2 = {v3.x - v1.x, v3.y - v1.y, v3.z - v1.z}; //edge 2
        Vector3D nhat = cross_product(e1, e2); //normal vector
        nhat = scalar_multiply(1./norm(nhat), nhat);

        // this was originally done to ensure the normal vector always points 'outwards'
        // but this will leade to mistakes for concave shapes!
        // if (dot_product(nhat, v1) < 0.0) {
        //     nhat = scalar_multiply(-1.0, nhat); //ensure pointing outwards
        // }

        nhat_faces[i] = nhat; //save for later

        Vector3D a_vec = {v1.x - r_field[0], v1.y - r_field[1], v1.z - r_field[2]}; //vector from field point to vertices
        Vector3D b_vec = {v2.x - r_field[0], v2.y - r_field[1], v2.z - r_field[2]};
        Vector3D c_vec = {v3.x - r_field[0], v3.y - r_field[1], v3.z - r_field[2]};
        double a = norm(a_vec);
        double b = norm(b_vec);
        double c = norm(c_vec);
        double wf = 2 * atan2(dot_product(a_vec, cross_product(b_vec, c_vec)), 
                                a*b*c + a*dot_product(b_vec, c_vec) + b*dot_product(c_vec, a_vec) + c*dot_product(a_vec, b_vec));

        double Ff[3][3];

        outer_product(nhat, nhat, Ff);
        Vector3D rf = {v1.x - r_field[0], v1.y - r_field[1], v1.z - r_field[2]};

        dU_toAdd = scalar_multiply(wf, matrix_multiply(Ff, rf));

        dU_face[0] += dU_toAdd.x;
        dU_face[1] += dU_toAdd.y;
        dU_face[2] += dU_toAdd.z;

        U_face += quadratic_form(Ff, rf) * wf;

    }
    // now iterate over edges
    for (i = 0; i < num_edges; i++) {
        int v1_idx = edges[i][0];
        int v2_idx = edges[i][1];
        Vector3D v1 = {verts[v1_idx][0], verts[v1_idx][1], verts[v1_idx][2]};
        Vector3D v2 = {verts[v2_idx][0], verts[v2_idx][1], verts[v2_idx][2]};

        Vector3D e_vec = {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z}; //vector pointing along the edge
        double e = norm(e_vec);
        Vector3D a_vec = {v1.x - r_field[0], v1.y - r_field[1], v1.z - r_field[2]};
        Vector3D b_vec = {v2.x - r_field[0], v2.y - r_field[1], v2.z - r_field[2]};
        double a = norm(a_vec);
        double b = norm(b_vec);
        double Le = log((a + b + e) / (a + b - e));

        Vector3D nhat_a = nhat_faces[faces_on_edge[i][0]];
        Vector3D nhat_b = nhat_faces[faces_on_edge[i][1]];

        Vector3D nhat_a_e = cross_product(e_vec, nhat_a);
        nhat_a_e = scalar_multiply(1./norm(nhat_a_e), nhat_a_e);

        Vector3D nhat_b_e = cross_product(e_vec, nhat_b);
        nhat_b_e = scalar_multiply(1./norm(nhat_b_e), nhat_b_e);


        int other_vert_a = -1;
        int other_vert_b = -1;
        int j;
        for (j = 0; j < 3; j++) {
            if (faces[faces_on_edge[i][0]][j] != v1_idx && faces[faces_on_edge[i][0]][j] != v2_idx) {
                other_vert_a = faces[faces_on_edge[i][0]][j];
                break;
            }
        }
        for (j = 0; j < 3; j++) {
            if (faces[faces_on_edge[i][1]][j] != v1_idx && faces[faces_on_edge[i][1]][j] != v2_idx) {
                other_vert_b = faces[faces_on_edge[i][1]][j];
                break;
            }
        }

        if (other_vert_a == -1){
            fprintf(stderr, "other_vert_a not assigned\n");
            exit(EXIT_FAILURE);
        } else if (other_vert_b == -1) { 
            fprintf(stderr, "other_vert_b not assigned\n");
            exit(EXIT_FAILURE);
        }

        Vector3D other_vert_a_vec = {verts[other_vert_a][0], verts[other_vert_a][1], verts[other_vert_a][2]};
        Vector3D other_vert_b_vec = {verts[other_vert_b][0], verts[other_vert_b][1], verts[other_vert_b][2]};

        if (dot_product(nhat_a_e, vector_subtract(other_vert_a_vec,v1)) > 0.0) {
            nhat_a_e = scalar_multiply(-1.0, nhat_a_e);
        }

        if (dot_product(nhat_b_e, vector_subtract(other_vert_b_vec,v1)) > 0.0) {
            nhat_b_e = scalar_multiply(-1.0, nhat_b_e);
        }

        double outev1[3][3];
        double outev2[3][3];
        double Ee[3][3];


        outer_product(nhat_a, nhat_a_e, outev1);
        outer_product(nhat_b, nhat_b_e, outev2);
        matrix_add(outev1, outev2, Ee);


        Vector3D re = {v1.x - r_field[0], v1.y - r_field[1], v1.z - r_field[2]};
        dU_toAdd = scalar_multiply(Le, matrix_multiply(Ee,re));

        dU_edge[0] += dU_toAdd.x;
        dU_edge[1] += dU_toAdd.y;
        dU_edge[2] += dU_toAdd.z;
        U_edge += quadratic_form(Ee, re) * Le;

    }
    // remember, G & sigma are both 1
    // depending on what units you are using, 
    // just multiply this result by the "correct" value of G and sigma
    // to obtain the desired physical units
    dU[0] = -G * sigma * dU_edge[0] + G * sigma * dU_face[0];
    dU[1] = -G * sigma * dU_edge[1] + G * sigma * dU_face[1];
    dU[2] = -G * sigma * dU_edge[2] + G * sigma * dU_face[2];
    U[0] = 0.5 * G * sigma * U_edge - 0.5 * G * sigma * U_face;
}



