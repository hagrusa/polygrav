import numpy as np
import polygrav as pg

obj = "a66391_1999kw4_secondary.obj"

correct_accel = np.array([-4.80799978e-06, -4.94669433e-06, -5.03747626e-06])
correct_potential = 0.007400527005605368

G = 6.67430e-11 #mks units
rho = 2000 #kg/m^3
r_field = np.array([500.0, 500.0, 500.0]) #compute gravitational acceleration and potential at this point


verts, faces = pg.obj2array(obj)
verts *= 1000.0 #convert km to m

normals = pg.get_face_normals(verts,faces) #this will throw an error if any surface normals are pointed inwards

edges = pg.get_edges(faces)
faces_on_edge = pg.get_faces_on_edge(faces, edges)


accel, potential = pg.calc_polygrav(r_field, verts, faces, edges, faces_on_edge)
#here, accel and potential need to be multiplied by G and rho to be in mks units:
accel *= G*rho
potential *= G*rho


assert(np.all(np.isclose(correct_accel, accel)))
assert(np.isclose(correct_potential, potential))
print('gravitational acceleration [m/s^2]:', accel)
print('gravitational potential [m^2/s^2]:', potential)

