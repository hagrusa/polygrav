#example script for how to compute gravity at surface of shape model to get surface slopes


import numpy as np
import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use("dark_background")


import polygrav as pg

obj = "a66391_1999kw4_primary.obj"



G = 6.67430e-11 #mks units
#nominal density and spin of kw4 
rho = 1.97*1000 #kg/m^3
w = 2*np.pi/(2.7650*3600) #spin rate
w_vec = np.array([0,0,w])

verts, faces = pg.obj2array(obj)
verts *= 1.e3 #convert km to m


#for extremeley large shape files, this stuff gets slow
#best to precompute this once and dump to a file.
face_midpoints = pg.get_face_midpoints(verts,faces)
normals = pg.get_face_normals(verts,faces) #this will throw an error if any surface normals are pointed inwards
edges = pg.get_edges(faces)
faces_on_edge = pg.get_faces_on_edge(faces, edges)

a_grav = np.empty(np.shape(normals))
U_grav = np.empty(len(faces))
print("computing gravitational accelerations...")
print("in practice, you would want to precompute this and dump to a file for large shape models.")
for i, r in enumerate(tqdm.tqdm(face_midpoints)):
	#get gravitational acceleration at center of each facet
	a_grav[i], U_grav[i] = pg.calc_polygrav(r, verts, faces, edges, faces_on_edge)

a_grav *= G*rho
U_grav *= G*rho

#compute centrifugal acceleration
a_centri = -np.cross(w_vec, np.cross(w_vec, face_midpoints))

a_net = a_grav + a_centri


a_mag = np.linalg.norm(a_net, axis=1)[:, np.newaxis]

slopes = np.degrees(np.arccos(np.sum(-normals*a_net/a_mag,axis=1)))


print("maximum slope:", np.max(slopes))

cmap = 'plasma'
vmax=75
norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
colors = mpl.colormaps.get_cmap(cmap)(norm(slopes))[:,:3]
bounds = np.max(np.abs(verts),axis=0)

fig= plt.figure(figsize=plt.figaspect(1))
ax = fig.add_subplot(111, projection='3d')##
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False  
ax.set_xlim([-bounds[0],bounds[0]])
ax.set_ylim([-bounds[1],bounds[1]])
ax.set_zlim([-bounds[2],bounds[2]])
ax.set_xlabel("x [m]")#, fontsize=medium_font)
ax.set_ylabel("y [m]")#, fontsize=medium_font)
ax.set_zlabel("z [m]")#, fontsize=medium_font)

trisurf = ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], triangles=faces)
trisurf.set_fc(colors)
cbar = plt.colorbar(mappable, location='top',ax=ax)
cbar.set_label("Slope", rotation=0, labelpad=10)



cbar.set_ticks(np.arange(0,vmax+1,15))
tickLabels = []
for tick in cbar.get_ticks():
    tickLabels += [r"${0}^{{\circ}}$".format(tick),]
cbar.set_ticklabels(tickLabels)

azim, elev = -140, 20
ax.view_init(azim=-140.0, elev=20.0)

ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
ax.zaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

plt.tight_layout()
plt.savefig('slopes.png')
print("3D surface slope plot saved to slopes.png")
# plt.close()


# a_normal = np.sum(a_net*normals,axis=1) #normal component magnitude
# a_hor = a_net - a_normal.reshape(-1,1)*normals #subtract off normal component
# a_hor_norm = a_hor/np.linalg.norm(a_hor,axis=1).reshape(-1,1) #acceleration parallel to surface, normalized
# nskip = 10
# arrow_length = 75
# distance = 1e6  # assume observer is far away 
# camera_position = np.array([
#     distance * np.cos(np.radians(elev)) * np.cos(np.radians(azim)),  # X
#     distance * np.cos(np.radians(elev)) * np.sin(np.radians(azim)),  # Y
#     distance * np.sin(np.radians(elev))                       # Z
# ])

# arrow_pos = face_midpoints[::nskip,:]
# arrow_dir = a_hor_norm[::nskip,:]
# arrow_dir /= np.linalg.norm(arrow_dir,axis=1).reshape(-1,1)
# arrow_norms = normals[::nskip,:]
# arrow_pos += arrow_norms*20

# to_arrows = arrow_pos - camera_position
# dots = np.sum(arrow_norms*to_arrows, axis=1)
# visible_arrows = np.where(dots<0.0)[0]
# x,y,z = arrow_pos[visible_arrows,0], arrow_pos[visible_arrows,1], arrow_pos[visible_arrows,2]
# u,v,w = arrow_dir[visible_arrows,0], arrow_dir[visible_arrows,1], arrow_dir[visible_arrows,2]

# quiver_plot=ax.quiver(x,y,z,u,v,w, length=arrow_length, normalize=False, color='w')#, zorder=10)
# plt.savefig('slopes_and_direction.png', dpi=300)






