import datetime
import numpy

import matplotlib.pyplot as plt

import hxform

def scene3d(xlims=(-1, 1), ylims=(-1, 1), zlims=(-1, 1), xlabel='X', ylabel='Y', zlabel='Z', figsize=(8, 8)):

  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D

  fig = plt.figure(figsize=figsize, dpi=300)
  ax = fig.add_subplot(111, projection='3d')

  ax.set_xlim(xlims)
  ax.set_ylim(ylims)
  ax.set_zlim(zlims)

  # Disable spines, background, and ticks
  ax.set_axis_off()

  # TODO: Handle case where, e.g., xlims[0] > 0 and/or xlims[1] < 0
  ax.text(1.1*xlims[1], 0, 0, xlabel, color='black')
  ax.text(0, 1.1*ylims[1], 0, ylabel, color='black')
  ax.text(0, 0, 1.1*zlims[1], zlabel, color='black')

  ax.plot([0, xlims[1]], [0, 0], [0, 0], color='k', linewidth=0.5)
  ax.plot([xlims[0], 0], [0, 0], [0, 0], color='k', linewidth=0.5, ls='--')
  ax.plot([0, 0], [0, ylims[1]], [0, 0], color='k', linewidth=0.5)
  ax.plot([0, 0], [ylims[0], 0], [0, 0], color='k', linewidth=0.5, ls='--')
  ax.plot([0, 0], [0, 0], [0, zlims[1]], color='k', linewidth=0.5)
  ax.plot([0, 0], [0, 0], [zlims[0], 0], color='k', linewidth=0.5, ls='--')

  #plt.title('3D Vector Plot using quiver3D')

  # TODO: Use _plane()
  x = numpy.linspace(xlims[0], xlims[1], 2, endpoint=True)
  y = numpy.linspace(ylims[0], ylims[1], 2, endpoint=True)
  x, y = numpy.meshgrid(x, y)
  z = 0*x
  ax.plot_surface(x, y, z, alpha=0.1)
  ax.view_init(elev=30, azim=30)

  return fig, ax

def _plane(ax, vertices, color='k', alpha=0.1):
  """
  Create a plane in 3D space defined by the x, y, and z coordinates.
  """
  from mpl_toolkits.mplot3d import Axes3D
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection

  poly = Poly3DCollection([vertices], alpha=0.6, facecolor='cyan', edgecolor=None)
  ax.add_collection3d(poly)

def _savefig(file_name):
  import matplotlib.pyplot as plt
  import os
  if not os.path.exists(os.path.dirname(file_name)):
    os.makedirs(os.path.dirname(file_name))
  print(f'Writing {file_name}...')
  plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0.1)

def _angle_wedge(ax, gsm_z_gse, mag_z_gse):
  from mpl_toolkits.mplot3d import Axes3D
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection

  # Wedge for dipole tilt
  vertices = [
    (0, 0, 0),
    (0.5*gsm_z_gse[0], 0.5*gsm_z_gse[1], 0.5*gsm_z_gse[2]),
    (0.5*mag_z_gse[0], 0.5*mag_z_gse[1], 0.5*mag_z_gse[2])
  ]

  # TODO: Make wedge have arc
  poly = Poly3DCollection([vertices], facecolor='black', edgecolor=None)
  ax.add_collection3d(poly)

def _title(gsm_z_gse, t, t_dt, dipole_tilt):
  time_string = t_dt.strftime("%Y-%m-%dT%H:%M:%S")
  title = f'{time_string}\n'
  title += r'$Z_{GSM}$ in $GSE$ = '
  title += f'[{gsm_z_gse[t, 0]:.2f}, {gsm_z_gse[t, 1]:.2f}, {gsm_z_gse[t, 2]:.2f}]'
  title += f'\nDipole tilt = ${dipole_tilt:.2f}^\circ$'
  return title

delta = {'minutes': 30}
to = datetime.datetime(2010, 12, 21, 0, 0, 0)
tf = datetime.datetime(2010, 12, 23, 0, 0, 0)

kwargs = {
  'csys_in': 'GSM',
  'csys_out': 'GSE',
  'ctype_in': 'car',
  'ctype_out': 'car',
  'lib': 'geopack_08_dp'
}

labels = {
  'xlabel': '$X_{GSE}$, $X_{GSM}$',
  'ylabel': '$Y_{GSE}$',
  'zlabel': '$Z_{GSE}$'
}

t = hxform.time_array(to, tf, delta)
t_dts = hxform.ints2datetime(t)

gsm_z = numpy.array([0., 0., 1.])
kwargs['csys_in'] = 'GSM'
gsm_z_gse = hxform.transform(gsm_z, t, **kwargs)
print(gsm_z_gse[1:5, :])

mag_z = numpy.array([0., 0., 1.])
kwargs['csys_in'] = 'MAG'
mag_z_gse = hxform.transform(mag_z, t, **kwargs)
print(mag_z_gse[1:5, :])


for t, t_dt in enumerate(t_dts):

  fig, ax = scene3d(**labels)

  vertices = [
    (-1, 0, 0),
    ( 1, 0, 0),
    ( 1, 1.5*gsm_z_gse[t, 1], 1.5*gsm_z_gse[t, 2]),
    (-1, 1.5*gsm_z_gse[t, 1], 1.5*gsm_z_gse[t, 2])
  ]

  # GSM X-Z plane
  _plane(ax, numpy.array(vertices))

  _angle_wedge(ax, gsm_z_gse[t, :], mag_z_gse[t, :])

  if t > 0:
    ax.scatter(gsm_z_gse[:t, 0], gsm_z_gse[:t, 1], gsm_z_gse[:t, 2], color='black', s=1)
    ax.scatter(mag_z_gse[:t, 0], mag_z_gse[:t, 1], gsm_z_gse[:t, 2], color='black', s=1)

  ax.quiver(0, 0, 0, *gsm_z_gse[t, :], color='red', arrow_length_ratio=0.1)
  ax.text(*gsm_z_gse[t, :], '$Z_{GSM}$', color='red')

  ax.quiver(0, 0, 0, *mag_z_gse[t, :], color='blue', arrow_length_ratio=0.1)
  ax.text(*mag_z_gse[t, :], '$Z_{MAG}$', color='blue')

  m_dot_s = numpy.dot([1, 0, 0], mag_z_gse[t, :])
  dipole_tilt = (180/numpy.pi)*numpy.arcsin(m_dot_s)

  ax.set_title(_title(gsm_z_gse, t, t_dt, dipole_tilt))

  file_name = f'figures/frame_vis/frame_vis_{t_dt.strftime("%Y%m%dT%H%M%S")}.png'
  _savefig(file_name)
  plt.close()

  #if t > 3: break