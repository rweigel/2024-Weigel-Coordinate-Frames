import numpy as np
import hxform
from hapiclient import hapi
from hapiclient import hapitime2datetime

# https://www.nasa.gov/missions/mms/nasas-mms-achieves-closest-ever-flying-formation/
# On Sept. 15, 2016, NASA’s Magnetospheric Multiscale, or MMS, mission achieved
# a new record: Its four spacecraft are flying only four-and-a-half miles apart,
# the closest separation ever of any multi-spacecraft formation.

# In the following, we compute the separation distances and angles from s/c
# pairs using position data from CDAWeb and SSCWeb. "closest separation" seems
# to mean the minimum distance of any s/c from centroid of the four s/c.

# See also https://mmsvis.gsfc.nasa.gov/ for plots

R_E = 6378.16

# "Its four spacecraft are flying only four-and-a-half miles apart" => 7.24 km
angle = 7.24/(8.79*R_E)
hxform.xprint(f"4.5 miles separation => {np.degrees(angle):.2e} deg at 8.79 R_E")

def report(times, positions, title):

  for i in range(4):
    r = np.linalg.norm(positions[i], axis=1)
    r_ave = np.mean(r)
    hxform.xprint(f'  MMS{i+1} mean distance: {r_ave/R_E:.2f} R_E')
    centroid = np.mean(np.array(positions), axis=0)
    min_dis = np.min(np.linalg.norm(positions[i]-centroid, axis=1))
    hxform.xprint(f'    Minimum distance from centroid: {min_dis:.2f} km')

  separations = []
  labels = []
  for i in range(4):
    for j in range(i+1, 4):
      label = f'MMS{i+1}-MMS{j+1}'
      labels.append(label)

      separation = np.linalg.norm(positions[i] - positions[j], axis=1)
      separations.append(separation)

      min_sep = np.min(separation)
      hxform.xprint(label)
      angle = np.degrees(min_sep/r_ave)
      hxform.xprint(f'  Minimum separation distance and angle: {min_sep:.2f} km  and {angle:.2e} deg')

  plot(times, separations, labels, title)

def plot(times, separations, labels, title):
  from matplotlib import pyplot as plt
  from hapiplot.plot.datetick import datetick

  for time, separation, label in zip(times, separations, labels):
    plt.plot(time, separation, label=label)

  plt.title(title)
  plt.ylabel('Separation Distance (km)')
  plt.grid()
  plt.legend()
  datetick(dir='x')

  fname = 'mms_separation_' + title.replace(" ", "_").replace("(", "").replace(")", "")
  writefig(fname)
  plt.close()

def writefig(fname):
  from matplotlib import pyplot as plt
  import os

  for fmt in ['pdf', 'svg', 'png']:
    subdir = fmt
    if fmt == 'pdf':
      subdir = ''
    fname_full = os.path.join('figures', 'mms_separation', subdir, fname)

    if not os.path.exists(os.path.dirname(fname_full)):
      os.makedirs(os.path.dirname(fname_full))

    hxform.xprint(f'Writing: {fname_full}.{fmt}')
    if fmt == 'png':
      plt.savefig(f'{fname_full}', dpi=300)
    else:
      plt.savefig(f'{fname_full}.{fmt}')


start = '2016-09-14'
stop  = '2016-09-16'
opts  = {'logging': False, 'usecache': True, 'cachedir': './data/hapi'}

print("Start: {start}, Stop: {stop}".format(start=start, stop=stop))


server = 'https://cdaweb.gsfc.nasa.gov/hapi'
frame = 'GSE'
dataset_suffix = 'MEC_SRVY_L2_EPHT89D'
title = f'CDAWeb {frame} (MMSi_{dataset_suffix})'
hxform.xprint(title)

times = []
positions = []
for s in range(1, 5):
  sc   = f'mms{s}'
  dataset = f'{sc.upper()}_MEC_SRVY_L2_EPHT89D'
  parameters = f'{sc}_mec_r_gse'
  # EPD data has time stamps that differ between s/c
  #dataset = f'{sc.upper()}_EPD-EIS_SRVY_L2_ELECTRONENERGY'
  #parameters = f'{sc}_epd_eis_srvy_l2_electronenergy_position_{frame.lower()}'
  data, meta = hapi(server, dataset, parameters, start, stop, **opts)

  times.append(hapitime2datetime(data['Time']))
  positions.append(data[parameters])

report(times, positions, title)

hxform.xprint('')


server = 'http://hapi-server.org/servers/SSCWeb/hapi'
frame = 'GSE'
times = []
positions = []
title = f'SSCWeb {frame}'
hxform.xprint(title)
for s in range(1, 5):
  dataset    = f'mms{s}'
  parameters = f'X_{frame},Y_{frame},Z_{frame}'

  data, meta = hapi(server, dataset, parameters, start, stop, **opts)

  xyz = np.column_stack((data[f'X_{frame}'], data[f'Y_{frame}'], data[f'Z_{frame}']))

  times.append(hapitime2datetime(data['Time']))
  positions.append(xyz*R_E)

report(times, positions, title)
