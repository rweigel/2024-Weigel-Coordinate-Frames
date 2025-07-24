# Needs SunPy 6.1.0 or later
# https://github.com/sunpy/sunpy/pull/8193
import os
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt

from hapiclient import hapi
from hapiclient import hapitime2datetime

import utilrsw
utilrsw.mpl.plt_config()

"""
Earth radius in km used by SSCWeb. See
https://sscweb.gsfc.nasa.gov/users_guide/Users_Guide_pt1.html#1.3
"""
R_E = 6378.16 # km
# Source code for SSCWeb is in private STCT Google Drive folder in file 
# icss_rel82.tar.gz (ISTP coordinate transformation library) it is not, and
# cannot, be made publicly available).
"""
In https://amda.irap.omp.eu/service/hapi/info?id=clust1-orb-all
the referenced SPASE record says
"Units": "Re",
"UnitsConversion": "6.4e6>m"
"""
# From Vincent Génot to Weigel on 2025-05-15: SPASE is wrong; correct value is:
R_E_AMDA = 6378.137 # km

################################################################################
# Options
################################################################################
cache_dir = os.path.join('2024-Weigel-Coordinate-Frames', 'data')

satellites_only = [] # Run all satellites
#satellites_only = ['THEMIS-A','THEMIS-B','THEMIS-C','THEMIS-D']
#satellites_only = ['Cluster-1', 'Cluster-2', 'Cluster-3', 'Cluster-4']
#satellites_only = ['MMS-1','MMS-2','MMS-3','MMS-4']
#satellites_only = ['DSCOVR']
#satellites_only = ['MMS-2']
satellites_only = ['Geotail', 'MMS-2']

hapi_logging = False
cos_warnings = False
print_first_last = True   # Print first and last un-interpolated values
interp_times = True
print_interp_vals = False # Print interpolated values
print_common_times = False # Print vals for common times

import warnings
warnings.filterwarnings("ignore", message="The argument 'infer_datetime_format'")
warnings.filterwarnings("ignore", message="Could not infer format")

# Set to ERROR to suppress INFO message for making request to Horizons server
import sunpy
sunpy.log.setLevel('ERROR')

legend_kwargs = {
  'loc': 'upper center',
  'markerscale': 3,
  'borderaxespad': 0,
  'framealpha': 1,
  'frameon': True,
  'ncol': 3,
  'handletextpad': 0.5,
  'columnspacing': 0.75,
}


def infos(satellite=None):

  # Add Polar, Swarm, and DMSP?
  satellites = [
                'Cluster-1',
                'Cluster-2',
                'Cluster-3',
                'Cluster-4',
                'DSCOVR',
                'Geotail',
                'THEMIS-A',
                'THEMIS-B',
                'THEMIS-C',
                'THEMIS-D',
                'MMS-1',
                'MMS-2',
                'MMS-3',
                'MMS-4'
              ]
  if satellite is None:
    return satellites

  if satellite not in satellites:
    raise ValueError(f"Unknown satellite: {satellite}. Must be one of {satellites}")

  infos_ = {'cdaweb': [], 'sscweb': [], 'amda': []}

  if satellite == '' or satellite.startswith('Cluster'):

    sc = f'cluster{satellite[-1]}'
    dataset = f'C{satellite[-1]}_CP_FGM_5VPS'
    start = '2010-09-01T09:09:00.100Z'
    stop  = '2010-09-02T00:00:00.000Z'

    for frame in ['GSE', 'GSM']:

      if frame == 'GSE':
        infos_['cdaweb'].append({
          'name': 'CDAWeb',
          'dataset': dataset,
          'parameter': f'sc_pos_xyz_{frame.lower()}__{dataset}',
          'start': start,
          'stop':  stop
        })
      else:
        # No GSM data in CDAWeb
        infos_['cdaweb'].append(None)

      infos_['sscweb'].append({
        'name': 'SSCWeb',
        'dataset': sc,
        'frame': frame,
        'start': start,
        'stop':  stop
      })

      infos_['amda'].append({
        'name': 'AMDA',
        'dataset': f'clust{satellite[-1]}-orb-all',
        'parameter': f'c{satellite[-1]}_xyz_{frame.lower()}',
        'start': start,
        'stop':  stop
      })

  if satellite == '' or satellite == 'DSCOVR':
    start = '2021-11-25T00:00:00Z'
    stop  = '2021-12-05T00:00:00Z'

    for frame in ['GSE', 'GCI']: # GCI must be last
      infos_["cdaweb"].append({
        'name': 'CDAWeb',
        'dataset': 'DSCOVR_ORBIT_PRE',
        'parameter': f'{frame}_POS',
        'start': start,
        'stop':  stop
      })

      infos_["sscweb"].append({
          'name': 'SSCWeb',
          'dataset': 'dscovr',
          'frame': frame,
          'start': start,
          'stop':  stop
        })

    infos_["cdaweb"].append(infos_["cdaweb"][-1].copy())
    infos_["sscweb"].append(infos_["sscweb"][-1].copy())
    infos_["sscweb"][-1]['frame'] = 'J2K'

  if satellite == '' or satellite.startswith('THEMIS'):

    sc = f'themis{satellite[-1].lower()}'
    start = '2016-09-01T00:00:00Z'
    stop  = '2016-09-02T00:00:00Z'
    for frame in ['GSE', 'GSM', 'GEI']: # GEI must be last
      if frame == 'GEI':
        # Metadata indicates COORDINATE_SYSTEM = GEI for this parameter
        parameter = f'th{sc[-1]}_pos'
      else:
        parameter = f'th{sc[-1]}_pos_{frame.lower()}'

      infos_['cdaweb'].append({
        'name': 'CDAWeb',
        'dataset': f'TH{sc[-1].upper()}_L1_STATE@0',
        'parameter': parameter,
        'start': start,
        'stop':  stop
      })

      infos_['sscweb'].append({
        'name': 'SSCWeb',
        'dataset': sc,
        'frame': frame,
        'start': start,
        'stop':  stop
      })

    infos_['cdaweb'].append(infos_['cdaweb'][-1].copy())
    infos_['sscweb'].append(infos_['sscweb'][-1].copy())
    infos_['sscweb'][-1]['frame'] = 'J2K'

  if satellite == '' or satellite.startswith('MMS'):

    sc   = f'mms{satellite[-1]}'
    start = '2016-09-01T00:00:00Z'
    stop  = '2016-09-02T00:00:00Z'
    for frame in ['GEI', 'GSE', 'GSM']:
      if frame in ['GSE', 'GSM']:
        infos_['cdaweb'].append({
          'name': 'CDAWeb',
          'dataset': f'{sc.upper()}_EPD-EIS_SRVY_L2_ELECTRONENERGY',
          'parameter': f'{sc}_epd_eis_srvy_l2_electronenergy_position_{frame.lower()}',
          'start': start,
          'stop':  stop,
          'frame_name': frame
        })
      else:
        # No J2K or GEI data in CDAWeb dataset
        infos_['cdaweb'].append(None)

      infos_['sscweb'].append({
        'name': 'SSCWeb',
        'dataset': sc,
        'frame': frame,
        'start': start,
        'stop':  stop,
        'frame_name': frame
      })

  if satellite == '' or satellite == 'Geotail':
    """
    https://cdaweb.gsfc.nasa.gov/misc/NotesG.html#GE_OR_DEF
    CDAWeb dataset has GCI, GSE, GSM, HEC
    """

    start      = '2021-11-25T00:00:00Z'
    stop       = '2021-12-05T00:00:00Z'

    for frame in ['GSE', 'GSM', 'GCI']: # GCI must be last
      infos_['cdaweb'].append({
        'name': 'CDAWeb',
        'dataset': 'GE_OR_DEF',
        'parameter': f'{frame}_POS',
        'start': start,
        'stop':  stop,
        'frame_name': frame
      })

      infos_['sscweb'].append({
        'name': 'SSCWeb',
        'dataset': 'geotail',
        'frame': frame if frame != 'GCI' else 'GEI',
        'start': start,
        'stop':  stop,
        'frame_name': frame if frame != 'GCI' else 'GEI'
      })

    # Also compare GCI from CDAWeb with J2K from SSCWeb (which is why GCI must
    # be last in the loop above).
    infos_['cdaweb'].append(infos_['cdaweb'][-1].copy())
    infos_['sscweb'].append(infos_['sscweb'][-1].copy())
    infos_['sscweb'][-1]['frame'] = 'J2K'
    infos_['sscweb'][-1]['frame_name'] = 'J2K'

  if len(infos_['amda']) == 0:
    infos_['amda'] = len(infos_['cdaweb'])*[None]

  return infos_

def jpl(info, logging=False):
  import pickle

  from sunpy.coordinates import get_horizons_coord

  if info['frame'] not in ['GSE', 'GEI', 'GSM']:
    return None

  # To find ids, see https://ssd.jpl.nasa.gov/horizons/app.html#/
  # and edit the "Target Body" field to find the id. Does not seem possible
  # to query for list of Target Body ids.
  # Note: No Geotail or themis{a,d}
  known_ids = {
    'ace': -92,
    'dscovr': -78,
    'mms1': -140482,
    'mms2': -140483,
    'mms3': -140484,
    'mms4': -140485,
    'themisb': -192,
    'themisc': -193
  }

  if info['dataset'] not in known_ids:
    return None
  satellite_id = known_ids[info['dataset']]

  to = info['time'][0].strftime('%Y-%m-%dT%H:%M:%S.%fZ')
  tf = info['time'][-1].strftime('%Y-%m-%dT%H:%M:%S.%fZ')
  cache_file = f"data/jpl/jpl-{info['dataset']}-{info['frame']}-{to}-{tf}.pkl"

  if False and os.path.exists(cache_file):
    print(f"  Saved pickle file: {cache_file}")
    with open(cache_file, 'rb') as f:
      return pickle.load(f)

  #solar_system_ephemeris.set('de432s')
  #solar_system_ephemeris.set('de440s')
  dt = info['time'][1] - info['time'][0]
  # Using, e.g., 60s gives error: https://github.com/sunpy/sunpy/issues/8188
  step_min = int(dt.total_seconds()/60)
  t = {
    "start": info['time'][0],
    "stop": info['time'][-1],
    "step": f'{step_min}m'
  }

  #import pdb; pdb.set_trace()
  if info['frame'] == 'GSE':
    data_jpl = get_horizons_coord(satellite_id, t).geocentricsolarecliptic.cartesian.xyz.to('km')
  if info['frame'] == 'GEI':
    data_jpl = get_horizons_coord(satellite_id, t).geocentricearthequatorial.cartesian.xyz.to('km')
  if info['frame'] == 'GSM':
    data_jpl = get_horizons_coord(satellite_id, t).geocentricsolarmagnetospheric.cartesian.xyz.to('km')

  resp = {'name': 'JPL', 'time': info['time'], 'xyz': data_jpl.value.T/R_E, 'frame_name': info['frame']}
  with open(cache_file, 'wb') as f:
    pickle.dump(resp, f)
    print(f"  Saved pickle file: {cache_file}")

  return resp

def sscweb(info, logging=False):

  """
  1. SSCWeb provides TOD, J2K, GEO, GM, GSE, and SM
  TOD in SSCWeb HAPI server is same as GEI from SSCWeb.
  (The reason the HAPI server uses TOD for GEI is that "TOD" is the 
  POST query parameter names used to request data in GEI is "TOD"
  and consistency in the query parameters was sought.)

  2. From https://sscweb.gsfc.nasa.gov/users_guide/Appendix_C.html,

  > "Geocentric Equatorial Inertial system. This system has X-axis
  pointing from the Earth toward the first point of Aries (the position of
  the Sun at the vernal equinox). This direction is the intersection of the
  Earth's equatorial plane and the ecliptic plane and thus the X-axis lies
  in both planes. The Z-axis is parallel to the rotation axis of the Earth,
  and y completes the right-handed orthogonal set (Y = Z * X). Geocentric
  Inertial (GCI) and Earth-Centered Inertial (ECI) are the same as GEI."

  Based on the above quote, we treat GCI from CDAWeb as the same as GEI from SSCWeb.
  """
  server     = 'http://hapi-server.org/servers/SSCWeb/hapi'
  dataset    = info['dataset']
  frame      = info['frame']
  if frame == 'GEI':
    frame = 'TOD' # See SSCWeb note 1. above.
  if frame == 'GCI':
    frame = 'TOD' # See SSCWeb note 2. above.
  start      = info['start']
  stop       = info['stop']
  parameters = f'X_{frame},Y_{frame},Z_{frame}'
  opts       = {'logging': logging, 'usecache': True, 'cachedir': './data/hapi'}

  data, meta = hapi(server, dataset, parameters, start, stop, **opts)

  info['xyz'] = np.column_stack((data[f'X_{frame}'], data[f'Y_{frame}'], data[f'Z_{frame}']))
  # Convert from YYYY-DOY to YYYY-MM-DD date format
  info['time'] = hapitime2datetime(data['Time'])

  # Return not needed b/c info is modified in place. Keep for clarity.
  return info

def cdaweb(info, logging=False):

  if info is None:
    return None

  server     = 'https://cdaweb.gsfc.nasa.gov/hapi'
  dataset    = info['dataset']
  parameter  = info['parameter']
  start      = info['start']
  stop       = info['stop']
  opts       = {'logging': logging, 'usecache': True, 'cachedir': './data/hapi'}

  data, meta = hapi(server, dataset, parameter, start, stop, **opts)

  xyz = data[parameter]
  info['xyz'] = xyz/R_E # Convert from km to R_E
  # The HAPI SSCWeb server provides ephemeris as three separate parameters. Here
  # we combine parameters into a list. SSCWeb reports in R_E while CDAWeb in km.
  # Convert CDAWeb data to R_E.
  info['time'] = hapitime2datetime(data['Time'])

  # Return not needed b/c info is modified in place. Keep for clarity.
  return info

def amda(info, logging=False):

  if info is None:
    return None

  server     = 'https://amda.irap.omp.eu/service/hapi'
  dataset    = info['dataset']
  parameter  = info['parameter']
  start      = info['start']
  stop       = info['stop']
  opts       = {'logging': logging, 'usecache': True, 'cachedir': './data/hapi'}

  data, meta = hapi(server, dataset, parameter, start, stop, **opts)

  xyz = data[parameter]
  info['xyz'] = xyz*R_E_AMDA/R_E # Convert from AMDA Re to SSCWeb R_E
  info['time'] = hapitime2datetime(data['Time'])

  # Return not needed b/c info is modified in place. Keep for clarity.
  return info

def print_first_last_(sscweb_, cdaweb_, amda_, jpl_):
  if not print_first_last:
    return

  frame = sscweb_['frame']
  for i in [0, -1]:

    xyz_sscweb = sscweb_['xyz'][i]
    if cdaweb_ is not None:
      xyz_cdaweb = cdaweb_['xyz'][i]
    if jpl_ is not None:
      xyz_jpl = jpl_['xyz'][i]
    if amda_ is not None:
      xyz_amda = amda_['xyz'][i]

    time_sscweb = sscweb_['time'][i].strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    if cdaweb_ is not None:
      time_cdaweb = cdaweb_['time'][i].strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    if jpl_ is not None:
      time_jpl = jpl_['time'][i].strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    if amda_ is not None:
      time_amda = amda_['time'][i].strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    u = 'R_E'
    if i == 0:
      print(f'  First values in {u}:')
    else:
      print(f'  Last values in {u}:')

    print('            {0:8s} {1:8s} {2:8s}'.format(f' X_{frame}', f' Y_{frame}', f' Z_{frame}'))
    print('    SSCWeb: {0:<8.4f} {1:<8.4f} {2:<8.4f} {3:s}'.format(*xyz_sscweb, time_sscweb))
    if cdaweb_ is not None:
      print('    CDAWeb: {0:<8.4f} {1:<8.4f} {2:<8.4f} {3:s}'.format(*xyz_cdaweb, time_cdaweb))
    if jpl_ is not None:
      print('       JPL: {0:<8.4f} {1:<8.4f} {2:<8.4f} {3:s}'.format(*xyz_jpl, time_jpl))
    if amda_ is not None:
      print('      AMDA: {0:<8.4f} {1:<8.4f} {2:<8.4f} {3:s}'.format(*xyz_amda, time_amda))

def interp(timei, time, xyz):
  from scipy.interpolate import interp1d

  # Interpolate CDAWeb data onto the union of timestamps
  interp_func = interp1d(
    [t.timestamp() for t in time],
    xyz,
    axis=0,
    bounds_error=False,
    fill_value="extrapolate"
  )

  return interp_func([t.timestamp() for t in timei])

def compute_diffs(info1, info2):

  print(f"  Computing differences between {info1['name']} and {info2['name']}")

  info1['time'] = np.array([t.replace(tzinfo=None) for t in info1['time']])
  info2['time'] = np.array([t.replace(tzinfo=None) for t in info2['time']])

  def _interp_dts(times1, times2):
    # For each time in time1, compute minimum absolute difference to any time in time2
    dt_mins = np.empty(len(times1), dtype=float)
    for i, time1 in enumerate(times1):
      dt_mins[i] = np.min(np.abs(time1-times2)).total_seconds()
    return dt_mins

  dt1_mins = _interp_dts(info1['time'], info2['time'])
  print(f"    {info1['name']} max dt to {info2['name']}: {np.max(dt1_mins):.3f} s")
  dt2_mins = _interp_dts(info2['time'], info1['time'])
  print(f"    {info2['name']} max dt to {info1['name']}: {np.max(dt2_mins):.3f} s")

  # Compute time differences in seconds between consecutive samples in each source
  dt1 = np.diff([t.timestamp() for t in info1['time']])
  dt2 = np.diff([t.timestamp() for t in info2['time']])

  # Round each dt to 3 decimal places
  dt1 = np.round(dt1, 3)
  dt2 = np.round(dt2, 3)

  # Compute histogram (as Counter) for each
  hist1 = Counter(dt1)
  hist2 = Counter(dt2)
  # Sort histogram by frequency (descending)
  hist1 = dict(sorted(hist1.items(), key=lambda item: item[1], reverse=True))
  hist2 = dict(sorted(hist2.items(), key=lambda item: item[1], reverse=True))
  # Keep first 10 most common time differences
  hist1 = dict(list(hist1.items())[:10])
  hist2 = dict(list(hist2.items())[:10])

  print(f"    {info1['name']} dt histogram (s): {dict(hist1)}")
  print(f"    {info2['name']} dt histogram (s): {dict(hist2)}")

  if interp_times:
    # Only keep times where min dt to nearest time in other source is less
    # than 10 seconds
    print("    Filtering timestamps with dt to nearest time in other source >= 10 seconds.")
    mask1 = dt1_mins < 10
    time1 = info1['time'][mask1]
    xyz1  = info1['xyz'][mask1]
    print(f"    {info1['name']} has {len(time1)} timestamps after filtering (originally {len(info1['time'])})")
    mask2 = dt2_mins < 10
    time2 = info2['time'][mask2]
    xyz2 = info2['xyz'][mask2]
    print(f"    {info2['name']} has {len(time2)} timestamps after filtering (originally {len(info2['time'])})")

    print("    Computing union of timestamps.")
    # Union of timestamps from CDAWeb and SSCWeb
    union_times = set(time1).union(set(time2))
    union_times = sorted(union_times)

    print(f"    Interpolating to {len(union_times)} timestamps.")
    info1['xyz_i'] = interp(union_times, time1, xyz1)
    info1['time_i'] = union_times
    info2['xyz_i'] = interp(union_times, time2, xyz2)
    info2['time_i'] = union_times

    time_key = 'time_i'
    xyz_key = 'xyz_i'
  else:
    # Common timestamps
    print("    Computing common timestamps.")
    common_times = set(info2['time']).intersection(set(info1['time']))
    common_times = sorted(common_times)
    common_times = sorted(common_times)
    print(f"    {len(common_times)} common timestamps.")

    # Find the indices of common times
    idx1 = np.nonzero(np.isin(info1['time'], common_times))[0]
    idx2 = np.nonzero(np.isin(info2['time'], common_times))[0]

    info1['xyz_c'] = np.array(info1['xyz'][idx1])
    info1['time_c'] = common_times
    info2['xyz_c'] = np.array(info2['xyz'][idx2])
    info2['time_c'] = common_times
    time_key = 'time_c'
    xyz_key = 'xyz_c'

    if not interp_times and print_common_times:
      for i in range(len(common_times)):
        t_str = common_times[i].strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        print(f"{t_str} | {info1['name']}: {info1[xyz_key][i]} | {info2['name']}: {info2[xyz_key][i]}")

  nt = len(info1[time_key]) # Number of timestamps
  Δθ = np.full(nt, np.nan)
  Δr = np.full(nt, np.nan)
  t = np.empty(nt, dtype='datetime64[ns]')
  t[:] = np.datetime64('NaT')
  # Average r
  r_ave = 0.5*(np.linalg.norm(info1[xyz_key], axis=1) + np.linalg.norm(info2[xyz_key], axis=1))

  for i in range(nt):
    t[i] = np.datetime64(info1[time_key][i])
    Δr[i] = np.linalg.norm(info1[xyz_key][i] - info2[xyz_key][i])
    # d = denominator for angle calculation
    # Δθ[i] = arccos [ (a·b)/(|a|*|b|) ] = arccos (n/d)
    n = np.dot(info1[xyz_key][i], info2[xyz_key][i])
    d = np.linalg.norm(info1[xyz_key][i])*np.linalg.norm(info2[xyz_key][i])
    if np.abs(n) > np.abs(d):
      if np.abs(n) - np.abs(d) > 1e-10:
        raise ValueError(f"|n| - |d| > 1e-10. n = {n:.16f} d = {d:.16f}, n-d = {n-d:.16f}")
      if cos_warnings:
        # If n == d, then the angle is 0 degrees, but numpy gives
        # RuntimeWarning: invalid value encountered in arccos. Did rounding
        # cause n > d?
        # TODO: Consider https://github.com/sunpy/sunpy/pull/7530#issuecomment-2020890282
        wmsg =  f"    Warning: {t[i]}:\n"
        wmsg +=  "      |n| > |d| in Δθ[i] = arccos [ (a·b)/(|a|*|b|) ] = arccos (n/d)\n"
        wmsg += f"      a = {info1[xyz_key][i]}\n"
        wmsg += f"      b = {info2[xyz_key][i]}\n"
        wmsg += f"      n = {n:.16f}\n"
        wmsg += f"      d = {d:.16f}\n"
        wmsg += f"      n/d = {n/d}\n"
        wmsg +=  "      Using Δθ = 0\n"
        print(wmsg)
      Δθ[i] = 0
    else:
      Δθ[i] = (180/np.pi)*np.arccos(n/d)

    #import pdb; pdb.set_trace()

    if interp_times and print_interp_vals:
      t_str = info1[time_key][i].strftime('%Y-%m-%dT%H:%M:%S.%fZ')
      print(f"    t = {t_str} | Δr = {Δr[i]:.5f} [R_E] | Δ∠: {Δθ[i]:.5f}°")

  return t, r_ave, Δr, Δθ

def plot_xyz(ax, info1, info2, t, r_ave):

  print(f"  Plotting xyz for {info1['name']} and {info2['name']}")

  colors = ['r', 'g', 'b']
  component = ['X', 'Y', 'Z']

  for c in range(3):
    label1 = f"{info1['name']}/{component[c]}"
    label2 = f"{info2['name']}/{component[c]}"
    ax.plot(info1['time'], info1['xyz'][:,c],
            label=label1, lw=2, linestyle='-', color=colors[c])
    ax.plot(info2['time'], info2['xyz'][:,c],
            label=label2, lw=3, linestyle='--', color=colors[c])

  label = '$\\overline{r}$'
  ax.plot(t, r_ave, label=label, lw=2, linestyle='-', color='k')

  _adjust_y_range(ax, gap_fraction=1)
  ax.set_ylabel('$R_E$', rotation=0)
  ax.grid()

  ax.legend(**{**legend_kwargs, 'ncol': 4})
  ax.set_xticklabels([])

def plot_diffs(ax, t, r_ave, Δr, Δθ):

  print("  Plotting diffs")

  lw = 2 # line width

  Δr_rel = Δr/r_ave
  Δr_rel_max = np.nanmax(Δr_rel)
  #A = f'{Δr_rel_max:.1e}'.split('e')[0]
  #e = str(int(np.floor(np.log10(Δr_rel_max))))
  #Δr_rel_max_str = f" ($Δr_{{\\rm{{max}}}}/\\overline{{r}}$ = {A}·$10^{{{e}}}$)"
  Δr_rel_max_str = f" (max= 1/{1/Δr_rel_max:.0f})"

  Δr_max = np.nanmax(Δr)
  Δr_max_str = f'($|Δ\\mathbf{{r}}|_{{\\rm{{max}}}} = {Δr_max*R_E:.1f}$ [km])'

  print(f"    Δr_max = {Δr_max:.5f} [R_E]")
  print(f"    Δr_max = {Δr_max*R_E:.1f} [km]")

  ax.plot(t, Δr, 'r-', lw=lw, label=f'$|Δ\\mathbf{{r}}|/R_E$ {Δr_max_str}')
  ax.plot(t, Δr_rel, 'b-', lw=lw, label=f'$|Δ\\mathbf{{r}}|/\\overline{{r}}$ {Δr_rel_max_str}')
  ax.plot(t, Δθ, 'g-', lw=lw, label='$Δθ$ [deg]')

  _adjust_y_range(ax, bottom=0, gap_fraction=1)
  ax.legend(**legend_kwargs)
  ax.grid()

def _figprep():
  gs = plt.gcf().add_gridspec(2)
  axes = gs.subplots(sharex=True)

  for ax in axes:
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # Remove minor tick lines next to y-axis numbers
    ax.tick_params(axis='y', which='minor', length=0)
    # Remove tick lines next to y-axis numbers
    ax.tick_params(axis='y', length=0)
    ax.grid(which='minor', linestyle=':', linewidth=0.5, color=[0.75]*3)
    ax.minorticks_on()

  return axes

def _savefigs(fname):
  for fmt in ['svg', 'png', 'pdf']:
    kwargs = {'bbox_inches': 'tight'}
    if fmt == 'png':
      kwargs['dpi'] = 300

    if fmt == 'pdf':
      fname_full = f'figures/ephemeris/{fname}.{fmt}'
    else:
      fname_full = f'figures/ephemeris/{fmt}/{fname}.{fmt}'
    os.makedirs(os.path.dirname(fname_full), exist_ok=True)
    print(f"  Writing {fname_full}")
    plt.savefig(fname_full, bbox_inches='tight')
  plt.close()

def _adjust_xlim(axes):
  from datetime import timedelta
  import matplotlib.dates as mdates
  from datetick import datetick

  datetick('x')#, debug=True)

  if False:
    # Subtract 1 minute to the x-axis limits to make sure the first tick is shown
    m = timedelta(minutes=1)
    xlim = [mdates.num2date(xlim) for xlim in axes[0].get_xlim()]
    hm = timedelta(hours=1, minutes=1)
    # Round to the next hour + 1 minute
    last = xlim[1].replace(minute=0, second=0, microsecond=0) + hm
    axes[0].set_xlim(xlim[0] - m, last)
    axes[1].set_xlim(xlim[0] - m, last)

    labels = axes[1].get_xticklabels()
    if labels:
      if "\n" in labels[0].get_text():
        labels[0].set_horizontalalignment('left')
      if "\n" in labels[-1].get_text():
        labels[-1].set_horizontalalignment('right')

def _adjust_y_range(ax, gap_fraction=0.25, bottom=None):
  # Adjust the y-axis range to add a gap at the top so legend does not overlap
  # with the data. Seems that the needed gap could be computed.
  ylim = ax.get_ylim()
  yticks = ax.get_yticks()
  gap = gap_fraction * (yticks[1] - yticks[0])
  if bottom is None:
    bottom = ylim[0]
  ax.set_ylim(bottom, ylim[1] + gap)

def _plot(axes, satellite, info1, info2):
  fname = f"{satellite}_{info1['name']}-{info1['frame_name']}_vs_{info2['name']}-{info2['frame_name']}"
  axes[0].set_title(f"{satellite} {info1['name']}/{info1['frame_name']} vs. {info2['name']}/{info2['frame_name']}")
  t, r_ave, Δr, Δθ = compute_diffs(info1, info2)
  plot_xyz(axes[0], info1, info2, t, r_ave)
  plot_diffs(axes[1], t, r_ave, Δr, Δθ)
  _adjust_xlim(axes)
  _savefigs(fname)

satellites = infos()
for satellite in satellites:

  if len(satellites_only) > 0 and satellite not in satellites_only:
    continue

  infos_ = infos(satellite)

  if False:
    # Compare SSCWeb J2K with SSCWeb GEI
    for info in infos_['sscweb']:
      if info is not None and info.get('frame_name') == 'J2K':
        sscweb_j2k = sscweb(info, logging=hapi_logging)
      if info is not None and info.get('frame_name') == 'GEI':
        sscweb_gei = sscweb(info, logging=hapi_logging)

    axes = _figprep()
    _plot(axes, satellite, sscweb_j2k, sscweb_gei)

  for i in range(len(infos_['sscweb'])):

    frame = infos_['sscweb'][i]['frame_name']

    print(f"\n{satellite} {frame}")

    # Adds xyz and time to info dict
    sscweb_ = sscweb(infos_['sscweb'][i], logging=hapi_logging)
    cdaweb_ = cdaweb(infos_['cdaweb'][i], logging=hapi_logging)
    amda_ = amda(infos_['amda'][i], logging=hapi_logging)
    jpl_ = jpl(infos_['sscweb'][i], logging=hapi_logging)

    print_first_last_(sscweb_, cdaweb_, amda_, jpl_)

    sources = [sscweb_, cdaweb_, amda_, jpl_]

    for i, source1 in enumerate(sources):
      for j, source2 in enumerate(sources):
        if i >= j or source1 is None or source2 is None:
          continue
        s1 = f"{source1['name']}/{source1['frame_name']}"
        s2 = f"{source2['name']}/{source2['frame_name']}"
        print(f"Comparing {s1} with {s2}")
        axes = _figprep()
        _plot(axes, satellite, source1, source2)
