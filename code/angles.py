import os
import datetime
import itertools

import numpy
import pandas

import hxform
import utilrsw

log = utilrsw.logger(console_format='%(message)s')

def libs(csys_in, csys_out, excludes=None):
  lib_infos = hxform.info.known_libs(info=True)
  libs_avail = []
  for i, lib_info in enumerate(lib_infos):
    lib = lib_info['name']
    if excludes is not None and lib in excludes:
      continue
    if csys_in not in lib_info['systems'] or csys_out not in lib_info['systems']:
      log.warning(f"Skipping {lib} because it does not support both {csys_in} and {csys_out} systems")
      continue
    libs_avail.append(lib)
  return libs_avail


def angles(to, tf, axis, delta, libs, transform_kwargs):

  if axis not in ['x', 'y', 'z']:
    raise ValueError(f"Invalid axis: {axis}. Must be one of 'x', 'y', or 'z'.")

  t = hxform.time_array(to, tf, delta)
  t_dts = hxform.ints2datetime(t)
  Δθ = numpy.full((t.shape[0], len(libs)), numpy.nan)
  δψ = numpy.full((t.shape[0], len(libs)), numpy.nan) # Uncertainty in ψ ≔ Δθ

  for i, lib in enumerate(libs):
    #log.info(f"Processing {lib}...")
    transform_kwargs['lib'] = lib

    if axis == 'x':
      p_in = numpy.array([1., 0., 1.])
    if axis == 'y':
      p_in = numpy.array([0., 1., 0.])
    if axis == 'z':
      p_in = numpy.array([0., 0., 1.])

    p_out = hxform.transform(p_in, t, **transform_kwargs)

    n = numpy.dot(p_out, p_in)
    d = numpy.linalg.norm(p_out, axis=1)*numpy.linalg.norm(p_in)
    Δθ[:, i] = (180.0/numpy.pi)*numpy.arccos(n/d)

    ε = numpy.spacing(numpy.abs(Δθ[:, i]).max())
    if lib == 'sscweb':
      # values reported to two decimal places => max uncertainty of 0.005
      ε = 0.005
    δψ[:, i] = _angle_uncert(p_in, p_out, n, ε)

    if lib.startswith('spiceypy1'):
      years = numpy.array([dt.year for dt in t_dts])
      mask = (years < 1990) | (years >= 2020)
      Δθ[mask, i] = numpy.nan

    if lib.startswith('spiceypy2'):
      years = numpy.array([dt.year for dt in t_dts])
      mask = (years < 1990) | (years >= 2030)
      Δθ[mask, i] = numpy.nan

  return t_dts, Δθ, δψ


def _angle_uncert(p_in, p_out, n, ε):
  """Estimate uncertainty in angular difference Δθ using propagation of error.
  ψ ≔ Δθ = acos(n/d)
  assume d = 1

  dψ = -dn/(sqrt(1-n^2))

  n = dot(p_out, p_in) = p_out_x*p_in_x + p_out_y*p_in_y + p_out_z*p_in_z

  dn = ( |∂n/∂p_out_x| * Δp_out_x
       + |∂n/∂p_out_y| * Δp_out_y
       + |∂n/∂p_out_z| * Δp_out_z
       + |∂n/∂p_in_x| * Δp_in_x
       + |∂n/∂p_in_y| * Δp_in_y
       + |∂n/∂p_in_z| * Δp_in_z)

  ∂n/∂p_out_x = p_in_x
  ∂n/∂p_out_y = p_in_y
  ∂n/∂p_out_z = p_in_z
  ∂n/∂p_in_x  = p_out_x
  ∂n/∂p_in_y  = p_out_y
  ∂n/∂p_in_z  = p_out_z

  Let ε ≔ Δp_out_i = Δp_in_i for i = x, y, z

  Then
  dn =  |p_in_x| * ε
      + |p_in_y| * ε
      + |p_in_z| * ε
      + |p_out_x| * ε
      + |p_out_y| * ε
      + |p_out_z| * ε

  Compute uncertainty in ψ using propagation of error formula:
  |δψ| = |dn/(sqrt(1-n^2))|
  |δψ| = |-ε/(sqrt(1-n^2))| (|p_in_x| + |p_in_y| + |p_in_z| + |p_out_x| + |p_out_y| + |p_out_z|)

  or more conservative estimate:

  |δψ| = |-ε/(sqrt(1-n^2))| sqrt((1 + |p_in_x|^2 + |p_in_y|^2 + |p_in_z|^2))
  """

  return (180.0/numpy.pi)*(ε/(1+n**2))*(numpy.sum(numpy.abs(p_in)) + numpy.sum(numpy.abs(p_out), axis=1))


def _print_and_write(xform, dfs, dir_table, libs_avail):

  df_info = {
    'values': ['Δθ', 'in degrees'],
    'uncert': ['Δθ-uncert', '(uncert in Δθ in degrees)'],
    'diffs': ['Δθ-diff', 'Δθ - (Δθ geopack_08_dp) in degrees']
  }

  for key in df_info.keys():
    df_str = dfs[xform][key].to_string()
    log.info(f"\n{xform} {df_info[key][1]}\n{df_str}")
    file_table = os.path.join(dir_table, f'{xform}_{df_info[key][0]}.txt')
    print(f"Writing {file_table}")
    utilrsw.write(file_table, df_str)

    print(dfs[xform][key].describe())

if True:
  axis = 'z'
  delta = {'days': 1}
  to = datetime.datetime(2010, 1, 1, 0, 0, 0)
  tf = datetime.datetime(2015, 1, 1, 0, 0, 0)
  excludes = ['sscweb', 'cxform']

if False:
  axis = 'z'
  delta = {'days': 1}
  to = datetime.datetime(2010, 1, 1, 0, 0, 0)
  tf = datetime.datetime(2010, 1, 3, 0, 0, 0)
  excludes = ['sscweb', 'cxform']

if False:
  axis = 'z'
  delta = {'minutes': 10}
  to = datetime.datetime(2010, 12, 21, 0, 0, 0)
  tf = datetime.datetime(2010, 12, 23, 0, 0, 0)
  excludes = ['cxform']

to_str = datetime.datetime.strftime(to, '%Y%m%d')
tf_str = datetime.datetime.strftime(tf, '%Y%m%d')
delta_unit = list(delta.keys())[0]
delta_str = f'{delta[delta_unit]}{delta_unit}'

run = f'{axis}-delta={delta_str}_{to_str}-{tf_str}'
dir_table = os.path.join('figures', 'angles', run)
file_out = os.path.join('data', 'angles', f'{run}.pkl')

transform_kwargs = { 'ctype_in': 'car', 'ctype_out': 'car'}

csys_list = ['GEO', 'MAG', 'GSE', 'GSM']
combinations = list(itertools.combinations(csys_list, 2))

dfs = {}
df_deltas = {}
for combination in combinations:
  csys_in, csys_out = combination

  transform_kwargs['csys_in'] = csys_in
  transform_kwargs['csys_out'] = csys_out
  libs_avail = libs(csys_in, csys_out, excludes=excludes)
  t_dts, Δθ, δψ = angles(to, tf, axis, delta, libs_avail, transform_kwargs)

  xform = f"{csys_in}_{csys_out}"
  dfs[xform] = {'values': None, 'diffs': None, 'uncert': None}

  index = pandas.to_datetime(t_dts)
  dfs[xform]['values'] = pandas.DataFrame(Δθ, index=index, columns=libs_avail)
  dfs[xform]['uncert'] = pandas.DataFrame(δψ, index=index, columns=libs_avail)

  # Compute diff DataFrame
  columns_diff = libs_avail.copy()
  columns_diff.remove('geopack_08_dp')
  dfs[xform]['diffs'] = pandas.DataFrame(numpy.nan, index=index, columns=columns_diff)
  if 'geopack_08_dp' in libs_avail:
    for lib in libs_avail:
      if lib != 'geopack_08_dp':
        diff = dfs[xform]['values'][lib] - dfs[xform]['values']['geopack_08_dp']
        dfs[xform]['diffs'][lib] = diff

  max_min = dfs[xform]['values'].max(axis=1) - dfs[xform]['values'].min(axis=1)
  dfs[xform]['diffs']['|max-min|'] = numpy.abs(max_min)

  _print_and_write(xform, dfs, dir_table, libs_avail)

print(f"Writing {file_out}")
utilrsw.write(file_out, dfs)

utilrsw.rm_if_empty('angles.errors.log')
