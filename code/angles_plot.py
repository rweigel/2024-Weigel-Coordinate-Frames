import os

import numpy
from matplotlib import pyplot as plt

import utilrsw

#run = 'delta=10days_20100101-20101231'
#run = 'delta=10minutes_20101221-20101223'
#run = 'delta=1days_20101221-20101223'
#run = 'delta=10minutes_20101221-20101223'
run = 'delta=1days_20100101-20150101'

in_file = os.path.join('data','angles', f'{run}.pkl')
out_dir = os.path.join('figures', 'angles', run)

def fig_save(fname):
  import os
  for fmt in ['svg', 'png', 'pdf']:
    kwargs = {'bbox_inches': 'tight'}
    if fmt == 'png':
      kwargs['dpi'] = 300

    if fmt == 'pdf':
      fname_full = os.path.join(out_dir, f'{fname}.{fmt}')
    else:
      fname_full = os.path.join(out_dir, fmt, f'{fname}.{fmt}')

    os.makedirs(os.path.dirname(fname_full), exist_ok=True)
    print(f"  Writing {fname_full}")
    plt.savefig(fname_full, bbox_inches='tight')
  plt.close()

def fig_prep():
  gs = plt.gcf().add_gridspec(3)
  axes = gs.subplots(sharex=True)
  return axes

def plot(df, tranform_str):

  line_map = {
    'geopack_08_dp': ['black', '-'],
    'spacepy': ['blue', '-'],
    'spacepy-irbem': ['blue', '--'],
    'spiceypy1': ['red', '-'],
    'spiceypy2': ['red', '--'],
    'sunpy': ['orange', '-'],
    'pyspedas': ['green', '-'],
    'sscweb': ['purple', '-'],
    'cxform': ['brown', '-'],
    '|max-min|': ['black', '-']
  }

  axes = fig_prep()

  for column in df['values'].columns:
    lib = column
    axes[0].plot(df['values'].index, df['values'][column],
                 label=lib, color=line_map[lib][0], linestyle=line_map[lib][1])
  axes[0].grid(True)
  axes[0].set_ylabel(tranform_str)
  axes[0].legend()

  for column in df['diffs'].columns:
    if column == '|max-min|':
      continue

    stat = utilrsw.format_exponent(numpy.max(numpy.abs(df['diffs'][column])), 0)
    label = f"{column} (${stat}$)"
    axes[1].plot(df['diffs'].index, df['diffs'][column],
                 label=label, color=line_map[column][0], linestyle=line_map[column][1])
  axes[1].grid(True)
  axes[1].set_ylabel('Diff. relative to geopack_08_dp [deg]')
  axes[1].legend()

  axes[2].plot(df['diffs'].index, df['diffs']['|max-min|'],
               label='|max-min|', color=line_map['|max-min|'][0], linestyle=line_map['|max-min|'][1])
  axes[2].grid(True)
  axes[2].set_ylabel('|max-min| [deg]')
  axes[2].legend()

  axes[2].set_xlabel('Year')


utilrsw.mpl.plt_config()
data = utilrsw.read(in_file)

for transform_key in list(data.keys()):
  df = data[transform_key]
  tranform_str = transform_key.split('_')
  tranform_str = fr"$\angle$ ($Z_{{{tranform_str[0]}}}$, $Z_{{{tranform_str[1]}}}$)"

  plot(df, tranform_str)
  fig_save(f'{transform_key}')
