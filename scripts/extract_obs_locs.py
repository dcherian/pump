import subprocess

johnson_sections = [-95, -110, ]
indir = 'TPOS_MITgcm/HOLD/fix_record/'
outdir = 'TPOS_MITgcm/HOLD/obs_subset/'

infiles = indir + 'Day_*.nc'

print('Extracting surface variables...')
subprocess.call("ncrcat -d depth,0 " + infiles + ' -o surface.nc', shell=True)

print('Extracting Johnson sections...')

print('Extracting TAO timeseries...')
