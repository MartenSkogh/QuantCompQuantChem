import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib2tikz 
import itertools

rc('font',**{'family':'serif','serif':['Computer Modern']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rcParams['lines.markersize'] = 10


molecules = ['H2']
var_methods = ['UCCSD', 'RY', 'RYRZ', 'SwapRZ']
var_depths = ['2']#[1,2,3,4,5]

csv_name = '/36143.csv'
paths = [item for item in itertools.product(*[molecules, var_methods, var_depths])]

stringfy = lambda arr: [str(element) for element in arr]



filenames = [('Results/Variational_Layers/' + '/'.join(stringfy(path)) + csv_name) for path in paths]


molecule_name = {'H2':'H$_2$', 'LiH':'LiH', 'BeH2':'BeH$_2$', 'H2O':'H$_2$O'}
map_name = {'jordan_wigner':'Jordan-Wigner', 'parity':'Parity', 'bravyi_kitaev':'Bravyi-Kitaev'}

#
# VARIATIONAL DEPTH
#

fig = plt.figure()
ax = plt.subplot('111')
fig.set_size_inches(14,10)
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

legends = []

markers = ['o', '^', 's', 'd']
colors = ['xkcd:cerulean', 'xkcd:leaf green', 'orange', 'red', 'xkcd:medium pink']

for filename, path, color in zip(filenames, paths, colors):
    print(filename)

    raw_data = np.genfromtxt(filename, delimiter=',')
    sorted_data = raw_data[raw_data[:,0].argsort()]


    distance = sorted_data[:,0]
    hf_energy = sorted_data[:,2]
    vqe_energy = sorted_data[:,4]

    min_energy_idx = vqe_energy.argmin()
    min_energy = vqe_energy[min_energy_idx]
    min_distance = distance[min_energy_idx]

    min_hf_energy_idx = hf_energy.argmin()
    min_hf_energy = hf_energy[min_hf_energy_idx]
    min_hf_distance = distance[min_hf_energy_idx]

    #plt.plot(distance, hf_energy)
    plt.plot(distance, vqe_energy ,color=color)


    legends.append(', '.join(stringfy(path)))

plt.title('HF vs VQE Energy H$_2$O')
plt.xlabel('Atomic Seperation [Å]')
plt.ylabel('Energy [Hartree]')

plt.legend(legends)
plt.grid(True)
plt.show()
