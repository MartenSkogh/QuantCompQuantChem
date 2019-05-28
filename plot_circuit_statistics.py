#!/usr/bin/env python3
#-*- encoding: utf-8 -*-

import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib2tikz 

rc('font',**{'family':'serif','serif':['Computer Modern']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 25})
plt.rcParams['lines.markersize'] = 10
#plt.style.use("ggplot")

class Data_set():
    pass


filename = 'Circuits/circuit_table.csv'

raw_data = np.genfromtxt(filename, dtype=None, delimiter=',', encoding='utf-8')
#sorted_data = raw_data[raw_data[:,0].argsort()]
print(raw_data[0][0])
#molecules = list(set(raw_data[:,0]))
#basis_sets = list(set(raw_data[:,1]))
#freeze_index = list(set(raw_data[:,2]))
#remove_index = list(set(raw_data[:,3]))
#simulators = list(set(raw_data[:,4]))
#mappings = list(set(raw_data[:,5]))
#var_methods = list(set(raw_data[:,6]))
#var_depth = list(set(raw_data[:,7]))
#classical_opts = list(set(raw_data[:,8]))

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

start = 0
number_lines = 4
small_offset = 5 
atom_offsets = [0, 120, 180, 300]

markers = ['o', '^', 's', 'd']
colors = ['xkcd:cerulean', 'xkcd:leaf green', 'orange', 'red']

for atom_offset, color in zip(atom_offsets, colors):
    for i, marker in zip(range(start,start + number_lines), markers):
        start_idx = i*small_offset + atom_offset
        end_idx = start_idx + small_offset
        plt.semilogy([int(var_depth[7]) for var_depth in raw_data[start_idx:end_idx]],
                 [gates[-2] for gates in raw_data[start_idx:end_idx]],
                 linestyle='-', marker=marker, color=color)
        
        fc = lambda fl: 'core not frozen' if fl == '"[]"' else 'core frozen'

        legends.append(', '.join([molecule_name[raw_data[start_idx][0]], raw_data[start_idx][6]]))#raw_data[start_idx][1], raw_data[start_idx][5], raw_data[start_idx][6], raw_data[start_idx][9], fc(raw_data[start_idx][2])]))
    
plt.xticks([1,2,3,4,5])
plt.title('Number of Gates vs Variational Layers')
plt.xlabel('Variational Circuit Layers')
plt.ylabel('Number of Gates')
plt.legend(legends, loc='center left', bbox_to_anchor=(1, 0.5),
          fancybox=True, shadow=False, ncol=1)
#plt.legend(legends)
plt.grid(True)

image_filename = '../../Images/variational_depth.pdf'
plt.savefig(image_filename, bbox_inches='tight')


#
# FERMIONIC MAPPING
#

fig = plt.figure()
ax = plt.subplot('111')
fig.set_size_inches(14,10)
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

legends = []

start = 0
number_lines = 3
small_offset = 20
atom_offsets = [0, 120, 180, 300]

markers = ['o', '^', 's', 'd']
colors = ['xkcd:cerulean', 'xkcd:leaf green', 'orange', 'red']

for atom_offset, color in zip(atom_offsets, colors):
    for i, marker in zip(range(start,start + number_lines), markers):
        start_idx = i*small_offset + atom_offset
        end_idx = start_idx + small_offset
        plt.semilogy(raw_data[start_idx][-3],raw_data[start_idx][-2],
                 linestyle='',marker=marker, color=color)
        

        legends.append(', '.join([molecule_name[raw_data[start_idx][0]], map_name[raw_data[start_idx][5]]]))#raw_data[start_idx][1], raw_data[start_idx][5], raw_data[start_idx][6], raw_data[start_idx][9], fc(raw_data[start_idx][2])]))
    
#plt.xticks([1,2,3,4,5])
plt.title('Number of Gates vs Fermionic Mapping')
plt.xlabel('Number Qubits')
plt.ylabel('Number of Gates')
plt.legend(legends, loc='center left', bbox_to_anchor=(1, 0.5),
          fancybox=True, shadow=False, ncol=1)
#plt.legend(legends)
plt.grid(True)

image_filename = '../../Images/fermionic_mapping.pdf'
plt.savefig(image_filename, bbox_inches='tight')

plt.show()
