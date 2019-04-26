#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# import common packages
import sys
import shutil
import numpy as np

# Fromt the Qiskit base package
from qiskit import Aer
from qiskit import QuantumRegister, QuantumCircuit

# lib from Qiskit Aqua
from qiskit.aqua import Operator, QuantumInstance
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.aqua.components.optimizers import COBYLA

# lib from Qiskit Aqua Chemistry
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock

def print_cmd_help():
    size = shutil.get_terminal_size()
    cols = size.columns
    break_filler = cols*'-'
    help_string = (
'''
OVERVIEW:
Ground energy calculations using Qiskit framework.
Usage: python3 ground_state_energy.py [-option=value]

{0}

OPTIONS:

    --distance, -d      Distance between atoms. (Defualt is 1.0 Å)

    --backend, -b       The backend to use for simulation. (Default is QASM simualtor)
                        Available backends are:
                            qasm_simulator
                            statevector_simulator
                            unitary_simulator

    --output-file, -o   File to write result to. (Default is none)
                        Appends the data to the result. Creates a file if one does not exist.

    --optimizer, -O     Classical optimizer to use. (Default is COBYLA)

{0}

EXAMPLE: 

Calculae the ground state energy for LiH where the Li and H atoms are seperated by 1.5 Å using COBYLA as the classical optimizer and the QASM backend.

$ python3 ground_state_energy.py -d=1.5 -b=qasm -O=COBYLA
'''.format(break_filler))
    print(help_string)


# Setting the atomic distance between atoms from command line argument
default_atomic_distance = 1.0
defualt_backend_string = 'qasm_simulator' # Default backend
defualt_optimizer_string = 'COBYLA'
default_data_file = None

optimizer_string = None
atomic_distance = None
backend_string = None
data_file = None

if len(sys.argv) == 1:
    print_cmd_help()
    quit()
else:
    pass
    #print(sys.argv)


for argument in sys.argv:
    if '--help' in argument:
        print_cmd_help()
        quit()
    if '--distance=' in argument or '-d=' in argument:
        atomic_distance = float(argument.split('=')[1])

    elif '--backend=' in argument or '-b=' in argument:
        given_string = argument.split('=')[1]
        if given_string in ['qs','qasm', 'qasm_simulator']:
            backend_string = 'qasm_simulator'
        elif given_string in ['svs','statevector', 'statevector_simulator']: 
            backend_string = 'statevector_simulator'
        elif given_string in ['us','unitary', 'unitary_simulator']:
            backend_string = 'unitary_simulator'
        else:
            print('WARNING: Incorrect backend given!')
            print('Available backends are: ')
            print('\tqasm_simulator')
            print('\tstatevector_simulator')
            print('\tunitary_simulator')
            print('Continuing with default backend: {}'.format(backend_string))

    elif '--output-file=' in argument or '-o=' in argument:
        data_file = argument.split('=',1)[1]

    elif '--optimizer=' in argument or '-O=' in argument:
        optimizer_string = argument.split('=')[1]
        # TODO: Print all available optimizers:
    elif '--atom=' in argument:
        atom_string = argument.split('=')[1]

# Use defualt values if none given
if not atomic_distance and not atom_string:
    print('Warning: No distance between atoms given.')
    print('Using default value: {} Å'.format(default_atomic_distance))
    atomic_distance = default_atomic_distance

if not backend_string:
    backend_string = defualt_backend_string

if not optimizer_string:
    optimizer_string = defualt_optimizer_string

if not data_file:
    data_file = default_data_file

if not atom_string:
    atom_string = 'Li .0 .0 .0; H .0 .0 %f' % (atomic_distance)
else:
    atoms = atom_string.split(';')
    coords = []
    for atom in atoms:
        coords.append([float(c) for c in atom.split()[1:]])
    atomic_distance = np.sqrt(sum((np.array(coords[0])-np.array(coords[1]))**2))

print('==== Simulation setup ====')
print(f'    Atomic Setup: {atom_string}')
print(f'    Distance: {atomic_distance}')
print(f'    Backend: {backend_string}')
print(f'    Classical Optimizer: {optimizer_string}')
print(f'    Output file: {data_file}')


backend_string_short = backend_string.split('_')[0]

# using driver to get fermionic Hamiltonian
# PySCF example
driver = PySCFDriver(atom=atom_string, 
                     unit=UnitsType.ANGSTROM,
                     charge=0, spin=0, basis='sto3g')
molecule = driver.run()

molecule_formula = ''.join(molecule.atom_symbol)

print('Number orbitals for {} is {}'.format(molecule_formula,molecule.num_orbitals))

# please be aware that the idx here with respective to original idx
freeze_list = [] #[0]
remove_list = [] #[-3,-2] # negative number denotes the reverse order
map_type = 'parity'

h1 = molecule.one_body_integrals
h2 = molecule.two_body_integrals
nuclear_repulsion_energy = molecule.nuclear_repulsion_energy

num_particles = molecule.num_alpha + molecule.num_beta
num_spin_orbitals = molecule.num_orbitals * 2
# Should the nuclear repulsion energy be added or removed?
print("HF energy - nuclear repulsion: {}".format(molecule.hf_energy - molecule.nuclear_repulsion_energy))
print("HF energy + nuclear repulsion: {}".format(molecule.hf_energy + molecule.nuclear_repulsion_energy))
print("# of electrons: {}".format(num_particles))
print("# of spin orbitals: {}".format(num_spin_orbitals))

# prepare full idx of freeze_list and remove_list
# convert all negative idx to positive
remove_list = [x % molecule.num_orbitals for x in remove_list]
freeze_list = [x % molecule.num_orbitals for x in freeze_list]
# update the idx in remove_list of the idx after frozen, since the idx of orbitals are changed after freezing
remove_list = [x - len(freeze_list) for x in remove_list]
remove_list += [x + molecule.num_orbitals - len(freeze_list)  for x in remove_list]
freeze_list += [x + molecule.num_orbitals for x in freeze_list]

print('Freeze list:')
print(freeze_list)
print('Remove list:')
print(remove_list)

# prepare fermionic hamiltonian with orbital freezing and eliminating, and then map to qubit hamiltonian
# and if PARITY mapping is selected, reduction qubits
energy_shift = 0.0
qubit_reduction = True if map_type == 'parity' else False

ferOp = FermionicOperator(h1=h1, h2=h2)
if len(freeze_list) > 0:
    ferOp, energy_shift = ferOp.fermion_mode_freezing(freeze_list)
    num_spin_orbitals -= len(freeze_list)
    num_particles -= len(freeze_list)
if len(remove_list) > 0:
    ferOp = ferOp.fermion_mode_elimination(remove_list)
    num_spin_orbitals -= len(remove_list)

qubitOp = ferOp.mapping(map_type=map_type, threshold=0.00000001)
qubitOp = qubitOp.two_qubit_reduced_operator(num_particles) if qubit_reduction else qubitOp
qubitOp.chop(10**-10)

# Using exact eigensolver to get the smallest eigenvalue
exact_eigensolver = ExactEigensolver(qubitOp, k=1)
ret = exact_eigensolver.run()
#print('The computed energy is: {:.12f}'.format(ret['eigvals'][0].real))
#print('The total ground state energy is: {:.12f}'.format(ret['eigvals'][0].real + energy_shift + nuclear_repulsion_energy))

backend = Aer.get_backend(backend_string)

# setup COBYLA optimizer
max_eval = 10000
if optimizer_string == 'COBYLA':
    optimizer = COBYLA(maxiter=max_eval)
else:
    print('ERROR: {} is not a recognized classical optimizer!'.format(optimizer_string))

# setup HartreeFock state
HF_state = HartreeFock(qubitOp.num_qubits, num_spin_orbitals, num_particles, map_type, 
                       qubit_reduction)

# setup UCCSD variational form
var_form = UCCSD(qubitOp.num_qubits, depth=1, 
                 num_orbitals=num_spin_orbitals, num_particles=num_particles, 
                 active_occupied=[0], active_unoccupied=[0, 1],
                 initial_state=HF_state, qubit_mapping=map_type, 
                 two_qubit_reduction=qubit_reduction, num_time_slices=1)

# setup VQE
backend_options = {'max_parallel_threads': 0, 'max_parallel_shots': 0}
vqe = VQE(qubitOp, var_form, optimizer, 'matrix') # , circuit_file='circuit.txt')

vqe.convergence_print = True
vqe.convergence_file = 'convergence_' + molecule_formula + '_' + backend_string_short +  '_' + str(atomic_distance) + '.csv'

quantum_instance = QuantumInstance(backend=backend, backend_options=backend_options)

print('Running simulation...')
results = vqe.run(quantum_instance)
#print('The computed ground state energy is: {:.12f}'.format(results['eigvals'][0]))
#print('The total ground state energy is: {:.12f}'.format(results['eigvals'][0] + energy_shift + nuclear_repulsion_energy))
#print("Parameters: {}".format(results['opt_params']))


if data_file:
    with open(data_file,'a') as outfile:
        outfile.write('%f,%f,%f\n' % (atomic_distance, results['eigvals'][0], results['eigvals'][0] + energy_shift + nuclear_repulsion_energy))
else: 
    print('Result: %f,%f,%f' % (atomic_distance, results['eigvals'][0], results['eigvals'][0] + energy_shift + nuclear_repulsion_energy))

print("Simulation completed!")
