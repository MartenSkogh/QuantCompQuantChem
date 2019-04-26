#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# import common packages
import sys
import os
import time
import datetime
import numpy as np
from multiprocessing import Pool

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

def calculate_ground_state_energy(backend_string='qasm_simulator', optimizer_string='COBYLA', atom_string=None, callback=None, vqe_callback=None):
    if not atom_string:
        raise ValueError('No atomic string given!')
    else:
        atoms = atom_string.split(';')
        coords = []
        for atom in atoms:
            coords.append([float(c) for c in atom.split()[1:]])
        atomic_distance = np.sqrt(sum((np.array(coords[0])-np.array(coords[1]))**2))

    print()
    print('==== Simulation setup ====')
    print(f'    Atomic Setup: {atom_string}')
    print(f'    Distance: {atomic_distance}')
    print(f'    Backend: {backend_string}')
    print(f'    Classical Optimizer: {optimizer_string}')
    print()

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
    print("HF energy: {}".format(molecule.hf_energy))
    print("HF energy + nuclear repulsion: {}".format(molecule.hf_energy + molecule.nuclear_repulsion_energy))
    print("HF energy - nuclear repulsion: {}".format(molecule.hf_energy - molecule.nuclear_repulsion_energy))
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
    max_eval = 1000
    if optimizer_string == 'COBYLA':
        optimizer = COBYLA(maxiter=max_eval)
    else:
        print('ERROR: {} is not a recognized classical optimizer!'.format(optimizer_string))

    # setup HartreeFock state
    HF_state = HartreeFock(qubitOp.num_qubits, 
                           num_spin_orbitals, 
                           num_particles, 
                           map_type, 
                           qubit_reduction)

    # setup UCCSD variational form
    var_form = UCCSD(qubitOp.num_qubits, 
                     depth=1, 
                     num_orbitals=num_spin_orbitals, 
                     num_particles=num_particles, 
                     #active_occupied=[0], 
                     #active_unoccupied=[0, 1],
                     initial_state=HF_state, 
                     qubit_mapping=map_type, 
                     two_qubit_reduction=qubit_reduction, 
                     #num_time_slices=1
                     )



    # setup VQE
    backend_options = {'max_parallel_threads': 0, 'max_parallel_shots': 0}
    vqe = VQE(qubitOp, var_form, optimizer, 'matrix', callback=vqe_callback) # , circuit_file='circuit.txt')


    quantum_instance = QuantumInstance(backend=backend, backend_options=backend_options)

    print('Running simulation...', flush=True)
    results = vqe.run(quantum_instance)
    #print('The computed ground state energy is: {:.12f}'.format(results['eigvals'][0]))
    #print('The total ground state energy is: {:.12f}'.format(results['eigvals'][0] + energy_shift + nuclear_repulsion_energy))
    #print("Parameters: {}".format(results['opt_params']))

    callback(molecule.hf_energy, results['eigvals'][0], results['eigvals'][0] + energy_shift + nuclear_repulsion_energy)

    return [molecule.hf_energy, results['eigvals'][0], results['eigvals'][0] + energy_shift + nuclear_repulsion_energy]

def write_to_file(data_file, data):
    with open(data_file,'a') as outfile:
        for row in data:
            data_string = ','.join([str(d) for d in row]) + '\n'
            outfile.write(data_string)



if __name__ == '__main__':
    molecule = "LiH"
    distances = np.linspace(0.5, 2.0, 30)
    angles = np.linspace(0, 2*np.pi, 1)
    backend="statevector_simulator"
    optimizer="COBYLA"

    result_filename = f"Results/{molecule}_{backend}_{optimizer}_distance_angle_{datetime.date.today()}.csv"
    result_file = open(result_filename, 'a')
    print(f"Writing to '{result_filename}'")

    # Pre-calculate the number of iterations we plan to do
    tot_nbr_runs = len(distances) * len(angles)

    start_time = time.perf_counter()

    # Allocate larger arrays
    results = [ [0.0, 0.0, 0.0, 0.0] for i in range(tot_nbr_runs) ]
    iteration_times = [ 0.0 for i in range(tot_nbr_runs) ]

    for i, dist in enumerate(distances):
        for j, ang in enumerate(angles):
            curr_iteration_start = time.perf_counter()
            curr_nbr_runs = i*len(angles) + j + 1

            print(f"Run {curr_nbr_runs} of {tot_nbr_runs}")
            print(f"Distance: {dist}\nAngle: {ang}")

            x = np.cos(ang)*dist
            y = np.sin(ang)*dist


            # Create callback function for VQE
            convergence_file = ('Results/' 
                                + 'convergence_' 
                                + molecule + '_' 
                                + backend +  '_' 
                                + str(x) + '_' 
                                + str(y) + '_' 
                                + optimizer
                                + '.csv')


            # Create a callback function for the energy function
            def result_callback(hf_energy, binding_energy, total_energy):
                result_file.write(f"{dist},{ang},{hf_energy},{binding_energy},{total_energy}\n")

            conv_file = open(convergence_file, 'w')

            def vqe_callback(eval_count, parameter_sets, mean, std):
                print(eval_count, mean, std)
                conv_file.write(f'{eval_count},{mean},{std}\n')
                if eval_count >= 1000:
                    conv_file.write('#Did not converge')

            atom_string = f"Li .0 .0 .0; H {x} {y} .0"
            result = calculate_ground_state_energy(backend_string=backend, 
                                                   optimizer_string=optimizer, 
                                                   atom_string=atom_string,
                                                   callback=result_callback,
                                                   vqe_callback=vqe_callback)
            

            conv_file.close()

            results[curr_nbr_runs - 1] = [dist, ang] + result

            curr_time = time.perf_counter()
            time_since_start = curr_time - start_time
            iteration_time = curr_time - curr_iteration_start
            avg_run_time = time_since_start/curr_nbr_runs
            exp_finish_time = avg_run_time*(tot_nbr_runs - curr_nbr_runs)

            exp_finish_string = datetime.timedelta(seconds=exp_finish_time)
            iteration_times[curr_nbr_runs - 1] = iteration_time

            print(f"Time since start: {datetime.timedelta(seconds=time_since_start)}")
            print(f"Latest iteration time: {datetime.timedelta(seconds=iteration_time)}")
            print(f"Average run time (for {curr_nbr_runs} runs): {datetime.timedelta(seconds=avg_run_time)}")
            print(f"Expeceted to finish in {exp_finish_string}")


    print(f"Total number of runs: {tot_nbr_runs}")
    #write_to_file(output_file, results)
    result_file.close()
