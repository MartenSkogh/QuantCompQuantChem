#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# import common packages
import sys
import os
import time
import datetime
import numpy as np
from pprint import pprint
import itertools
from random import randint

# Fromt the Qiskit base package
from qiskit import Aer
from qiskit import QuantumRegister, QuantumCircuit

# lib from Qiskit Aqua
from qiskit.aqua import Operator, QuantumInstance
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.components.variational_forms import RY, RYRZ, SwapRZ
from qiskit.aqua.aqua_error import AquaError

# lib from Qiskit Aqua Chemistry
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock

class Settings():
    def __init__(self,
                 settings_array=None,
                 molecule=None,
                 basis_set=None,
                 freeze_list=None,
                 remove_list=None,
                 backend=None,
                 mapping=None,
                 variational_method=None,
                 variational_depth=None,
                 optimizer=None,
                 operator_mode=None):

        if settings_array == None:
            self.settings_array = [molecule,
                                   basis_set,
                                   freeze_list,
                                   remove_list,
                                   backend,
                                   mapping,
                                   variational_method,
                                   variational_depth,
                                   optimizer,
                                   operator_mode]

        if settings_array != None:
            self.molecule = settings_array[0]
            self.basis_set = settings_array[1]
            self.freeze_list = settings_array[2]
            self.remove_list = settings_array[3]
            self.backend = settings_array[4]
            self.mapping = settings_array[5]
            self.variational_method = settings_array[6]
            self.variational_depth = settings_array[7]
            self.optimizer = settings_array[8]
            self.operator_mode=settings_array[9]

        self.atom_string = None
        self.only_generate_circuit = False


        self.circuit_filename = (f"Circuits/"
                                 f"{molecule}_"
                                 f"{mapping}_"
                                 f"{variational_method}="
                                 f"{variational_depth}_"
                                 f"{basis_set}.csv")

        # Create callback function for VQE
        self.convergence_file = (f"Results/"
                                 f"convergence_"
                                 f"{molecule}_"
                                 f"{backend}_"
                                 f"{x}_"
                                 f"{y}_"
                                 f"{optimizer}"
                                 f".csv")



    def vqe_callback(self, eval_count, parameter_sets, mean, std):
        with open(convergence_file) as conv_file:
            print(eval_count, mean, std)
            conv_file.write(f'{eval_count},{mean},{std}\n')
            if eval_count >= 1000:
                conv_file.write('#Did not converge')


    def add_row_circuit_table(self,
                              number_qubits=None,
                              cicuit_depth=None,
                              number_gates=None):
        
        with open('Circuits/circuit_table.csv','a') as table_file:
            row = [str(self),
                   number_qubits,
                   number_gates,
                   cicuit_depth]

            row_string = ','.join([str(element) for element in row])

            table_file.write(row_string + '\n')

    def circuit_callback(self, quantum_circuit):
        '''Save the circuit so we can see the size that this specific configuration generates.'''
        with open(self.circuit_filename, 'w') as circuit_file:
            for circuit in quantum_circuit:
                number_qubits = len(circuit.qubits)
                depth = circuit.depth()
                size = circuit.size()
                qasm_rep = circuit.qasm()
                #Unsure how to make comments in QASM, my guess is '#' but it could be wrong...
                circuit_file.write(f'#Number of qubits: {number_qubits}\n')
                circuit_file.write(f'#Number of gates: {size}\n')
                circuit_file.write(f'#Depth: {depth}\n')
                circuit_file.write(qasm_rep + '\n')

                self.add_row_circuit_table(number_qubits=number_qubits,
                                           number_gates=size,
                                           cicuit_depth=depth)
        
    def __str__(self):
        row = [self.molecule, 
               self.basis_set, 
               '"[' + ' '.join([ str(i) for i in self.freeze_list]) + ']"',
               '"[' + ' '.join([ str(i) for i in self.remove_list]) + ']"',
               self.backend, 
               self.mapping, 
               self.variational_method, 
               self.variational_depth, 
               self.optimizer,
               self.operator_mode]

        row_string = ','.join([str(element) for element in row])
        return row_string

def calculate_ground_state_energy(settings):

    backend_string = settings.backend
    basis_set = settings.basis_set
    optimizer_string = settings.optimizer
    atom_string = settings.atom_string
    freeze_list = settings.freeze_list 
    remove_list = settings.remove_list
    mapping = settings.mapping
    variational_method = settings.variational_method
    variational_depth = settings.variational_depth
    vqe_callback = settings.vqe_callback
    circuit_callback = settings.circuit_callback
    generate_circuit = settings.generate_circuit
    run_vqe = settings.run_vqe
    operator_mode = settings.operator_mode

    if not atom_string:
        raise ValueError('No atomic string given!')
    

    print()
    print('==== Simulation setup ====')
    print(f'    Atomic Setup: {atom_string}')
    print(f'    Backend: {backend_string}')
    print(f'    Basis Set: {basis_set}')
    print(f'    Ferminic Mapping: {mapping}')
    print(f'    Variational Method: {variational_method}')
    print(f'    Variational Depth: {variational_depth}')
    print(f'    Classical Optimizer: {optimizer_string}')
    print()

    backend_string_short = backend_string.split('_')[0]

    # using driver to get fermionic Hamiltonian
    # PySCF example
    driver = PySCFDriver(atom=atom_string, 
                         unit=UnitsType.ANGSTROM,
                         charge=0, spin=0, basis=basis_set)
    molecule = driver.run()

    molecule_formula = ''.join(molecule.atom_symbol)

    print('Number orbitals for {} is {}'.format(molecule_formula,molecule.num_orbitals))
    # please be aware that the idx here with respective to original idx
    freeze_list = freeze_list #[0]
    remove_list = remove_list #[-3,-2] # negative number denotes the reverse order
    map_type = mapping

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

    if variational_method == 'UCCSD':
        # setup UCCSD variational form
        var_form = UCCSD(qubitOp.num_qubits, 
                         depth=variational_depth, 
                         num_orbitals=num_spin_orbitals, 
                         num_particles=num_particles, 
                         #active_occupied=[0], 
                         #active_unoccupied=[0, 1],
                         initial_state=HF_state, 
                         qubit_mapping=map_type, 
                         two_qubit_reduction=qubit_reduction, 
                         #num_time_slices=1
                         )
    elif variational_method == 'RY':
        var_form = RY(qubitOp.num_qubits,
                      depth=variational_depth,
                      initial_state=HF_state
                      )
    elif variational_method == 'RYRZ':
        var_form = RYRZ(qubitOp.num_qubits,
                        depth=variational_depth,
                        initial_state=HF_state
                        )
    elif variational_method == 'SwapRZ':
        var_form = SwapRZ(qubitOp.num_qubits,
                          depth=variational_depth,
                          initial_state=HF_state
                          )
    else:
        raise ValueError('Incorrect variational method "{variational_method}" entered!')


    # setup VQE
    backend_options = {'max_parallel_threads': 0, 'max_parallel_shots': 0}
    vqe = VQE(qubitOp, var_form, optimizer, operator_mode, callback=vqe_callback) # , circuit_file='circuit.txt')


    quantum_instance = QuantumInstance(backend=backend, backend_options=backend_options)
    
    # Extract the gate sequence, with wrong parameters though
    if generate_circuit:
        quantum_circuit = vqe.construct_circuit(np.zeros(var_form.num_parameters),backend=backend)
        circuit_callback(quantum_circuit)

    if run_vqe:
        print('Running simulation...', flush=True)
        results = vqe.run(quantum_instance)
        #callback(molecule.hf_energy, results['eigvals'][0], results['eigvals'][0] + energy_shift + nuclear_repulsion_energy)
        return [molecule.hf_energy, results['eigvals'][0], results['eigvals'][0] + energy_shift + nuclear_repulsion_energy]
    else:
        return None
    #print('The computed ground state energy is: {:.12f}'.format(results['eigvals'][0]))
    #print('The total ground state energy is: {:.12f}'.format(results['eigvals'][0] + energy_shift + nuclear_repulsion_energy))
    #print("Parameters: {}".format(results['opt_params']))



def write_to_file(data_file, data):
    with open(data_file,'a') as outfile:
        for row in data:
            data_string = ','.join([str(d) for d in row]) + '\n'
            outfile.write(data_string)

def get_atom_str(molecule):
    '''Made this function so I don't have to enter these strings manually every time.'''
    atom_strings = {"H2": "H .0 .0 .0; H {} {} .0",
                    "LiH": "Li .0 .0 .0; H {} {} .0",
                    "BeH2": "Be .0 .0 .0; H {} {} .0; H -1.33376 .0 .0",
                    "H2O": "O .0 .0 .0; H {} {} 0; H -0.97 .0 .0",
                    "HCN": "C .0 .0 .0; N -1.1560 .0 .0; H {} {} .0"}

    return atom_strings[molecule] 


if __name__ == '__main__':
    # Slurm environmetal variables
    result_dir = "./Results"#os.environ['RESULT_DIR']
    #nbr_tasks = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
    #task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    job_id = randint(1,2**16)#int(os.environ['SLURM_ARRAY_JOB_ID'])
    #file_identifier = os.environ['FILE_ID']

    molecules = ["H2", "LiH", "BeH2", "H2O"] #, "HCN"]
    distances = np.linspace(0.5, 2.0, 1)
    angles = np.linspace(0, 2*np.pi, 1)
    backends = ["statevector_simulator"]#, "qasm_simulator", "unitary_simulator"]
    mappings = ["parity", "jordan_wigner", "bravyi_kitaev"]
    optimizers = ["COBYLA"]
    variational_methods = ["UCCSD", "RY", "RYRZ", "SwapRZ"]
    variational_depths = [1,2,3,4,5]
    basis_sets = ["sto-3g", "sto-6g", "3-21g", "cc-pvdz"]
    freeze_lists = [[], [0]]
    remove_lists = [[]]
    operator_modes = ["paulis", "matrix"]

    

    molecules_to_use = molecules[1:]
    backends_to_use = backends
    mappings_to_use = mappings
    optimizers_to_use = optimizers
    variational_methods_to_use = variational_methods[1:2]
    variational_depths_to_use = variational_depths[0:1]
    basis_sets_to_use = basis_sets[2:3]
    freeze_lists_to_use = freeze_lists[1:]
    remove_lists_to_use = remove_lists
    operator_modes_to_use = ['matrix'] #operator_modes

    all_individual_settings = [molecules_to_use, 
                               basis_sets_to_use, 
                               freeze_lists_to_use, 
                               remove_lists_to_use, 
                               backends_to_use,
                               mappings_to_use,
                               variational_methods_to_use,
                               variational_depths_to_use,
                               optimizers_to_use,
                               operator_modes_to_use] 

    all_possible_settings = [setting for setting in itertools.product(*all_individual_settings)]

    # Pre-calculate the number of iterations we plan to do
    tot_nbr_runs = len(distances) * len(angles) * len(all_possible_settings)

    start_time = time.perf_counter()

    # Allocate larger arrays
    results = [ [0.0, 0.0, 0.0, 0.0] for i in range(tot_nbr_runs) ]
    iteration_times = [ 0.0 for i in range(tot_nbr_runs) ]


    for i, dist in enumerate(distances):
        for j, ang in enumerate(angles):
            curr_iteration_start = time.perf_counter()

            print(f"Distance: {dist}\nAngle: {ang}")

            x = np.cos(ang)*dist
            y = np.sin(ang)*dist


            print(f"Number of different settings: {len(all_possible_settings)}")
            for n, settings_array in enumerate(all_possible_settings):
                curr_nbr_runs = (i*len(angles)*len(all_possible_settings) 
                                 + j*len(all_possible_settings) 
                                 + 1)
                print(f"Run {curr_nbr_runs} of {tot_nbr_runs}")
                print(f"Setting {n+1} of {len(all_possible_settings)}")
                pprint(settings_array)

                settings =  Settings(settings_array)
                settings.atom_string = get_atom_str(settings.molecule).format(x,y)
                settings.generate_circuit = True
                settings.run_vqe = False
                settings.vqe_callback = None

                try:
                    result = calculate_ground_state_energy(settings)
                except (AquaError, AttributeError) as e:
                    print(e)
                    result = None

                if settings.run_vqe and result != None:
                    
                    experiment_type = "Variational_Layers"
                    experiment_dir = (f"/{experiment_type}"
                                      f"/{settings.molecule}"
                                      f"/{settings.variational_method}"
                                      f"/{settings.variational_depth}")

                    result_filename = (f"{result_dir}"
                                       f"{experiment_dir}/"
                                       f"{job_id}.csv")

                    result_file = open(result_filename, 'a')
                    print(f"Writing to '{result_filename}'")
                    
                    results[curr_nbr_runs - 1] = [dist, ang] + result
                    with open(result_filename, 'a') as result_file:
                        result_string = ','.join([str(result) for result in results[curr_nbr_runs-1]])
                        result_file.write(result_string + '\n')
                
                time.sleep(1)

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
            print(f"Expected to finish in {exp_finish_string}")


    print(f"Total number of runs: {tot_nbr_runs}")
    #write_to_file(output_file, results)
