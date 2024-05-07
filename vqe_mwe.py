"""
Contains the main file for running a complete VQE of the FeNTA system
"""
import numpy as np
import pickle
import json
import time
from src.vqe_cudaq_qnp import VqeQnp
from src.utils_cudaq import get_cudaq_hamiltonian


if __name__ == "__main__":

    mpi_support = False
    target = "nvidia-mqpu"

    num_active_orbitals = 5
    num_active_electrons = 5
    n_vqe_layers = 1
    spin = 1
    hamiltonian_fname = "FeNTA_s_1_cc-pvtz_5e_5o/ham_FeNTA_cc-pvtz_5e_5o.pickle"
    optimizer_type = "cudaq"

    print("# Mpi_support", mpi_support)
    print("# Target", target)

    with open(hamiltonian_fname, 'rb') as filehandler:
        jw_hamiltonian = pickle.load(filehandler)

    start = time.time()
    hamiltonian_cudaq, energy_core = get_cudaq_hamiltonian(jw_hamiltonian)
    end = time.time()
    print("# Time for preparing the cudaq hamiltonian:", end - start)

    n_qubits = 2 * num_active_orbitals

    empty_orbitals = num_active_orbitals - ((num_active_electrons // 2) + (num_active_electrons % 2))

    n_alpha = int((num_active_electrons + spin) / 2)
    n_beta = int((num_active_electrons - spin) / 2)

    n_alpha_vec = np.array([1] * n_alpha + [0] * (num_active_orbitals - n_alpha))
    n_beta_vec = np.array([1] * n_beta + [0] * (num_active_orbitals - n_beta))
    init_mo_occ = (n_alpha_vec + n_beta_vec).tolist()

    options = {'maxiter': 50000,
               'optimizer_type': optimizer_type,
               'energy_core': energy_core,
               'mpi_support': mpi_support,
               'initial_parameters': None}

    print("# init_mo_occ", init_mo_occ)
    print("# layers", n_vqe_layers)
    time_start = time.time()
    vqe = VqeQnp(n_qubits=n_qubits,
                 n_layers=n_vqe_layers,
                 init_mo_occ=init_mo_occ,
                 target=target)

    energy_optimized, best_parameters, callback_energies = vqe.run_vqe_cudaq(hamiltonian_cudaq,
                                                                             options=options)
    time_end = time.time()
    print(f"# Best energy {energy_optimized}")
    print(f"# VQE time {time_end - time_start}")
