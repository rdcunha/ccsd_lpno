'''
Computing the CCSD correlation energy using an RHF reference
References used:
    - http://github.com/CrawfordGroup/ProgrammingProjects
    - Stanton:1991:4334
    - https://github.com/psi4/psi4numpy
'''

import numpy as np
import psi4

import time

# Set memory
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
np.set_printoptions(precision=8, linewidth=400, suppress=True)
numpy_memory = 2

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

psi4.set_options({'basis': 'STO-3g', 'scf_type': 'pk', 'mp2_type': 'conv',
    'freeze_core': 'false', 'e_convergence': 1e-10, 'd_convergence':1e-10, 'save_jk': 'true'})

# Set for CCSD
E_conv = 1e-6
maxiter = 30
print_amps = False
compare_psi4 = False

# N dimensional dot
# Like a mini DPD library
# Using from helper_CC.py by dgasmith
def ndot(input_string, op1, op2, prefactor=None):
    """
    No checks, if you get weird errors its up to you to debug.

    ndot('abcd,cdef->abef', arr1, arr2)
    """
    inp, output_ind = input_string.split('->')
    input_left, input_right = inp.split(',')

    size_dict = {}
    for s, size in zip(input_left, op1.shape):
        size_dict[s] = size
    for s, size in zip(input_right, op2.shape):
        size_dict[s] = size

    set_left = set(input_left)
    set_right = set(input_right)
    set_out = set(output_ind)

    idx_removed = (set_left | set_right) - set_out
    keep_left = set_left - idx_removed
    keep_right = set_right - idx_removed

    # Tensordot axes
    left_pos, right_pos = (), ()
    for s in idx_removed:
        left_pos += (input_left.find(s), )
        right_pos += (input_right.find(s), )
    tdot_axes = (left_pos, right_pos)

    # Get result ordering
    tdot_result = input_left + input_right
    for s in idx_removed:
        tdot_result = tdot_result.replace(s, '')

    rs = len(idx_removed)
    dim_left, dim_right, dim_removed = 1, 1, 1
    for key, size in size_dict.items():
        if key in keep_left:
            dim_left *= size
        if key in keep_right:
            dim_right *= size
        if key in idx_removed:
            dim_removed *= size

    shape_result = tuple(size_dict[x] for x in tdot_result)
    used_einsum = False

    # Matrix multiply
    # No transpose needed
    if input_left[-rs:] == input_right[:rs]:
        new_view = np.dot(op1.reshape(dim_left, dim_removed), op2.reshape(dim_removed, dim_right))

    # Transpose both
    elif input_left[:rs] == input_right[-rs:]:
        new_view = np.dot(op1.reshape(dim_removed, dim_left).T, op2.reshape(dim_right, dim_removed).T)

    # Transpose right
    elif input_left[-rs:] == input_right[-rs:]:
        new_view = np.dot(op1.reshape(dim_left, dim_removed), op2.reshape(dim_right, dim_removed).T)

    # Tranpose left
    elif input_left[:rs] == input_right[:rs]:
        new_view = np.dot(op1.reshape(dim_removed, dim_left).T, op2.reshape(dim_removed, dim_right))

    # If we have to transpose vector-matrix, einsum is faster
    elif (len(keep_left) == 0) or (len(keep_right) == 0):
        new_view = np.einsum(input_string, op1, op2)
        used_einsum = True

    else:
        new_view = np.tensordot(op1, op2, axes=tdot_axes)

    # Make sure the resulting shape is correct
    if (new_view.shape != shape_result) and not used_einsum:
        if (len(shape_result) > 0):
            new_view = new_view.reshape(shape_result)
        else:
            new_view = np.squeeze(new_view)

    # In-place mult by prefactor if requested
    if prefactor is not None:
        new_view *= prefactor

    # Do final tranpose if needed
    if used_einsum:
        return new_view
    elif tdot_result == output_ind:
        return new_view
    else:
        return np.einsum(tdot_result + '->' + output_ind, new_view)

# Compute RHF energy with psi4
e_scf, wfn = psi4.energy('SCF', return_wfn=True)

print('SCF energy: {}\n'.format(e_scf))

# Get no_occ, no_mo, e_scf(?), eps
C = wfn.Ca()
no_occ = wfn.doccpi()[0]
no_mo = wfn.nmo()
eps = np.asarray(wfn.epsilon_a())
J = wfn.jk().J()[0].to_array()
K = wfn.jk().K()[0].to_array()

mints = psi4.core.MintsHelper(wfn.basisset())

H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())

F = H + 2*J - K

# Make spin-orbital MO integrals
MO = np.asarray(mints.mo_spin_eri(C, C))

# change no. of MOs, no_occ, no_vir
no_mo = no_mo * 2
no_occ = no_occ * 2
no_vir = no_mo - no_occ
eps = np.repeat(eps, 2)

eps_occ = eps[:no_occ]
eps_vir = eps[no_occ:]
# note that occ.transpose(col) - vir(row) gives occ x vir matrix of differences
# needs F_occ and F_vir separate (will need F_vir for semi-canonical basis later)

# Make F MO basis
F = np.einsum('uj, vi, uv', C, C, F)
F = np.repeat(F, 2, axis=0)
F = np.repeat(F, 2, axis=1)

# Make F block diagonal
spin_ind = np.arange(F.shape[0], dtype=np.int) % 2
F *= (spin_ind.reshape(-1, 1) == spin_ind)

F_occ = F[:no_occ, :no_occ]
F_vir = F[no_occ:, no_occ:]

# init T1s
t_ia = np.zeros((no_occ, no_vir))

# init T2s
d_ia = eps_occ.reshape(-1, 1) - eps_vir
d_ijab =  eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1, 1) - eps_vir.reshape(-1,1) - eps_vir
t_ijab = MO[:no_occ, :no_occ, no_occ:, no_occ:] / d_ijab 

#print('D_ia\'s : \n {} \n'.format(d_ia))
#print('Initial T2s: \n {}\n'.format(t_ijab))

# Initialize PNOs
    # Identify weak pairs using MP2 pair corr energy

    # Create Tij and Ttij

    # Form pair densities

    # Diagonalize pair densities to get PNOs (Q) and occ_nos

    # Get semicanonical transforms
        # transform F_vir to PNO basis
        # Diagonalize F_pno, get L
        # save virtual orb. energies

# Update Ts
    # Transform Rs using Q
    # Transform RQs using L
    # Use vir orb. energies from semicanonical
    # Back transform to RQs
    # Back transform to Rs

# Make intermediates, Staunton:1991 eqns 3-11
def make_taut(t_ia, t_ijab):
    tau_t = t_ijab + 0.5* (np.einsum('ia,jb->ijab',t_ia,t_ia) - np.einsum('ib,ja->ijab',t_ia, t_ia))
    return tau_t

def make_tau(t_ia, t_ijab):
    tau = t_ijab + (np.einsum('ia,jb->ijab',t_ia,t_ia) - np.einsum('ib,ja->ijab',t_ia, t_ia)) 
    return tau

def make_Fae(taut, t_ia, t_ijab):
    Fae = F_vir.copy()
    Fae[np.diag_indices_from(Fae)] = 0
    Fae -= ndot('me,ma->ae', F[:no_occ, no_occ:], t_ia, prefactor=0.5)
    Fae += ndot('mf,mafe->ae',t_ia,MO[:no_occ, no_occ:, no_occ:, no_occ:])
    Fae -= ndot('mnaf,mnef->ae', taut,MO[:no_occ, :no_occ, no_occ:, no_occ:], prefactor=0.5)
    return Fae

def make_Fmi(taut, t_ia, t_ijab):
    Fmi = F_occ.copy()
    Fmi[np.diag_indices_from(Fmi)] = 0
    Fmi += ndot('ie,me->mi', t_ia, F[:no_occ, no_occ:], prefactor=0.5)
    Fmi += ndot('ne,mnie->mi',t_ia,MO[:no_occ, :no_occ, :no_occ, no_occ:])
    Fmi += ndot('inef,mnef->mi', taut, MO[:no_occ, :no_occ, no_occ:, no_occ:], prefactor=0.5)
    return Fmi

def make_Fme(t_ia, t_ijab):
    Fme = F[:no_occ, no_occ:].copy() 
    Fme += ndot('nf,mnef->me',t_ia, MO[:no_occ, :no_occ, no_occ:, no_occ:])
    return Fme

def make_Wmnij(tau, t_ia, t_ijab):
    Wmnij = MO[:no_occ, :no_occ, :no_occ, :no_occ].copy() 
    Wmnij += ndot('je,mnie->mnij', t_ia, MO[:no_occ, :no_occ, :no_occ, no_occ:]) 
    Wmnij -= ndot('ie,mnje->mnij', t_ia, MO[:no_occ, :no_occ, :no_occ, no_occ:])
    Wmnij += ndot('ijef,mnef->mnij', tau, MO[:no_occ, :no_occ, no_occ:, no_occ:], prefactor=0.25)
    return Wmnij

def make_Wabef(tau, t_ia, t_ijab):
    Wabef = MO[no_occ:, no_occ:, no_occ:, no_occ:].copy()
    Wabef -= ndot('mb,amef->abef', t_ia, MO[no_occ:, :no_occ, no_occ:, no_occ:]) 
    Wabef += ndot('ma,bmef->abef', t_ia, MO[no_occ:, :no_occ, no_occ:, no_occ:])
    Wabef += ndot('mnab,mnef->abef', tau, MO[:no_occ, :no_occ, no_occ:, no_occ:], prefactor=0.25)
    return Wabef

def make_Wmbej(t_ia, t_ijab):
    Wmbej = MO[:no_occ, no_occ:, no_occ:, :no_occ].copy() 
    Wmbej += ndot('jf,mbef->mbej', t_ia, MO[:no_occ, no_occ:, no_occ:, no_occ:]) 
    Wmbej -= ndot('nb,mnej->mbej', t_ia, MO[:no_occ, :no_occ, no_occ:, :no_occ])
    Wmbej -= ndot('jnfb,mnef->mbej', t_ijab, MO[:no_occ, :no_occ, no_occ:, no_occ:], prefactor=0.5)
    Wmbej -= np.einsum('jf,nb,mnef->mbej', t_ia, t_ia, MO[:no_occ, :no_occ, no_occ:, no_occ:])
    return Wmbej

# Update T1 and T2 amplitudes
def update_ts(tau, tau_t, t_ia, t_ijab):

    # Build intermediates
    Fae = make_Fae(tau_t, t_ia, t_ijab)
    Fmi = make_Fmi(tau_t, t_ia, t_ijab)
    Fme = make_Fme(t_ia, t_ijab)

    Wmnij = make_Wmnij(tau, t_ia, t_ijab)
    Wabef = make_Wabef(tau, t_ia, t_ijab)
    Wmbej = make_Wmbej(t_ia, t_ijab)

    # Create residual T1s
    Ria = F[:no_occ, no_occ:].copy()
    Ria += ndot('ie,ae->ia',t_ia, Fae)
    Ria -= ndot('ma,mi->ia',t_ia, Fmi)
    Ria += ndot('imae,me->ia', t_ijab, Fme) 
    Ria -= ndot('nf,naif->ia',t_ia, MO[:no_occ, no_occ:, :no_occ, no_occ:])
    Ria -= ndot('imef,maef->ia', t_ijab, MO[:no_occ, no_occ:, no_occ:, no_occ:], prefactor=0.5)
    Ria -= ndot('mnae,nmei->ia', t_ijab, MO[:no_occ, :no_occ, no_occ:, :no_occ], prefactor=0.5)

    # Create residual T2s
    Rijab = MO[:no_occ, :no_occ, no_occ:, no_occ:].copy()
    # Term 2
    Rijab += ndot('ijae,be->ijab', t_ijab, Fae) 
    Rijab -= 0.5 * np.einsum('ijae,mb,me->ijab', t_ijab, t_ia, Fme)
    # Term 2 permuted
    Rijab -= ndot('ijbe,ae->ijab', t_ijab, Fae)
    Rijab += 0.5 * np.einsum('ijbe,ma,me->ijab', t_ijab, t_ia, Fme)
    # Term 3
    Rijab -= ndot('imab,mj->ijab', t_ijab, Fmi) 
    Rijab -= 0.5 * np.einsum('imab,je,me->ijab', t_ijab, t_ia, Fme)
    # Term 3 permuted
    Rijab += ndot('jmab,mi->ijab', t_ijab, Fmi) 
    Rijab += 0.5 * np.einsum('jmab,ie,me->ijab', t_ijab, t_ia, Fme)
    # Term 4
    Rijab += ndot('mnab,mnij->ijab', tau, Wmnij, prefactor=0.5)
    # Term 5
    Rijab += ndot('ijef,abef->ijab', tau, Wabef, prefactor=0.5)
    # Term 6 # 1
    Rijab += ndot('imae,mbej->ijab', t_ijab, Wmbej)
    Rijab -= np.einsum('ie,ma,mbej->ijab', t_ia, t_ia, MO[:no_occ, no_occ:, no_occ:, :no_occ])
    # Term 6 ij permuted
    Rijab -= ndot('jmae,mbei->ijab', t_ijab, Wmbej)
    Rijab += np.einsum('je,ma,mbei->ijab', t_ia, t_ia, MO[:no_occ, no_occ:, no_occ:, :no_occ])
    # Term 6 ab permuted
    Rijab -= ndot('imbe,maej->ijab', t_ijab, Wmbej)
    Rijab += np.einsum('ie,mb,maej->ijab', t_ia, t_ia, MO[:no_occ, no_occ:, no_occ:, :no_occ])
    # Term 6 ij and ab permuted 
    Rijab += ndot('jmbe,maei->ijab', t_ijab, Wmbej)
    Rijab -= np.einsum('je,mb,maei->ijab', t_ia, t_ia, MO[:no_occ, no_occ:, no_occ:, :no_occ])
    # Term 7
    Rijab += ndot('ie,abej->ijab', t_ia, MO[no_occ:, no_occ:, no_occ:, :no_occ])
    # Term 7 ij permuted
    Rijab -= ndot('je,abei->ijab', t_ia, MO[no_occ:, no_occ:, no_occ:, :no_occ])
    # Term 8
    Rijab -= ndot('ma,mbij->ijab', t_ia, MO[:no_occ, no_occ:, :no_occ, :no_occ])
    # Term 8 ab permuted
    Rijab += ndot('mb,maij->ijab', t_ia, MO[:no_occ, no_occ:, :no_occ, :no_occ])

    # Apply denominators
    new_tia = Ria/d_ia
    new_tijab = Rijab/d_ijab 

    return new_tia, new_tijab

# Compute CCSD correlation energy
def corr_energy(t_ia, t_ijab):
    E_corr = ndot('ia,ia->',F[:no_occ, no_occ:], t_ia)
    E_corr += ndot('ijab,ijab->', MO[:no_occ, :no_occ, no_occ:, no_occ:], t_ijab, prefactor=0.25)
    E_corr += 0.5 * np.einsum('ijab,ia,jb->', MO[:no_occ, :no_occ, no_occ:, no_occ:], t_ia, t_ia)
    return E_corr

old_e = corr_energy(t_ia, t_ijab)
print('Iteration\t\t CCSD Correlation energy\n0\t\t {}'.format(old_e)) 

# Iterate until convergence
for i in range(maxiter):
    tau_t = make_taut(t_ia, t_ijab)
    tau = make_tau(t_ia, t_ijab)
    new_tia, new_tijab = update_ts(tau, tau_t, t_ia, t_ijab)
    new_e = corr_energy(new_tia, new_tijab)
    print('{}\t\t\t {}\n'.format(i, new_e))
    if(abs(new_e - old_e) < E_conv):
        print('Convergence reached.\n CCSD Correlation energy: {}\n'.format(new_e))
        break
    t_ia = new_tia
    t_ijab = new_tijab
    old_e = new_e
