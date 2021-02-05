'''
Computing the CCSD-LR dipole polarizability  using an RHF reference
References used:
    - http://github.com/CrawfordGroup/ProgrammingProjects
    - Stanton:1991:4334
    - https://github.com/psi4/psi4numpy
'''

import numpy as np
import psi4
import ccsd_lpno

cutoffs = [1e-10, 1e-5]
#cutoffs =[0]

optrot_lg_list = []
optrot_mvg_list = []

optrot_compare_list_mvg = [1131.5129117543029, 437.9717697996912]
optrot_compare_list_lg = [1323.4690320008708, 1010.6043948493821]
polar_compare_list = [19.3061590291987, 18.855612676086874]

psi4.core.clean()

# Set memory
psi4.set_memory('2 GB')
psi4.core.set_output_file('optrot.dat', False)
np.set_printoptions(precision=12, threshold=np.inf, linewidth=200, suppress=True)

# Set Psi4 options
geom = ccsd_lpno.mollib.mollib["h2_4"]
mol = psi4.geometry(geom)

psi4.set_options({'basis': 'aug-cc-pvdz', 'scf_type': 'pk',
                  'freeze_core': 'false', 'e_convergence': 1e-12,
                  'd_convergence': 1e-12, 'save_jk': 'true'})

# Set for CCSD
E_conv = 1e-10
R_conv = 1e-8
maxiter = 40
compare_psi4 = False

# Set for linear response calculation
omega_nm = 589

# Compute RHF energy with psi4
psi4.set_module_options('SCF', {'E_CONVERGENCE': 1e-12})
psi4.set_module_options('SCF', {'D_CONVERGENCE': 1e-12})
e_scf, wfn = psi4.energy('SCF', return_wfn=True)
print('SCF energy: {}\n'.format(e_scf))
print('Nuclear repulsion energy: {}\n'.format(mol.nuclear_repulsion_energy()))

no_vir = wfn.nmo() - wfn.doccpi()[0] - wfn.frzcpi()[0]

# Set for LPNO
localize=True
#local=False
pert='mu'

def test_or_lg():
    i = 0
    for cut in cutoffs:
        pno_cut = cut

        # Do the linear response calculation
        optrot_lg = ccsd_lpno.do_linresp(wfn, omega_nm, mol, method='optrot', gauge='length', localize=localize, pert=pert, pno_cut=pno_cut) 
        print("Optical rotation(LG) = {}".format(optrot_lg))
        assert np.allclose(optrot_lg, optrot_compare_list_lg[i], atol=1e-4)
        i += 1
        
def test_or_mvg():
    i = 0
    for cut in cutoffs:
        pno_cut = cut

        # Do the linear response calculation
        optrot_mvg = ccsd_lpno.do_linresp(wfn, omega_nm, mol, method='optrot', gauge='velocity', localize=localize, pert=pert, pno_cut=pno_cut) 
        print("Optical rotation(MVG) = {}".format(optrot_mvg))
        assert np.allclose(optrot_mvg, optrot_compare_list_mvg[i], atol=1e-4)
        i += 1

def test_polar():
    i = 0
    for cut in cutoffs:
        pno_cut = cut

        # Do the linear response calculation
        polar = ccsd_lpno.do_linresp(wfn, omega_nm, mol, method='polar', localize=localize, pert=pert, pno_cut=pno_cut)
        assert np.allclose(polar, polar_compare_list[i], atol=1e-4)
        print("Polarizability = {}".format(polar))
        i += 1
