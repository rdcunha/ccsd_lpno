'''
Computing the CCSD-LR dipole polarizability  using an RHF reference

References used:
    - http://github.com/CrawfordGroup/ProgrammingProjects
    - Stanton:1991:4334
    - https://github.com/psi4/psi4numpy
'''

import numpy as np
import psi4
from .helper_cc import *
from .cc_hbar import *
from .cc_lambda import *
from .cc_pert import *
from .local import *
from psi4 import constants as pc 

# Bring in wfn from psi4
def do_linresp(wfn, omega_nm, mol, return_en=False, method='polar', gauge='length', e_conv=1e-10, r_conv=1e-10, localize=False, pert=None, pno_cut=0, e_cut=0): 
    
    # Create Helper_local object
    if localize:
        no_vir = wfn.nmo() - wfn.doccpi()[0] - wfn.frzcpi()[0]
        local = HelperLocal(wfn.doccpi()[0], no_vir)
    else:
        local=None

    # Set the frequency in hartrees
    if omega_nm == 0:
        omega = 0.0
    else:
        omega = (pc.c * pc.h * 1e9) / (pc.hartree2J * omega_nm)

    # Create Helper_CCenergy object
    hcc = HelperCCEnergy(wfn, local=local, pert=pert, pno_cut=pno_cut, e_cut=e_cut, omega=omega) 
    ccsd_e = hcc.do_CC(local=local, e_conv=e_conv, r_conv=r_conv, maxiter=40, start_diis=0)

    print('CCSD correlation energy: {}'.format(ccsd_e))
    # Create HelperCCHbar object
    hbar = HelperHbar(hcc, ccsd_e)

    # Create HelperLamdba object
    lda = HelperLambda(hcc, hbar)
    pseudo_e = lda.iterate(local=local, e_conv=e_conv, r_conv =r_conv, maxiter=30)

    if method=='polar':
        # Get the perturbation A for Xs and Ys
        dipole_array = hcc.mints.ao_dipole()

        # Create HelperPert object and solve for xs and ys
        Mu = {}
        hpert = {}
        hresp = {}
        polar = {}

        i=0
        for string in ['X', 'Y', 'Z']:
            Mu[string] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(dipole_array[i]))
            hpert[string] = HelperPert(hcc, hbar, lda, Mu[string], omega, local=local)

            i += 1
            for hand in ['right', 'left']:
                pseudoresponse = hpert[string].iterate(hand, r_conv=r_conv, local=local)

        for string in ['X', 'Y', 'Z']:
            for string2 in ['X', 'Y', 'Z']:
                hresp[string+string2] = HelperResp(lda, hpert[string], hpert[string2])
                polar[string+string2] = hresp[string+string2].linear_resp()
            
        print('Polarizability tensor:')
        for string in ['X', 'Y', 'Z']:
            for string2 in ['X', 'Y', 'Z']:
                if string != string2:
                    polar[string+string2+'_new'] = 0.5 * (polar[string+string2] + polar[string2+string])
                else:
                    polar[string+string2+'_new'] = polar[string+string2]
                print("{} {}: {}\n".format(string, string2, polar[string+string2+'_new']))

        trace = polar['XX'] + polar['YY'] + polar['ZZ']
        isotropic_polar = trace / 3.0

        new_x_y = np.reshape(hresp['YY'].x_ijab, (hcc.no_occ*hcc.no_occ, hcc.no_vir, hcc.no_vir))
        
        '''
        # This code block saves amplitudes in order to look at sparsity of the different methods

        if localize:
            NO_x_y = np.zeros_like(new_x_y)
            NO_t = np.zeros_like(new_x_y)
            for ij in range(hcc.no_occ*hcc.no_occ):
                NO_x_y[ij] = np.einsum('Aa,ab,bB->AB', local.Q_list[ij].T, new_x_y[ij], local.Q_list[ij])
                NO_t[ij] = np.einsum('Aa,ab,bB->AB', local.Q_list[ij].T, np.reshape(hcc.t_ijab, (hcc.no_occ*hcc.no_occ, hcc.no_vir, hcc.no_vir))[ij], local.Q_list[ij])
            
            print("Pert: {}".format(pert))
            if pert is None:
                np.save('X2_pno_y', NO_x_y)
                np.save('T2_pno', NO_t)

                np.save('X2_pno_y_untrans', new_x_y)
                np.save('T2_pno_untrans', np.reshape(hcc.t_ijab, (hcc.no_occ*hcc.no_occ, hcc.no_vir, hcc.no_vir)))
            else:
                np.save('X2_pnopp_y', NO_x_y)
                np.save('T2_pnopp', NO_t)

                np.save('X2_pnopp_y_untrans', new_x_y)
                np.save('T2_pnopp_untrans', np.reshape(hcc.t_ijab, (hcc.no_occ*hcc.no_occ, hcc.no_vir, hcc.no_vir)))

        else:
            np.save('X2_can_y', new_x_y)
            np.save('T2_can', np.reshape(hcc.t_ijab, (hcc.no_occ*hcc.no_occ, hcc.no_vir, hcc.no_vir)))
        '''

        if return_en == True:
            return ccsd_e, isotropic_polar
        else:
            return isotropic_polar
    elif method=='optrot':
        if gauge=='both':

            ### Length gauge OR calculation
            ### Form of linear response function: <<mu;L>>

            # Get the perturbation mu
            dipole_array = hcc.mints.ao_dipole()

            # Get the angular momentum L
            angular_momentum = hcc.mints.ao_angular_momentum()

            # Create HelperPert objects for both
            Mu = {}
            pert1 = {}
            L = {}
            pert2 = {}

            # Rosenfeld tensor
            beta = {}
            betap = {}
            beta_new = {}

            i=0
            for string in ['X', 'Y', 'Z']:
                Mu[string] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(dipole_array[i]))
                pert1[string] = HelperPert(hcc, hbar, lda, Mu[string], omega, local=local)
                L[string] = -0.5 * np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(angular_momentum[i]))
                pert2[string] = HelperPert(hcc, hbar, lda, L[string], omega, local=local)

                i+=1
                for hand in ['right', 'left']:
                    pseudoresponse1 = pert1[string].iterate(hand, r_conv=r_conv, local=local)
                for hand in ['right', 'left']:
                    pseudoresponse2 = pert2[string].iterate(hand, r_conv=r_conv, local=local)

            print('Rosenfeld tensor:')
            for string1 in ['X', 'Y', 'Z']:
                for string2 in ['X', 'Y', 'Z']:
                    beta[string1+string2] = HelperResp(lda, pert1[string1], pert2[string2]).linear_resp()
                    betap[string1+string2] = HelperResp(lda, pert2[string2], pert1[string1]).linear_resp()

                    beta_new[string1+string2] = 0.5 * (beta[string1+string2] - betap[string1+string2])
                    print(' {} {} : {}'.format(string1, string2, beta_new[string1+string2]))

            trace = 0.0
            for string in ['XX','YY','ZZ']:
                trace += beta_new[string]
            trace /= 3.0

            # Calculation of the specific rotation
            Mass = 0
            for atom in range(mol.natom()):
                Mass += mol.mass(atom)
            h_bar = pc.h / (2.0 * np.pi)
            prefactor = -72e6 * h_bar**2 * pc.na / (pc.c**2 * pc.me**2 * Mass)
            # Have to multiply with omega for length gauge
            optrot_lg = prefactor * trace * omega

            ### Velocity gauge OR calculation
            ### Form of linear response function: <<p;L>>

            # Get the momentum P
            p_array = hcc.mints.ao_nabla()

            # Get the angular momentum L
            angular_momentum = hcc.mints.ao_angular_momentum()

            # Create HelperPert objects for both
            P = {}
            pert1 = {}
            L = {}
            pert2 = {}

            # Rosenfeld tensor
            beta = {}
            betap = {}
            beta_new = {}

            i=0
            for string in ['X', 'Y', 'Z']:
                P[string] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(p_array[i]))
                pert1[string] = HelperPert(hcc, hbar, lda, P[string], omega, local=local)
                L[string] = -0.5 * np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(angular_momentum[i]))
                pert2[string] = HelperPert(hcc, hbar, lda, L[string], omega, local=local)

                i+=1
                for hand in ['right', 'left']:
                    pseudoresponse1 = pert1[string].iterate(hand, r_conv=r_conv, local=local)
                for hand in ['right', 'left']:
                    pseudoresponse2 = pert2[string].iterate(hand, r_conv=r_conv, local=local)

            print('Rosenfeld tensor:')
            for string1 in ['X', 'Y', 'Z']:
                for string2 in ['X', 'Y', 'Z']:
                    beta[string1+string2] = HelperResp(lda, pert1[string1], pert2[string2]).linear_resp()
                    betap[string1+string2] = HelperResp(lda, pert2[string2], pert1[string1]).linear_resp()

                    beta_new[string1+string2] = 0.5 * (beta[string1+string2] + betap[string1+string2])
                    print(' {} {} : {}'.format(string1, string2, beta_new[string1+string2]))

            trace = 0.0
            for string in ['XX','YY','ZZ']:
                trace += beta_new[string]
            trace /= 3.0

            # Calculation of the specific rotation
            Mass = 0
            for atom in range(mol.natom()):
                Mass += mol.mass(atom)
            h_bar = pc.h / (2.0 * np.pi)
            prefactor = -72e6 * h_bar**2 * pc.na / (pc.c**2 * pc.me**2 * Mass)
            optrot_vg = prefactor * trace
            # So velocity gauge is / omega

            ### Modified velocity gauge OR calculation
            ### Form of linear response function: <<p;L>> - <<p;L>>_0
            ### Using the velocity gauge OR value and subtracting the static value
            omega = 0.0

            # Get the perturbation P
            p_array = hcc.mints.ao_nabla()

            # Get the angular momentum L
            angular_momentum = hcc.mints.ao_angular_momentum()

            # Create HelperPert objects for both
            P = {}
            pert1 = {}
            L = {}
            pert2 = {}

            # Rosenfeld tensor
            beta = {}
            betap = {}
            beta_new = {}

            i=0
            for string in ['X', 'Y', 'Z']:
                P[string] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(p_array[i]))
                pert1[string] = HelperPert(hcc, hbar, lda, P[string], omega, local=local)
                L[string] = -0.5 * np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(angular_momentum[i]))
                pert2[string] = HelperPert(hcc, hbar, lda, L[string], omega, local=local)

                i+=1
                for hand in ['right', 'left']:
                    pseudoresponse1 = pert1[string].iterate(hand, r_conv=r_conv, local=local)
                for hand in ['right', 'left']:
                    pseudoresponse2 = pert2[string].iterate(hand, r_conv=r_conv, local=local)

            print('Rosenfeld tensor:')
            for string1 in ['X', 'Y', 'Z']:
                for string2 in ['X', 'Y', 'Z']:
                    beta[string1+string2] = HelperResp(lda, pert1[string1], pert2[string2]).linear_resp()
                    betap[string1+string2] = HelperResp(lda, pert2[string2], pert1[string1]).linear_resp()

                    beta_new[string1+string2] = 0.5 * (beta[string1+string2] + betap[string1+string2])
                    print(' {} {} : {}'.format(string1, string2, beta_new[string1+string2]))

            trace = 0.0
            for string in ['XX','YY','ZZ']:
                trace += beta_new[string]
            trace /= 3.0

            # Calculation of the specific rotation
            Mass = 0
            for atom in range(mol.natom()):
                Mass += mol.mass(atom)
            h_bar = pc.h / (2.0 * np.pi)
            prefactor = -72e6 * h_bar**2 * pc.na / (pc.c**2 * pc.me**2 * Mass)
            optrot_diff = prefactor * trace

            optrot_mvg = optrot_vg - optrot_diff
            
            if return_en == True:
                return ccsd_e, optrot_lg, optrot_mvg
            else:
                return optrot_lg, optrot_mvg
            
        elif gauge=='length':

            ### Length gauge OR calculation
            ### Form of linear response function: <<mu;L>>

            # Get the perturbation mu
            dipole_array = hcc.mints.ao_dipole()

            # Get the angular momentum L
            angular_momentum = hcc.mints.ao_angular_momentum()

            # Create HelperPert objects for both
            Mu = {}
            pert1 = {}
            L = {}
            pert2 = {}

            # Rosenfeld tensor
            beta = {}
            betap = {}
            beta_new = {}

            i=0
            for string in ['X', 'Y', 'Z']:
                Mu[string] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(dipole_array[i]))
                pert1[string] = HelperPert(hcc, hbar, lda, Mu[string], omega, local=local)
                L[string] = -0.5 * np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(angular_momentum[i]))
                pert2[string] = HelperPert(hcc, hbar, lda, L[string], omega, local=local)

                i+=1
                for hand in ['right', 'left']:
                    pseudoresponse1 = pert1[string].iterate(hand, r_conv=r_conv, local=local)
                for hand in ['right', 'left']:
                    pseudoresponse2 = pert2[string].iterate(hand, r_conv=r_conv, local=local)

            print('Rosenfeld tensor:')
            for string1 in ['X', 'Y', 'Z']:
                for string2 in ['X', 'Y', 'Z']:
                    beta[string1+string2] = HelperResp(lda, pert1[string1], pert2[string2]).linear_resp()
                    betap[string1+string2] = HelperResp(lda, pert2[string2], pert1[string1]).linear_resp()

                    beta_new[string1+string2] = 0.5 * (beta[string1+string2] - betap[string1+string2])
                    print(' {} {} : {}'.format(string1, string2, beta_new[string1+string2]))

            trace = 0.0
            for string in ['XX','YY','ZZ']:
                trace += beta_new[string]
            trace /= 3.0

            # Calculation of the specific rotation
            Mass = 0
            for atom in range(mol.natom()):
                Mass += mol.mass(atom)
            h_bar = pc.h / (2.0 * np.pi)
            prefactor = -72e6 * h_bar**2 * pc.na / (pc.c**2 * pc.me**2 * Mass)
            # Have to multiply with omega for length gauge
            optrot_lg = prefactor * trace * omega
            
            if return_en == True:
                return ccsd_e, optrot_lg
            else:
                return optrot_lg

        elif gauge=='velocity':
            ### Velocity gauge OR calculation
            ### Form of linear response function: <<p;L>>

            # Get the perturbation P
            p_array = hcc.mints.ao_nabla()

            # Get the angular momentum L
            angular_momentum = hcc.mints.ao_angular_momentum()

            # Create HelperPert objects for both
            P = {}
            pert1 = {}
            L = {}
            pert2 = {}

            # Rosenfeld tensor
            beta = {}
            betap = {}
            beta_new = {}

            i=0
            for string in ['X', 'Y', 'Z']:
                P[string] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(p_array[i]))
                pert1[string] = HelperPert(hcc, hbar, lda, P[string], omega, local=local)
                L[string] = -0.5 * np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(angular_momentum[i]))
                pert2[string] = HelperPert(hcc, hbar, lda, L[string], omega, local=local)

                i+=1
                for hand in ['right', 'left']:
                    pseudoresponse1 = pert1[string].iterate(hand, r_conv=r_conv, local=local)
                for hand in ['right', 'left']:
                    pseudoresponse2 = pert2[string].iterate(hand, r_conv=r_conv, local=local)

            print('Rosenfeld tensor:')
            for string1 in ['X', 'Y', 'Z']:
                for string2 in ['X', 'Y', 'Z']:
                    beta[string1+string2] = HelperResp(lda, pert1[string1], pert2[string2]).linear_resp()
                    betap[string1+string2] = HelperResp(lda, pert2[string2], pert1[string1]).linear_resp()

                    beta_new[string1+string2] = 0.5 * (beta[string1+string2] + betap[string1+string2])
                    print(' {} {} : {}'.format(string1, string2, beta_new[string1+string2]))

            trace = 0.0
            for string in ['XX','YY','ZZ']:
                trace += beta_new[string]
            trace /= 3.0

            # Calculation of the specific rotation
            Mass = 0
            for atom in range(mol.natom()):
                Mass += mol.mass(atom)
            h_bar = pc.h / (2.0 * np.pi)
            prefactor = -72e6 * h_bar**2 * pc.na / (pc.c**2 * pc.me**2 * Mass)
            optrot_vg = prefactor * trace
            # So velocity gauge is / omega

            ### Modified velocity gauge OR calculation
            ### Form of linear response function: <<p;L>> - <<p;L>>_0
            ### Using the velocity gauge OR value and subtracting the static value
            omega = 0.0

            # Get the perturbation P
            p_array = hcc.mints.ao_nabla()

            # Get the angular momentum L
            angular_momentum = hcc.mints.ao_angular_momentum()

            # Create HelperPert objects for both
            P = {}
            pert1 = {}
            L = {}
            pert2 = {}

            # Rosenfeld tensor
            beta = {}
            betap = {}
            beta_new = {}

            i=0
            for string in ['X', 'Y', 'Z']:
                P[string] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(p_array[i]))
                pert1[string] = HelperPert(hcc, hbar, lda, P[string], omega, local=local)
                L[string] = -0.5 * np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(angular_momentum[i]))
                pert2[string] = HelperPert(hcc, hbar, lda, L[string], omega, local=local)

                i+=1
                for hand in ['right', 'left']:
                    pseudoresponse1 = pert1[string].iterate(hand, r_conv=r_conv, local=local)
                for hand in ['right', 'left']:
                    pseudoresponse2 = pert2[string].iterate(hand, r_conv=r_conv, local=local)

            print('Rosenfeld tensor:')
            for string1 in ['X', 'Y', 'Z']:
                for string2 in ['X', 'Y', 'Z']:
                    beta[string1+string2] = HelperResp(lda, pert1[string1], pert2[string2]).linear_resp()
                    betap[string1+string2] = HelperResp(lda, pert2[string2], pert1[string1]).linear_resp()

                    beta_new[string1+string2] = 0.5 * (beta[string1+string2] + betap[string1+string2])
                    print(' {} {} : {}'.format(string1, string2, beta_new[string1+string2]))

            trace = 0.0
            for string in ['XX','YY','ZZ']:
                trace += beta_new[string]
            trace /= 3.0

            # Calculation of the specific rotation
            Mass = 0
            for atom in range(mol.natom()):
                Mass += mol.mass(atom)
            h_bar = pc.h / (2.0 * np.pi)
            prefactor = -72e6 * h_bar**2 * pc.na / (pc.c**2 * pc.me**2 * Mass)
            optrot_diff = prefactor * trace

            optrot_mvg = optrot_vg - optrot_diff

            if return_en == True:
                return ccsd_e, optrot_mvg
            else:
                return optrot_mvg

