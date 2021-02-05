'''
HelperCCEnergy class definition and function definitions
For RHF CCSD calculations
Includes localization via a HelperLocal class
'''

import numpy as np
import psi4
from . import diis
from .diis import *
from opt_einsum import contract

class HelperCCEnergy(object):
    '''
    Class for setting up and running a CCSD correlation energy calculation.

    Spin-adapted equations using an RHF wavefunction.
    Need to instantiate the object and then call the do_cc() function 

    :param rhf_wfn: Psi4 wavefunction returned from RHF calculation
    :type rhf_wfn: class 'psi4.core.RHF' 
    :param local: Object containing functions for local correlation calculations
    :type local: class 'ccsd_lpno.HelperLocal'
    :param pert: Flag to use perturbed density (density is currently hardcoded)
    :type pert: bool
    :param pno_cut: Occupation number cutoff for truncating PNO space
    :type pno_cut: double
    :param e_cut: Weak pair cutoff for truncation of occupied pairs
    :type e_cut: double
    :param ppno_correction: Flag to compute the PNO++ correction
    :type ppno_correction: bool
    '''
    def __init__(self, rhf_wfn, local=None, local_occ=True, pert=False, pno_cut=0, e_cut=0, omega=0.0774):
        # Set energy and wfn from Psi4
        print(type(rhf_wfn))
        self.wfn = rhf_wfn

        # Get orbital coeffs from wfn
        C = self.wfn.Ca()
        C.print_out()
        self.C_arr = C.to_array()
        self.C_occ = self.wfn.Ca_subset("AO", "ACTIVE_OCC")
        basis = self.wfn.basisset()
        self.no_occ = self.wfn.doccpi()[0]
        self.no_mo = self.wfn.nmo()
        self.no_fz = self.wfn.frzcpi()[0]
        self.no_occ -= self.no_fz
        self.eps = np.asarray(self.wfn.epsilon_a())
        self.J = self.wfn.jk().J()[0].to_array()
        self.K = self.wfn.jk().K()[0].to_array()

        # Get No. of virtuals
        self.no_vir = self.no_mo - self.no_occ - self.no_fz
        print("Checking dimensions of orbitals: no_occ: {}\t no_fz: {}\t no_active: {}\t no_vir: {}\t total: {}".format(self.no_fz+self.no_occ, self.no_fz, self.no_occ, self.no_vir, self.no_occ+self.no_fz+self.no_vir))

        # Store the given cutoffs for localization
        self.pno_cut = pno_cut
        self.e_cut = e_cut

        self.mints = psi4.core.MintsHelper(self.wfn.basisset())

        self.H = np.asarray(self.mints.ao_kinetic()) + np.asarray(self.mints.ao_potential())

        self.I = self.mints.ao_eri()
        # Get localized occupied orbitals
        # Make MO integrals
        # Build Fock matrix
        if local is None:
            local_occ = False

        if local_occ:
            # Localizing occupied orbitals using Boys localization procedure
            print("Localize occupied orbital switch on. Localizing occupied orbitals.")
            Local = psi4.core.Localizer.build("PIPEK_MEZEY", basis, self.C_occ)
            Local.localize()
            new_C_occ = Local.L
            nc_arr = self.wfn.Ca().to_array()
            nco_arr = new_C_occ.to_array()
            print("Checking dimensions of localized active occupied:\nShape of local array: {}\nShape of C: {}\n".format(nco_arr.shape, nc_arr.shape))
            nc_arr[:, self.no_fz:(self.no_fz + self.no_occ)] = nco_arr[:,:]
            self.C_arr = psi4.core.Matrix.from_array(nc_arr)
            print("Shape of new MO coeff matrix: {}".format(self.C_arr.shape))
            self.MO_nfz = np.asarray(self.mints.mo_eri(self.C_arr, self.C_arr, self.C_arr, self.C_arr))
            self.MO = self.MO_nfz[self.no_fz:, self.no_fz:, self.no_fz:, self.no_fz:]
            print("Checking size of ERI tensor: {}".format(self.MO.shape))
            print("Checking size of ERI_nfz tensor: {}".format(self.MO_nfz.shape))

            # AO basis Fock matrix build
            De = contract('ui,vi->uv', nc_arr[:, :(self.no_fz+self.no_occ)], nc_arr[:, :(self.no_fz+self.no_occ)])
            self.F = self.H + 2.0 * contract('pqrs,rs->pq', self.I, De) - contract('prqs,rs->pq', self.I, De)
            self.F_nfz = contract('uj, vi, uv', self.C_arr, self.C_arr, self.F)
            self.F = self.F_nfz[self.no_fz:, self.no_fz:]
            print("Checking size of Fock matrix: {}".format(self.F.shape))
            self.F_ao = self.H + 2.0 * contract('pqrs,rs->pq', self.I, De) - contract('prqs,rs->pq', self.I, De)
            hf_e = contract('pq,pq->', self.H + self.F_ao, De)
        else:    
            self.MO_nfz = np.asarray(self.mints.mo_eri(C, C, C, C))
            self.MO = self.MO_nfz[self.no_fz:, self.no_fz:, self.no_fz:, self.no_fz:]
            self.F = self.H + 2.0 * self.J - self.K
            self.F_nfz = contract('uj, vi, uv', C, C, self.F)
            self.F = self.F_nfz[self.no_fz:, self.no_fz:]

        #test = self.H + 2.0 * self.J - self.K
        #test = contract('uj, vi, uv', C, C, test)

        # Need to change ERIs to physicist notation
        self.MO = self.MO.swapaxes(1, 2)

        # Need F_occ and F_vir separate (will need F_vir for semi-canonical basis later)
        self.F_occ = self.F[:self.no_occ, :self.no_occ]
        self.F_vir = self.F[self.no_occ:, self.no_occ:]

        #print("MO basis ERIs 2<ij|ab> - <ij|ba>:\n{}\n".format(2.0*self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:] - self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:].swapaxes(2,3)))
        #print("MO basis F_vir:\n{}\n".format(np.diag(self.F_vir)))
        #print("MO basis F_occ:\n{}\n".format(self.F_occ))

        # Once localized, the occupied orbital energies are no longer
        # equivalent to the diagonal of the Fock matrix
        self.eps_occ = np.diag(self.F_occ)
        self.eps_vir = np.diag(self.F_vir)

        # init T1s
        self.t_ia = np.zeros((self.no_occ, self.no_vir))

        # init T2s
        # note that occ.transpose(col) - vir(row) gives occ x vir matrix of differences
        self.d_ia = self.eps_occ.reshape(-1, 1) - self.eps_vir
        self.d_ijab = self.eps_occ.reshape(-1, 1, 1, 1) + self.eps_occ.reshape(-1, 1, 1) - self.eps_vir.reshape(-1, 1) - self.eps_vir
        #print("D2 denom:\n{}\n".format(1.0/self.d_ijab))
        self.t_ijab = self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:].copy()
        # T2s matching!
        self.t_ijab /= self.d_ijab
        #print("T2s:\n{}\n".format(self.t_ijab))
        mp2_e = 2.0 * contract('ijab,ijab->', self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:], self.t_ijab)
        mp2_e -= contract('ijba,ijab->', self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:], self.t_ijab)
        print("MP2 energy(without truncation): {}".format(mp2_e))


        if local:
            # Initialize PNOs
            print('Local switch on. Initializing PNOs.')
            # Identify weak pairs using MP2 pair corr energy
            self.e_ij = np.zeros((self.no_occ, self.no_occ))
            self.e_ij += 2.0 * contract('ijab,ijab->ij', self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:], self.t_ijab)
            self.e_ij -= contract('ijba,ijab->ij', self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:], self.t_ijab)
            #print('MP2 correlation energy: {}\n'.format(self.mp2_e))
            #print('Pair corr energy matrix:\n{}'.format(e_ij))
            str_pair_list = abs(self.e_ij) > e_cut
            print('Strong pair list:\n{}'.format(str_pair_list))

            if pert:
                print("Perturbed density on. Preparing perturbed density PNOs.")
                # Hbar_ii  = f_ii + t_inef ( 2 * <in|ef> - <in|fe> ) 
                Hbar_oo = self.F_occ.copy()
                Hbar_oo += 2.0 * contract('inef,mnef->mi', self.t_ijab, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
                Hbar_oo -= contract('inef,mnfe->mi', self.t_ijab, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])

                Hbar_ii = Hbar_oo.diagonal().copy()
                # Hbar_aa = f_aa - t_mnfa (2 * <mn|fa> - <mn|af> )
                Hbar_vv = self.F_vir.copy()
                Hbar_vv -= 2.0 * contract('mnfa,mnfe->ae', self.t_ijab, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
                Hbar_vv += contract('mnfa,mnef->ae', self.t_ijab, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
                Hbar_aa = Hbar_vv.diagonal().copy()
                denom_ia = Hbar_ii.reshape(-1,1) - Hbar_aa
                denom_ia += omega
                denom_ijab = Hbar_ii.reshape(-1, 1, 1, 1) + Hbar_ii.reshape(-1, 1, 1) - Hbar_aa.reshape(-1, 1) - Hbar_aa
                denom_ijab += omega
                self.denom_tuple = (denom_ia, denom_ijab)

                # Prepare the perturbation
                A_list = {}
                if pert == 'mu' or pert == 'mu+unpert':
                    ## Here, perturbation is dipole moment
                    dipole_array = self.mints.ao_dipole()
                    dirn = ['X','Y','Z']
                    for i in range(3):
                        A_list[dirn[i]] = np.einsum('uj,vi,uv', self.C_arr, self.C_arr, np.asarray(dipole_array[i]))
                        test = A_list[dirn[i]]
                    local.init_PNOs(pno_cut, self.t_ijab, self.F_vir, pert=pert, A_list=A_list, str_pair_list=str_pair_list, denom=self.denom_tuple)            
                if pert == 'l' or pert == 'l+unpert':
                    # Here, perturbation is angular momentum
                    angular_momentum = self.mints.ao_angular_momentum()
                    dirn = ['X','Y','Z']
                    for i in range(3):
                        A_list[dirn[i]] = np.einsum('uj,vi,uv', self.C_arr, self.C_arr, np.asarray(angular_momentum[i]))
                    local.init_PNOs(pno_cut, self.t_ijab, self.F_vir, pert=pert, A_list=A_list, str_pair_list=str_pair_list, denom=self.denom_tuple)            
                if pert == 'p' or pert == 'p+unpert':
                    # Here, perturbation is angular momentum
                    nabla = self.mints.ao_nabla()
                    dirn = ['X','Y','Z']
                    for i in range(3):
                        A_list[dirn[i]] = np.einsum('uj,vi,uv', self.C_arr, self.C_arr, np.asarray(nabla[i]))
                    local.init_PNOs(pno_cut, self.t_ijab, self.F_vir, pert=pert, A_list=A_list, str_pair_list=str_pair_list, denom=self.denom_tuple)            
                if pert == 'mu+l+unpert' or pert == 'pdt': 
                    ## Here, both perturbations are passed in to create the density
                    dipole_array = self.mints.ao_dipole()
                    angular_momentum = self.mints.ao_angular_momentum()
                    dirn = ['X','Y','Z']
                    A_list_2 = {}
                    for i in range(3):
                        A_list[dirn[i]] = np.einsum('uj,vi,uv', self.C_arr, self.C_arr, np.asarray(dipole_array[i]))
                        A_list_2[dirn[i]] = np.einsum('uj,vi,uv', self.C_arr, self.C_arr, np.asarray(angular_momentum[i]))
                    local.init_PNOs(pno_cut, self.t_ijab, self.F_vir, pert=pert, A_list=A_list, A_list_2=A_list_2, str_pair_list=str_pair_list, denom=self.denom_tuple)            
                '''
                # This section was used for computing the density as pdt of the perturbed density
                # and the other perturbation operator directly
                if pert == 'mu_pdt':
                    dipole_array = self.mints.ao_dipole()
                    angular_momentum = self.mints.ao_angular_momentum()
                    dirn = ['X','Y','Z']
                    A_list_2 = {}
                    for i in range(3):
                        A_list[dirn[i]] = np.einsum('uj,vi,uv', self.C_arr, self.C_arr, np.asarray(dipole_array[i]))
                        A_list_2[dirn[i]] = np.einsum('uj,vi,uv', self.C_arr, self.C_arr, np.asarray(angular_momentum[i]))
                    local.init_PNOs(pno_cut, self.t_ijab, self.F_vir, pert=pert, A_list=A_list, A_list_2=A_list_2, str_pair_list=str_pair_list, denom=self.denom_tuple)            
                if pert == 'l_pdt':
                    dipole_array = self.mints.ao_dipole()
                    angular_momentum = self.mints.ao_angular_momentum()
                    dirn = ['X','Y','Z']
                    A_list_2 = {}
                    for i in range(3):
                        A_list[dirn[i]] = np.einsum('uj,vi,uv', self.C_arr, self.C_arr, np.asarray(angular_momentum[i]))
                        A_list_2[dirn[i]] = np.einsum('uj,vi,uv', self.C_arr, self.C_arr, np.asarray(dipole_array[i]))
                    local.init_PNOs(pno_cut, self.t_ijab, self.F_vir, pert=pert, A_list=A_list, A_list_2=A_list_2, str_pair_list=str_pair_list, denom=self.denom_tuple)            
                '''
            else:
                local.init_PNOs(pno_cut, self.t_ijab, self.F_vir, str_pair_list=str_pair_list)            

            self.pno_correct = local.PNO_correction(self.t_ijab, self.MO)
            print("PNO correction:\n{}".format(self.pno_correct))
            Ria = np.zeros((self.no_occ, self.no_vir))
            self.tia, self.t_ijab = local.increment(Ria, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:], self.F_occ)
            #new_tia, new_t_ijab = local.increment(Ria, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:], self.F_occ)
            #print("The local filtered T2 matches original T2: {}".format(np.allclose(self.t_ijab, new_t_ijab)))
            #self.t_ijab = local.increment(Ria, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:], self.F_occ)
        print("MP2 energy here: {}".format(self.corr_energy(self.t_ia, self.t_ijab))) 

    # Make intermediates, Staunton:1991 eqns 3-11
    # Spin-adapted, every TEI term is modified to include
    # antisymmetrized term
    def make_taut(self, t_ia, t_ijab):
        tau_t = t_ijab + 0.5 * (contract('ia,jb->ijab', t_ia, t_ia))
        return tau_t


    def make_tau(self, t_ia, t_ijab):
        tau = t_ijab + (contract('ia,jb->ijab', t_ia, t_ia))
        return tau


    def make_Fae(self, taut, t_ia, t_ijab):
        Fae = self.F_vir.copy()
        #Fae[np.diag_indices_from(Fae)] = 0
        Fae -= 0.5 * contract('me,ma->ae', self.F[:self.no_occ, self.no_occ:], t_ia)
        Fae += 2.0 * contract('mf,mafe->ae', t_ia, self.MO[:self.no_occ, self.no_occ:, self.no_occ:, self.no_occ:])
        Fae -= contract('mf,maef->ae', t_ia, self.MO[:self.no_occ, self.no_occ:, self.no_occ:, self.no_occ:])
        Fae -= 2.0 * contract('mnaf,mnef->ae', taut, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        Fae += contract('mnaf,mnfe->ae', taut, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return Fae

    def make_Fmi(self, taut, t_ia, t_ijab):
        Fmi = self.F_occ.copy()
        #Fmi[np.diag_indices_from(Fmi)] = 0
        Fmi += 0.5 * contract('ie,me->mi', t_ia, self.F[:self.no_occ, self.no_occ:])
        Fmi += 2.0 * contract('ne,mnie->mi', t_ia, self.MO[:self.no_occ, :self.no_occ, :self.no_occ, self.no_occ:])
        Fmi -= contract('ne,mnei->mi', t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, :self.no_occ])
        Fmi += 2.0 * contract('inef,mnef->mi', taut, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        Fmi -= contract('inef,mnfe->mi', taut, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return Fmi


    def make_Fme(self, t_ia, t_ijab):
        Fme = self.F[:self.no_occ, self.no_occ:].copy()
        Fme += 2.0 * contract('nf,mnef->me', t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        Fme -= contract('nf,mnfe->me', t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return Fme


    def make_Wmnij(self, tau, t_ia, t_ijab):
        Wmnij = self.MO[:self.no_occ, :self.no_occ, :self.no_occ, :self.no_occ].copy()
        Wmnij += contract('je,mnie->mnij', t_ia, self.MO[:self.no_occ, :self.no_occ, :self.no_occ, self.no_occ:])
        Wmnij += contract('ie,mnej->mnij', t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, :self.no_occ])
        Wmnij += contract('ijef,mnef->mnij', tau, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return Wmnij


    def make_Wmbej(self, t_ia, t_ijab):
        Wmbej = self.MO[:self.no_occ, self.no_occ:, self.no_occ:, :self.no_occ].copy()
        Wmbej += contract('jf,mbef->mbej', t_ia, self.MO[:self.no_occ, self.no_occ:, self.no_occ:, self.no_occ:])
        Wmbej -= contract('nb,mnej->mbej', t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, :self.no_occ])
        tmp = 0.5 * t_ijab.copy() + contract('jf,nb->jnfb', t_ia, t_ia)
        Wmbej -= contract('jnfb,mnef->mbej', tmp, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        Wmbej += contract('njfb,mnef->mbej', t_ijab, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        Wmbej -= 0.5 * contract('njfb,mnfe->mbej', t_ijab, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return Wmbej


    def make_Wmbje(self, t_ia, t_ijab):
        Wmbje = -1.0 * self.MO[:self.no_occ, self.no_occ:, :self.no_occ, self.no_occ:].copy()
        Wmbje -= contract('jf,mbfe->mbje', t_ia, self.MO[:self.no_occ, self.no_occ:, self.no_occ:, self.no_occ:])
        Wmbje += contract('nb,mnje->mbje', t_ia, self.MO[:self.no_occ, :self.no_occ, :self.no_occ, self.no_occ:])
        tmp = 0.5 * t_ijab.copy() + contract('jf,nb->jnfb', t_ia, t_ia)
        Wmbje += contract('jnfb,mnfe->mbje', tmp, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return Wmbje


    def make_Zmbij(self, tau):
        Zmbij = 0
        Zmbij += contract('mbef,ijef->mbij', self.MO[:self.no_occ, self.no_occ:, self.no_occ:, self.no_occ:], tau)
        return Zmbij


    def update_ts(self, tau, tau_t, t_ia, t_ijab, local=None):
        ''' 
        Update T1 and T2 amplitudes

        :param tau: T + 1/2 (T1^2)
        :type tau: numpy array
        :param tau_t: tau^+
        :type tau_t: numpy array
        :param t_ia: old T1 amplitude to be used in the residual equations
        :type t_ia: numpy array
        :param t_ijab: old T2 amplitude to be used in the residual equations
        :type t_ijab: numpy array
        :param local: Object containing the increment function for local correlation calculations
        :type local: class 'ccsd_lpno.HelperLocal'

        :returns: Updated T1 and T2 amplitudes
        :rtype: numpy arrays
        '''
        no_occ = t_ia.shape[0]

        # Build intermediates
        Fae = self.make_Fae(tau_t, t_ia, t_ijab)
        Fmi = self.make_Fmi(tau_t, t_ia, t_ijab)
        Fme = self.make_Fme(t_ia, t_ijab)

        Wmnij = self.make_Wmnij(tau, t_ia, t_ijab)
        Wmbej = self.make_Wmbej(t_ia, t_ijab)
        Wmbje = self.make_Wmbje(t_ia, t_ijab)
        Zmbij = self.make_Zmbij(tau)

        # Create residual T1s
        Ria = self.F[:no_occ, no_occ:].copy()
        Ria += contract('ie,ae->ia', t_ia, Fae)
        Ria -= contract('ma,mi->ia', t_ia, Fmi)
        Ria += 2.0 * contract('imae,me->ia', t_ijab, Fme)
        Ria -= contract('imea,me->ia', t_ijab, Fme)
        Ria -= contract('nf,naif->ia', t_ia, self.MO[:no_occ, no_occ:, :no_occ, no_occ:])
        Ria += 2.0 * contract('nf,nafi->ia', t_ia, self.MO[:no_occ, no_occ:, no_occ:, :no_occ])
        Ria += 2.0 * contract('mief,maef->ia', t_ijab, self.MO[:no_occ, no_occ:, no_occ:, no_occ:])
        Ria -= contract('mife,maef->ia', t_ijab, self.MO[:no_occ, no_occ:, no_occ:, no_occ:])
        Ria -= 2.0 * contract('mnae,nmei->ia', t_ijab, self.MO[:no_occ, :no_occ, no_occ:, :no_occ])
        Ria += contract('mnae,nmie->ia', t_ijab, self.MO[:no_occ, :no_occ, :no_occ, no_occ:])

        # Create residual T2s
        Rijab = self.MO[:no_occ, :no_occ, no_occ:, no_occ:].copy()
        # Term 2
        tmp = contract('ijae,be->ijab', t_ijab, Fae)
        Rijab += tmp 
        Rijab += tmp.swapaxes(0, 1).swapaxes(2, 3)

        tmp = 0.5 * contract('ijae,mb,me->ijab', t_ijab, t_ia, Fme)
        Rijab -= tmp
        Rijab -= tmp.swapaxes(0, 1).swapaxes(2, 3)
        # Term 3
        tmp = contract('imab,mj->ijab', t_ijab, Fmi)
        Rijab -= tmp
        Rijab -= tmp.swapaxes(0, 1).swapaxes(2, 3)

        tmp = 0.5 * contract('imab,je,me->ijab', t_ijab, t_ia, Fme)
        Rijab -= tmp 
        Rijab -= tmp.swapaxes(0, 1).swapaxes(2, 3)
        # Term 4
        Rijab += contract('mnab,mnij->ijab', tau, Wmnij)
        # Term 5
        Rijab += contract('ijef,abef->ijab', tau, self.MO[no_occ:, no_occ:, no_occ:, no_occ:])
        # Extra term since Wabef is not formed
        tmp = contract('ma,mbij->ijab', t_ia, Zmbij)
        Rijab -= tmp
        Rijab -= tmp.swapaxes(0, 1).swapaxes(2, 3)
        # Term 6 # 1
        tmp = contract('imae,mbej->ijab', t_ijab, Wmbej)
        tmp -= contract('imea,mbej->ijab', t_ijab, Wmbej)
        Rijab += tmp
        Rijab += tmp.swapaxes(0, 1).swapaxes(2, 3)
        tmp1 = contract('ie,ma,mbej->ijab', t_ia, t_ia, self.MO[:no_occ, no_occ:, no_occ:, :no_occ])
        Rijab -= tmp1
        Rijab -= tmp1.swapaxes(0, 1).swapaxes(2, 3)
        # Term 6 # 2
        tmp = contract('imae,mbej->ijab', t_ijab, Wmbej)
        tmp += contract('imae,mbje->ijab', t_ijab, Wmbje)
        Rijab += tmp
        Rijab += tmp.swapaxes(0, 1).swapaxes(2, 3)
        # Term 6 # 3
        tmp = contract('mjae,mbie->ijab', t_ijab, Wmbje)
        Rijab += tmp
        Rijab += tmp.swapaxes(0, 1).swapaxes(2, 3)
        tmp1 = contract('ie,mb,maje->ijab', t_ia, t_ia, self.MO[:no_occ, no_occ:, :no_occ, no_occ:])
        Rijab -= tmp1
        Rijab -= tmp1.swapaxes(0, 1).swapaxes(2, 3)

        # Term 7
        tmp = contract('ie,abej->ijab', t_ia, self.MO[no_occ:, no_occ:, no_occ:, :no_occ])
        Rijab += tmp
        Rijab += tmp.swapaxes(0, 1).swapaxes(2, 3)
        # Term 8
        tmp = contract('ma,mbij->ijab', t_ia, self.MO[:no_occ, no_occ:, :no_occ, :no_occ])
        Rijab -= tmp
        Rijab -= tmp.swapaxes(0, 1).swapaxes(2, 3)

        # Update T1s
        new_tia =  t_ia.copy() 
        new_tijab = t_ijab.copy()

        # Update T2s
        if local:
            inc1, inc2 = local.increment(Ria, Rijab, self.F_occ)
            #inc2 = local.increment(Ria, Rijab, self.F_occ)
            #new_tia += Ria / self.d_ia
            new_tia += inc1
            new_tijab += inc2
        else:
            # Apply denominators
            new_tia += Ria / self.d_ia
            new_tijab += Rijab / self.d_ijab

        return new_tia, new_tijab

    def corr_energy(self, t_ia, t_ijab):
        '''
        Compute CCSD correlation energy

        :param t_ia: T1 used for corr energy calculation
        :type t_ia: numpy array
        :param t_ijab: T2 used for corr energy calculation
        :type t_ijab: numpy array

        :return: correlation energy
        :rtype: double
        '''
        no_occ = t_ia.shape[0] 
        E_corr = 2.0 * contract('ia,ia->', self.F[:no_occ, no_occ:], t_ia)
        singles_val = E_corr
        tmp_tau = self.make_tau(t_ia, t_ijab)
        E_corr += 2.0 * contract('ijab,ijab->', self.MO[:no_occ, :no_occ, no_occ:, no_occ:], tmp_tau)
        E_corr -= contract('ijba,ijab->', self.MO[:no_occ, :no_occ, no_occ:, no_occ:], tmp_tau)
        doubles_val = E_corr - singles_val
        
        # Looking at singles and doubles contri
        #print("Singles contribution: {}".format(singles_val))
        #print("Doubles contribution: {}".format(doubles_val))
        return E_corr

    def do_CC(self, local=None, e_conv=1e-8, r_conv=1e-7, maxiter=40, max_diis=8, start_diis=0):
        '''
        Do CCSD iterations with DIIS and local options

        :param local: Object containing the increment function for local correlation calculations
        :type local: class 'ccsd_lpno.HelperLocal'
        :param e_conv: Convergence threshold for pseudoenergy
        :type e_conv: double
        :param r_conv: Convergence threshold for Lambda RMSDs
        :type r_conv: double
        :param maxiter: Maximum no. of iterations
        :type maxiter: integer
        :param max_diis: Maximum no. of error vectors stored for DIIS
        :type max_diis: integer
        :param start_diis: Which iteration to start storing error vectors for DIIS
        :type start_diis: integer

        :return: Converged pseudoenergy
        :rtype: double
        '''
        self.old_e = self.corr_energy(self.t_ia, self.t_ijab)
        print('Iteration\t\t Correlation energy\tDifference\tRMS\nMP2\t\t\t {}'.format(self.old_e))
    # Set up DIIS
        diis = HelperDIIS(self.t_ia, self.t_ijab, max_diis)
        
        new_e = self.old_e
    # Iterate until convergence
        for i in range(maxiter):
            tau_t = self.make_taut(self.t_ia, self.t_ijab)
            tau = self.make_tau(self.t_ia, self.t_ijab)
            new_tia, new_tijab = self.update_ts(tau, tau_t, self.t_ia, self.t_ijab, local=local)
            new_e = self.corr_energy(new_tia, new_tijab)
            rms = np.linalg.norm(new_tia - self.t_ia)
            rms += np.linalg.norm(new_tijab - self.t_ijab)
            print('CC Iteration: {:3d}\t {:2.12f}\t{:1.12f}\t{:1.12f}\tDIIS Size: {}'.format(i, new_e, abs(new_e - self.old_e), rms, diis.diis_size))
            if(abs(new_e - self.old_e) < e_conv and abs(rms) < r_conv):
                print('Convergence reached.\n CCSD Correlation energy: {}\n'.format(new_e))
                self.t_ia = new_tia
                self.t_ijab = new_tijab
                break
            # Update error vectors for DIIS
            diis.update_err_list(new_tia, new_tijab)
            # Extrapolate using DIIS
            if(i >= start_diis):
                new_tia, new_tijab = diis.extrapolate(new_tia, new_tijab)

            self.t_ia = new_tia
            self.t_ijab = new_tijab
            self.old_e = new_e

        return new_e
