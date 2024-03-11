import numpy as np

import pycbc

import lal
import lalsimulation

import numpy as np
import scipy as sp
from scipy.optimize import minimize
from scipy.interpolate import interp1d

import random
from copy import deepcopy

import sys
sys.path.append("/home/nishkal.rao/GWMAT/pnt_Ff_lookup_table/src/cythonized_pnt_lens_class")   
import cythonized_pnt_lens_class as pnt_lens_cy 
sys.path.append('/home/nishkal.rao/gweat/src/')
import TEOBResumS_utils as ecc_gen

# constants
G = 6.67430*1e-11 
c = 299792458. 
M_sun = 1.989*1e30

def m1m2_to_mchirp(m1, m2):
    """
    Returns the chirp mass corresponding to the binary component masses.

    Parameters
    ----------
    m1 : float
        Mass of the first component of the binary, in Msun.
    m2 : float
        Mass of the second component of the binary, in Msun.

    Returns
    -------
    float
        Chirp mass, in Msun.

    """

    return (m1*m2)**(3/5.)/(m1+m2)**(1/5.)

def mchirp_q_to_m1m2(m_chirp, q):
    """
    Converts a given chirp mass and mass ratio to the binary component masses.

    Parameters
    ----------
    m_chirp : float
        Chirp mass of the binary.
    q : float
        Mass ratio of the binary.

    Returns
    -------
    float, float
        The binary component masses    

    """

    m2=(m_chirp*(1.+q)**(1/5.))/q**(3/5.)
    m1=q*m2
    return m1, m2

def mtot_q_to_mchirp(mtot, q): 
    """
    Converts a given total mass and mass ratio of the binary to its chirp mass.

    Parameters
    ----------
    mtot : float
        The total mass of the binary.
    q : float
        The mass ratio of the binary.

    Returns
    -------
    float
        Chirp mass of the binary.

    """        

    m1=mtot/(1+q)
    m2=q*m1
    return (m1*m2)**(3/5.)/(m1+m2)**(1/5.)

def mtot_q_to_m1m2(mtot, q): 
    """
    Converts a given total mass and mass ratio of the binary to the individual componenet masses.

    Parameters
    ----------
    mtot : float
        The total mass of the binary.
    q : float
        The mass ratio of the binary.

    Returns
    -------
    float, float
        Binary componenet masses.

    """        

    m1=mtot/(1+q)
    m2=q*m1
    return max(m1, m2),  min(m1, m2)

############################################################################################################################################################################
############################################################################################################################################################################

class CBC_parms_domain:
    """
    Contains functions relevant for fitting factor computation.

    """
    def wrap_reflective(self, x, x1, x2):
        """
        Function to wrap and reflect a real number around the points x1 and x2. 
        Example - For spins, we will have 'wrap_reflective(1.1, -1, 1) = 0.9'; 'wrap_reflective(-1.1, -1, 1) = -0.9'; and so on.

        Parameters
        ----------
        x : float
            Value to be reflected.
        x1 : float
            The LHS reflective point.
        x2 : float
            The RHS reflective point.

        Returns
        -------
        float
            Wrapped and reflective value of x around x1 and x2.

        """        

        if x2 == None: #i.e., x2 can be unbounded
            x2 = 1e4   #assign a big number (but not np.inf as it will bias the algebra)
        period = 2*(x2 - x1)
        x_frac = (x - x1) % period
        if x_frac <= period/2:
            x_ref = x_frac + x1  
        elif x_frac > period/2:
            x_ref = (period - x_frac) + x1 
        return x_ref

    # function to wrap a real number periodically around the points x1 and x2. 
    # Ex- for a sine func, we will have 'periodic(2*np.pi + x, 0, 2*np.pi) = x'.
    def wrap_periodic(self, x, x1, x2):
        """
        Function to wrap a real number periodically around the points x1 and x2. 
        Example - For spins, we will have 'wrap_periodic(2*np.pi + x, 0, 2*np.pi) = x'.

        Parameters
        ----------
        x : float
            Value to be reflected.
        x1 : float
            The LHS coordinate of boundary.
        x2 : float
            The RHS coordinate of boundary.

        Returns
        -------
        float
            Periodically wrapped value of x around x1 and x2.

        """    

        period = (x2 - x1)
        x_frac = (x - x1) % period
        x_p = x_frac + x1  
        return x_p

    # wraps x between (x1, x2) assuming boundaries to be either periodic or reflective.
    def wrap(self, x, x1, x2, boundary='reflective'):
        """
        Function to wrap a real number around the points x1 and x2 either periodically or reflectively. 
        Example - (i) For spins, we will have 'wrap(2*np.pi + x, 0, 2*np.pi, 'periodic') = x';
        'wrap(1.1, -1, 1, 'reflective') = 0.9'; 'wrap(-1.1, -1, 1, 'reflective') = -0.9'.

        Parameters
        ----------
        x : float
            Value to be reflected.
        x1 : float
            The LHS coordinate of boundary.
        x2 : float
            The RHS coordinate of boundary.
        boundary : {'reflective', 'periodic'}, optional.
            Boundary type to conisder while wrapping. Default = 'reflective'.
        Returns
        -------
        float
            Periodically wrapped value of x around x1 and x2.

        Raises
        ------
        KeyError
            Allowed keywords for boundary are: {'reflective', 'periodic'}.

        """        

        if (boundary == 'reflective' or boundary == 'Reflective'):
            return self.wrap_reflective(x, x1, x2)
        elif (boundary == 'periodic' or boundary == 'Periodic'):
            return self.wrap_periodic(x, x1, x2)
        else:
            raise KeyError("Incorrect keyword provided for the argument 'boundary'. Allowed keywords are: {'reflective', 'periodic'}.")

    def ext_sp(self, a):
        """
        Checks if the dimensionless spin magnitude `a` satisfies 0.998 < s < 1, otherwise assign a = 0.9.
        """
        if (abs(round(a, 3)) > 0.998):
            return 0.9*a/np.abs(a)
        else:
            return a

    def dom_indv_sp(self, x):
        """
        Domain of an individual spin component: wrapping and reflection of real line around (-1, 1).
        
        """

        sp_ref = self.wrap(x, -1., 1., boundary='reflective')
        sp_ref = self.ext_sp(sp_ref)
        return sp_ref  

    def dom_mag_sp(self, sp):
        """
        Domain of three spin components: ensures that spin magnitude is less than one for a given set of three spin components.

        """        

        try:
            assert len(sp) == 3, 'Spin should have three components [s_x, s_y, s_z], but entered spin has length = {} instead'.format(len(sp))
            sp = np.array(sp)
            a = np.linalg.norm([sp[0], sp[1], sp[2]])
            if a != 0:
                a_new = self.ext_sp(a)
                sp = a_new*sp/a
            return sp 
        except TypeError:
            return self.dom_indv_sp(sp) 
    
    def dom_sp(self, sp):
        """
        Final combined function for wrapping of spin values - can handle both 3-component and 1-component spin values.

        Parameters
        ----------
        sp : {float, list}
            Spin value(s).

        Returns
        -------
        {float, list}
            Wrapped spin value(s).

        """

        try:  
            sp = list(map(lambda s: self.dom_indv_sp(s), sp))
            sp = self.dom_mag_sp(sp)
        except TypeError:
            sp = self.dom_indv_sp(sp)
        return sp 
   

    def dom_m(self, x, m_min=3.5, m_max=None):
        """
        Returns wrapped mass value(s): wrapping and reflection of real line around (3.2, \inf), 
        where \inf is used so that `m > 3.2` is the only real restriction.

        Parameters
        ----------
        x : float
            Mass value to be wrapped within domain.
        m_min : float, optional
            Minimum mass to consider while wrapping. Default = 3.5.
        m_max : {None, float}, optional
            Maximum mass to consider while wrapping. Default = None.

        Returns
        -------
        float
            Wrapped Mass value within domain.

        """

        m_ref = self.wrap(x, m_min, m_max, boundary='reflective')
        return m_ref     

    def dom_chirp(self, x, cm_min=3.05, cm_max=None):   # because chirp(3.5, 3.5) ~ 3.05
        """
        Returns wrapped Chirp Mass value(s): wrapping and reflection of real line around (3, 1e4), 
        where 1e4 is a large enough number so that `CM > 3` is the only real restriction.

        Parameters
        ----------
        x : float
            Chirp mass value to be wrapped within domain.
        cm_min : float, optional
            Minimum Chirp mass to consider while wrapping. Default = 3.5.
        cm_max : {None, float}, optional
            Maximum Chirp mass to consider while wrapping. Default = None.

        Returns
        -------
        float
            Wrapped Chirp Mass value within domain.

        """       

        cm_ref = self.wrap(x, cm_min, cm_max, boundary='reflective')
        return cm_ref   

    # domain of Mass Ratio values: wrapping and reflection of real line around (~0, 1).
    def dom_q(self, x, q_min=1/18., q_max=1):
        """
        Returns wrapped mass ratio value(s): wrapping and reflection of real line around (~0, 1),
        assuming q = min(m1/m2, m2/m1) \in (0, 1).

        Parameters
        ----------
        x : float
            Mass ratio value to be wrapped within domain.
        q_min : float, optional
            Minimum mass ratio to consider while wrapping. Default = 3.5.
        q_max : {None, float}, optional
            Maximum mass ratio to consider while wrapping. Default = None.

        Returns
        -------
        float
            Wrapped mass ratio value within domain.

        """  

        x_wrap = self.wrap(x, q_min, q_max, boundary='reflective')
        return x_wrap

    # domain of Symmetric Mass Ratio values: wrapping and reflection of real line around (~0, 1/4).
    def dom_eta(self, x, eta_min=0.05, eta_max=1/4.):
        """
        Returns wrapped symmetric mass ratio value(s): wrapping and reflection of real line around (~0, 1/4.).

        Parameters
        ----------
        x : float
            Mass ratio value to be wrapped within domain.
        eta_min : float, optional
            Minimum symmetric mass ratio to consider while wrapping. Default = 3.5.
        eta_max : {None, float}, optional
            Maximum symmetric mass ratio to consider while wrapping. Default = None.

        Returns
        -------
        float
            Wrapped symmetric mass ratio value within domain.

        """  

        x_wrap = self.wrap(x, eta_min, eta_max, boundary='reflective')
        return x_wrap 

############################################################################################################################################################################
############################################################################################################################################################################

def match_wfs_fd(wf1, wf2, psd=None, f_low = 20., f_high=None, subsample_interpolation=True, is_asd_file=False):
    """
    Computes match (overlap maximised over phase and time) between two frequency domain WFs.

    Parameters
    ----------
    wf1 : pycbc.types.frequencyseries.FrequencySeries object
        PyCBC time domain Waveform.
    wf2 : pycbc.types.frequencyseries.FrequencySeries object
        PyCBC time domain Waveform.
    psd: {None, str}
        PSD file to use for computing match. Default = None.
        Predefined_PSDs: {'aLIGOZeroDetHighPower'}
    f_low : {None, float}, optional
        The lower frequency cutoff for the match computation. Default = 20.
    f_high : {None, float}, optional
        The upper frequency cutoff for the match computation. Default = None.
    subsample_interpolation : ({False, bool}, optional)
        If True the peak will be interpolated between samples using a simple quadratic fit. 
        This can be important if measuring matches very close to 1 and can cause discontinuities if you don’t use it as matches move between discrete samples. 
        If True the index returned will be a float instead of int. Default = True.
    is_asd_file : bool, optional
        Is psd provided corresponds to an asd file? Default = False.

    Returns
    -------
    match_val : float
        match value Phase to rotate complex waveform to get the match, if desired.
    index_shift : float
        The number of samples to shift to get the match.
    phase_shift : float
        Phase to rotate complex waveform to get the match, if desired.   

    """ 

    flen = max(len(wf1), len(wf2))
    wf1.resize(flen)
    wf2.resize(flen)

    delta_f = wf1.delta_f
    if psd is not None:
        if psd=='aLIGOZeroDetHighPower':
            psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, f_low)
        else:
            psd = pycbc.psd.from_txt(psd, flen, delta_f, f_low, is_asd_file=is_asd_file)   
    # match_val, index_shift, phase_shift = match( wf1, wf2, psd=psd, low_frequency_cutoff=f_low, high_frequency_cutoff=f_high, subsample_interpolation=subsample_interpolation, return_phase=True )
    match_val, index_shift = pycbc.filter.match( wf1, wf2, psd=psd, low_frequency_cutoff=f_low, high_frequency_cutoff=f_high )
    return match_val, index_shift #, phase_shift   

def match_wfs_td(wf1, wf2, psd=None, f_low = 20., f_high=None, subsample_interpolation=True, is_asd_file=False):    
    """
    Computes match (overlap maximised over phase and time) between two time domain WFs.

    Parameters
    ----------
    wf1 : pycbc.types.timeseries.TimeSeries object
        PyCBC time domain Waveform.
    wf2 : pycbc.types.timeseries.TimeSeries object
        PyCBC time domain Waveform.
    psd: {None, str}
        PSD file to use for computing match. Default = None.
        Predefined_PSDs: {'aLIGOZeroDetHighPower'}
    f_low : {None, float}, optional
        The lower frequency cutoff for the match computation. Default = 20.
    f_high : {None, float}, optional
        The upper frequency cutoff for the match computation. Default = None.
    subsample_interpolation : ({False, bool}, optional)
        If True the peak will be interpolated between samples using a simple quadratic fit. 
        This can be important if measuring matches very close to 1 and can cause discontinuities if you don’t use it as matches move between discrete samples. 
        If True the index returned will be a float instead of int. Default = True.
    is_asd_file : bool, optional
        Is psd provided corresponds to an asd file? Default = False.

    Returns
    -------
    match_val : float
        match value Phase to rotate complex waveform to get the match, if desired.
    index_shift : float
        The number of samples to shift to get the match.
    phase_shift : float
        Phase to rotate complex waveform to get the match, if desired.    

    """   

    tlen = max(len(wf1), len(wf2))
    wf1.resize(tlen)
    wf2.resize(tlen)

    delta_f = wf1.delta_f
    flen = tlen//2+1
    
    if psd is not None:
        if psd=='aLIGOZeroDetHighPower':
            psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, f_low)
        else:
            psd = pycbc.psd.from_txt(psd, flen, delta_f, f_low, is_asd_file=is_asd_file)      
    # match_val, index_shift, phase_shift = match(wf1, wf2, psd=psd, low_frequency_cutoff=f_low, high_frequency_cutoff=f_high, subsample_interpolation=subsample_interpolation, return_phase=True)
    match_val, index_shift = pycbc.filter.match(wf1, wf2, psd=psd, low_frequency_cutoff=f_low, high_frequency_cutoff=f_high)
    return match_val, index_shift #, phase_shift

############################################################################################################################################################################
############################################################################################################################################################################

class FF_2D_zero_spin(CBC_parms_domain):
    """
    Functions for `wf_model` == '2D'.

    """

    def gen_seed_prm_2D(self, chirp_mass=25, q=1, sigma_mchirp=1, sigma_q=0.2):
        """
        Generates seed point in 2D [m1, m2] for match maximisation; uses reasonable initial bounds.  
        
        """ 

        mchirp = np.random.normal(chirp_mass, sigma_mchirp, 1)[0]
        q = np.random.normal(q, sigma_q, 1)[0]
        return [self.dom_chirp(mchirp), self.dom_q(q)]
    
    def gen_seed_near_best_fit_2D(self, x, sigma_mchirp = 0.5, sigma_q = 0.1):
        chirp_mass, q = x
        mchirp = np.random.normal(chirp_mass, sigma_mchirp, 1)[0]
        q = np.random.normal(q, sigma_q, 1)[0]
        x_near = [self.dom_chirp(mchirp), self.dom_q(q)]
        return x_near
    
    def wf_gen_td(self, prms, dt, f_low=20, apx="IMRPhenomXPHM", **kwargs):
        """
        Generates timedomain Wf.

        """

        m1, m2 = prms
        td_hp, td_hc = pycbc.waveform.get_td_waveform(
            approximant = apx,
            mass1 = m1,
            mass2 = m2,
            spin1z = 0,
            spin2z = 0,
            distance = 100.,
            coa_phase = kwargs['coa_phase'],
            inclination = kwargs['inclination'],
            f_ref = kwargs['f_ref'],
            f_lower = f_low,
            delta_t = dt) 
        return td_hp, td_hc
    
    def wf_gen_fd(self, prms, df, f_low=20, apx="IMRPhenomXPHM", **kwargs):
        """
        Generates frequencydomain Wf.

        """

        m1, m2 = prms
        fd_hp, fd_hc = pycbc.waveform.get_fd_waveform(
            approximant = apx,
            mass1 = m1,
            mass2 = m2,
            spin1z = 0,
            spin2z = 0,
            distance = 100.,
            coa_phase = kwargs['coa_phase'],
            inclination = kwargs['inclination'],
            f_ref = kwargs['f_ref'],
            f_lower = f_low,
            delta_f = df) 
        return fd_hp, fd_hc
    
    def objective_func_2D(self, x, *args):
        """
        Objective function for the maximisation/minimsation.

        """

        x[0], x[1] = self.dom_chirp(x[0]), self.dom_q(x[1])
        m1, m2 = mchirp_q_to_m1m2(x[0], x[1])
        if (m1 < 3.5 or m2 < 3.5):
            return 1e4
        
        gen_prms = m1, m2

        signal, f_low, f_high, apx, kwargs = args
        dt_lw = signal.delta_t

        try:
            template = self.wf_gen_td(gen_prms, dt = dt_lw, f_low=f_low, apx=apx, **kwargs)[0]
            template.start_time = signal.start_time
            return np.log(1-match_wfs_td(signal, template, f_low=f_low, f_high=f_high, psd=None)[0])
        except ValueError:
            return 1e4
        
class FF_4D_aligned_spin(FF_2D_zero_spin):
    """
    Functions for `wf_model` == '4D'.

    """
    
    def gen_seed_prm_4D(self, chirp_mass=25, q=1, chi_1=0, chi_2=0, sigma_mchirp=1, sigma_q=0.2, sigma_chi=0.2):
        """
        Generates seed point in 4D [m1, m2, s_1z, s_2z] for match maximisation; uses reasonable initial bounds.  
        
        """ 

        mchirp = np.random.normal(chirp_mass, sigma_mchirp, 1)[0]
        q = np.random.normal(q, sigma_q, 1)[0]
        sz1 = np.random.normal(chi_1, sigma_chi, 1)[0]
        sz2 = np.random.normal(chi_2, sigma_chi, 1)[0]
        return [self.dom_chirp(mchirp), self.dom_q(q), self.dom_sp(sz1), self.dom_sp(sz2)]    

    def gen_seed_near_best_fit_4D(self, x, sigma_mchirp = 0.5, sigma_q = 0.1, sigma_chi = 0.1):
        chirp_mass, q, a1, a2 = x
        mchirp = np.random.normal(chirp_mass, sigma_mchirp, 1)[0]
        q = np.random.normal(q, sigma_q, 1)[0]
        sz1 = np.random.normal(a1, sigma_chi, 1)[0]
        sz2 = np.random.normal(a2, sigma_chi, 1)[0]
        x_near = [self.dom_chirp(mchirp), self.dom_q(q), self.dom_sp(sz1), self.dom_sp(sz2)]
        return x_near
    
    def wf_gen_td(self, prms, dt, f_low=20, apx="IMRPhenomXPHM", **kwargs): 
        """
        Generates timedomain Wf.

        """

        m1, m2, sz1, sz2 = prms
        spin1x, spin1y, sz1 = self.dom_sp([kwargs['spin1x'], kwargs['spin1y'], sz1])
        spin2x, spin2y, sz2 = self.dom_sp([kwargs['spin2x'], kwargs['spin2y'], sz2])
        td_hp, td_hc = pycbc.waveform.get_td_waveform(
            approximant = apx,
            mass1 = m1,
            mass2 = m2,
            spin1z = sz1,
            spin2z = sz2,
            spin1x = spin1x,
            spin1y = spin1y,
            spin2x = spin2x,
            spin2y = spin2y,
            distance = 100.,
            coa_phase = kwargs['coa_phase'],
            inclination = kwargs['inclination'],
            f_ref = kwargs['f_ref'],
            f_lower = f_low,
            delta_t = dt) 
        return td_hp, td_hc
   
    def wf_gen_fd(self, prms, df, f_low=20, apx="IMRPhenomXPHM", **kwargs): 
        """
        Generates frequencydomain Wf.

        """

        m1, m2, sz1, sz2 = prms
        spin1x, spin1y, sz1 = self.dom_sp([kwargs['spin1x'], kwargs['spin1y'], sz1])
        spin2x, spin2y, sz2 = self.dom_sp([kwargs['spin2x'], kwargs['spin2y'], sz2])
        fd_hp, fd_hc = pycbc.waveform.get_fd_waveform(
            approximant = apx,
            mass1 = m1,
            mass2 = m2,
            spin1z = sz1,
            spin2z = sz2,
            spin1x = spin1x,
            spin1y = spin1y,
            spin2x = spin2x,
            spin2y = spin2y,
            distance = 100.,
            coa_phase = kwargs['coa_phase'],
            inclination = kwargs['inclination'],
            f_ref = kwargs['f_ref'],
            f_lower = f_low,
            delta_f = df) 
        return fd_hp, fd_hc
            
    def objective_func_4D(self, x, *args):    
        """
        Objective function for the maximisation/minimsation.

        """

        x[0], x[1] = self.dom_chirp(x[0]), self.dom_q(x[1])
        m1, m2 = mchirp_q_to_m1m2(x[0], x[1])
        if (m1 < 3.5 or m2 < 3.5):  # if mchirp and q doesn't lead to reasonable binary masses, they will be avoided
            return 1e4

        x[2], x[3] = self.dom_sp(x[2]), self.dom_sp(x[3])

        gen_prms = m1, m2, x[2], x[3]

        signal, f_low, f_high, apx, kwargs = args
        dt_lw = signal.delta_t

        try:
            template = self.wf_gen_td(gen_prms, dt = dt_lw, f_low=f_low, apx=apx, **kwargs)[0]
            template.start_time = signal.start_time
            return np.log(1-match_wfs_td(signal, template, f_low=f_low, f_high=f_high, psd=None)[0])
        except ValueError:
            return 1e4

############################################################################################################################################################################
############################################################################################################################################################################

class FF_EC_3D_zero_spin(FF_2D_zero_spin):
    """
    Functions for `wf_model` == 'EC_3D'.

    """
    
    def gen_seed_prm_EC_3D(self, chirp_mass=25, q=1, ecc=0.1, ecc_min=1e-3, ecc_max=.5, sigma_mchirp=0.5, sigma_q=0.2, sigma_ecc=0.5):
        """
        Generates seed point in 3D [m1, m2, e] for match maximisation; uses reasonable initial bounds.

        """ 

        mchirp = np.random.normal(chirp_mass, sigma_mchirp, 1)[0]
        q = np.random.normal(q, sigma_q, 1)[0]
        ecc = sp.stats.loguniform.rvs(ecc_min, ecc_max, 1)
        return [self.dom_chirp(mchirp), self.dom_q(q), self.wrap_reflective(ecc, ecc_min, ecc_max)]    

    def gen_seed_near_best_fit_EC_3D(self, x, sigma_ecc = 0.5, sigma_mchirp = 0.5, sigma_q = 0.1):
        chirp_mass, q, ecc = x

        mchirp = np.random.normal(chirp_mass, sigma_mchirp, 1)[0]
        q = np.random.normal(q, sigma_q, 1)[0]
        ecc = np.random.normal(ecc, sigma_ecc, 1)[0]
        x_near = [self.dom_chirp(mchirp), self.dom_q(q), self.wrap_reflective(ecc, 1e-3, .5)]
        return x_near
    
    def wf_ecc_gen_td(self, prms, dt, f_low=20, apx="IMRPhenomXPHM", **kwargs):
        """
        Generate eccentric waveforms
        
        """

        m1, m2, ecc = prms
        waveform_params = {
            'approximant': 'IMRPhenomXPHM',
            'mass_1': m1,
            'mass_2': m2,
            'spin1x': 0,
            'spin1y': 0,
            'spin1z': 0,
            'spin2x': 0,
            'spin2y': 0,
            'spin2z': 0,
            'luminosity_distance': 100,
            'inclination': kwargs['inclination'],
            'coa_phase': kwargs['coa_phase'],
            'f_lower': f_low,
            'f_ref': kwargs['f_ref'],
            'delta_t': dt,
            'ecc': ecc
        }

        pars = ecc_gen.teobresums_pars_update(waveform_params)
        h = ecc_gen.teobresums_td_pure_polarized_wf_gen(**pars)
        hp, hc = pycbc.types.TimeSeries(h['hp'], delta_t = h['hp'].delta_t), pycbc.types.TimeSeries(h['hc'], delta_t = h['hc'].delta_t)

        return hp, hc
    
    def objective_func_EC_3D(self, x, *args):
        """
        Objective function for the maximisation/minimsation.

        """

        x[0], x[1] = self.dom_chirp(x[0]), self.dom_q(x[1])
        m1, m2 = mchirp_q_to_m1m2(x[0], x[1])
        if (m1 < 3.5 or m2 < 3.5 or m1 > 80 or m2 > 80):
            return 1e4
        
        x[2] = self.wrap_reflective(x[2], 1e-3, .5)

        gen_prms = m1, m2, x[2]

        signal, f_low, f_high, apx, kwargs = args
        dt_lw = signal.delta_t

        try:
            ec_template = self.wf_ecc_gen_td(gen_prms, dt = dt_lw, f_low=f_low, apx=apx, **kwargs)[0]
            ec_template.start_time = signal.start_time
            return np.log(1-match_wfs_td(signal, ec_template, f_low=f_low, f_high=f_high, psd=None)[0])
        except ValueError:
            return 1e4
        
class FF_EC_5D_aligned_spin(FF_EC_3D_zero_spin):   
    """
    Functions for `wf_model` == 'EC_5D'.

    """
    
    def gen_seed_prm_EC_5D(self, chirp_mass=25, q=1, chi_1=0, chi_2=0, ecc=0.1, ecc_min=1e-3, ecc_max=.5, sigma_mchirp=0.5, sigma_q=0.2, sigma_chi=0.2):
        """
        Generates seed point in 5D [m1, m2, s_1z, s_2z, e] for match maximisation; uses reasonable initial bounds.

        """ 

        mchirp = np.random.normal(chirp_mass, sigma_mchirp, 1)[0]
        q = np.random.normal(q, sigma_q, 1)[0]
        sz1 = np.random.normal(chi_1, sigma_chi, 1)[0]
        sz2 = np.random.normal(chi_2, sigma_chi, 1)[0]
        ecc = sp.stats.loguniform.rvs(ecc_min, ecc_max, 1)
        return [self.dom_chirp(mchirp), self.dom_q(q), self.dom_sp(sz1), self.dom_sp(sz2), self.wrap_reflective(ecc, ecc_min, ecc_max)]    

    def gen_seed_near_best_fit_EC_5D(self, x, sigma_ecc = 0.5, sigma_mchirp = 0.5, sigma_q = 0.1, sigma_chi = 0.1):
        chirp_mass, q, sz1, sz2, ecc = x

        mchirp = np.random.normal(chirp_mass, sigma_mchirp, 1)[0]
        q = np.random.normal(q, sigma_q, 1)[0]
        sz1 = np.random.normal(sz1, sigma_chi, 1)[0]
        sz2 = np.random.normal(sz2, sigma_chi, 1)[0]
        ecc = np.random.normal(ecc, sigma_ecc, 1)[0]
        x_near = [self.dom_chirp(mchirp), self.dom_q(q), self.dom_sp(sz1), self.dom_sp(sz2), self.wrap_reflective(ecc, 1e-3, .5)]
        return x_near

    def wf_ecc_gen_td(self, prms, dt, f_low=20, apx="IMRPhenomXPHM", **kwargs): 
        """
        Generate eccentric waveforms

        """

        m1, m2, sz1, sz2, ecc = prms
        spin1x, spin1y, sz1 = self.dom_sp([kwargs['spin1x'], kwargs['spin1y'], sz1])
        spin2x, spin2y, sz2 = self.dom_sp([kwargs['spin2x'], kwargs['spin2y'], sz2])   

        waveform_params = {
            'approximant': 'IMRPhenomXPHM',
            'mass_1': m1,
            'mass_2': m2,
            'spin1x': spin1x,
            'spin1y': spin1y,
            'spin1z': sz1,
            'spin2x': spin2x,
            'spin2y': spin2y,
            'spin2z': sz2,
            'luminosity_distance': 100,
            'inclination': kwargs['inclination'],
            'coa_phase': kwargs['coa_phase'],
            'f_lower': f_low,
            'f_ref': kwargs['f_ref'],
            'delta_t': dt,
            'ecc': ecc
        }

        pars = ecc_gen.teobresums_pars_update(waveform_params)
        h = ecc_gen.teobresums_td_pure_polarized_wf_gen(**pars)
        hp, hc = pycbc.types.TimeSeries(h['hp'], delta_t = h['hp'].delta_t), pycbc.types.TimeSeries(h['hc'], delta_t = h['hc'].delta_t)

        return hp, hc
        
    def objective_func_EC_5D(self, x, *args):
        """
        Objective function for the maximisation/minimsation.

        """

        x[0], x[1] = self.dom_chirp(x[0]), self.dom_q(x[1])
        m1, m2 = mchirp_q_to_m1m2(x[0], x[1])
        if (m1 < 3.5 or m2 < 3.5 or m1 > 80 or m2 > 80):  # if mchirp and q doesn't lead to reasonable binary masses, they will be avoided
            return 1e4

        x[2], x[3] = self.dom_sp(x[2]), self.dom_sp(x[3])
        x[4] = self.wrap_reflective(x[4], 1e-3, .5)

        gen_prms = m1, m2, x[2], x[3], x[4]

        signal, f_low, f_high, apx, kwargs = args
        dt_lw = signal.delta_t

        try:
            ec_template = self.wf_ecc_gen_td(gen_prms, dt = dt_lw, f_low=f_low, apx=apx, **kwargs)[0]
            ec_template.start_time = signal.start_time
            return np.log(1-match_wfs_td(signal, ec_template, f_low=f_low, f_high=f_high, psd=None)[0])
        except ValueError:
            return 1e4

############################################################################################################################################################################
############################################################################################################################################################################

class PointLensML:

    def __init__(self):        
        import pickle
        with open('/home/nishkal.rao/git_overlap/src/data/point_lens_Ff_lookup_table_Geo_relErr_1p0.pkl', 'rb') as f:
            self.Ff_grid = pickle.load(f)
            self.ys_grid, self.ws_grid = self.y_w_grid_data(self.Ff_grid) 
        
    ## functions
    def y_w_grid_data(self, Ff_grid):
        ys_grid = np.array([Ff_grid[str(i)]['y'] for i in range(len(Ff_grid))])
        ws_grid = Ff_grid['0']['ws']
        return ys_grid, ws_grid

    def y_index(self, yl, ys_grid):
        return np.argmin(np.abs(ys_grid - yl))

    def w_index(self, w, ws_grid):
        return np.argmin(np.abs(ws_grid - w))

    def pnt_Ff_lookup_table(self, fs, Mlz, yl, ys_grid=None, ws_grid=None, extrapolate=True):
        ys_grid, ws_grid = self.ys_grid, self.ws_grid
        wfs = np.array([pnt_lens_cy.w_of_f(f, Mlz) for f in fs])
        wc = pnt_lens_cy.wc_geo_re1p0(yl)

        wfs_1 = wfs[wfs <= np.min(ws_grid)]
        Ffs_1 = np.array([1]*len(wfs_1))

        wfs_2 = wfs[(wfs > np.min(ws_grid))&(wfs <= np.max(ws_grid))]
        wfs_2_wave = wfs_2[wfs_2 <= wc]
        wfs_2_geo = wfs_2[wfs_2 > wc]

        i_y  = self.y_index(yl, ys_grid)
        tmp_Ff_dict = self.Ff_grid[str(i_y)]
        ws = tmp_Ff_dict['ws']
        Ffs = tmp_Ff_dict['Ffs_real'] + 1j*tmp_Ff_dict['Ffs_imag']
        fill_val = ['interpolate', 'extrapolate'][extrapolate]
        i_Ff = interp1d(ws, Ffs, fill_value=fill_val)
        Ffs_2_wave = i_Ff(wfs_2_wave)

        Ffs_2_geo = np.array([pnt_lens_cy.point_Fw_geo(w, yl) for w in wfs_2_geo])

        wfs_3 = wfs[wfs > np.max(ws_grid) ]
        Ffs_3 = np.array([pnt_lens_cy.point_Fw_geo(w, Mlz) for w in wfs_3])

        Ffs = np.concatenate((Ffs_1, Ffs_2_wave, Ffs_2_geo, Ffs_3))
        assert len(Ffs)==len(fs), 'len(Ffs) = {} does not match len(fs) = {}'.format(len(Ffs), len(fs))
        return Ffs


class FF_ML_4D_zero_spin(PointLensML, FF_2D_zero_spin):
    """
    Functions for `wf_model` == 'ML_4D'.

    """
    
    def gen_seed_prm_ML_4D(self, chirp_mass=25, q=1, Mlz=500, y_lens=1, sigma_Mlz=10, sigma_y=0.5, sigma_mchirp=0.5, sigma_q=0.2):
        """
        Generates seed point in 4D [m1, m2, Mlz, y_lens] for match maximisation; uses reasonable initial bounds.

        """ 

        mchirp = np.random.normal(chirp_mass, sigma_mchirp, 1)[0]
        q = np.random.normal(q, sigma_q, 1)[0]
        Mlz = np.random.normal(Mlz, sigma_Mlz, 1)[0]
        y_lens = np.random.normal(y_lens, sigma_y, 1)[0]
        return [self.dom_chirp(mchirp), self.dom_q(q), self.wrap_reflective(Mlz, 10, 1e4), self.wrap_reflective(y_lens, 0.01, 3)]
    
    def gen_seed_near_best_fit_ML_4D(self, x, sigma_Mlz=10, sigma_y=0.5, sigma_mchirp = 0.5, sigma_q = 0.1):
        chirp_mass, q, Mlz, y_lens = x

        mchirp = np.random.normal(chirp_mass, sigma_mchirp, 1)[0]
        q = np.random.normal(q, sigma_q, 1)[0]
        Mlz = np.random.normal(Mlz, sigma_Mlz, 1)[0]
        y_lens = np.random.normal(y_lens, sigma_y, 1)[0]
        x_near = [self.dom_chirp(mchirp), self.dom_q(q), self.wrap_reflective(Mlz, 10, 1e4), self.wrap_reflective(y_lens, 0.01, 3)]
        return x_near
    
    def wf_ml_gen_fd(self, prms, df, f_low=20, apx="IMRPhenomXPHM", **kwargs):
        """
        Generates lensed Wf.

        """

        m1, m2, Mlz, yl = prms
        tmp_prms = m1, m2
        fd_hp, fd_hc = self.wf_gen_fd(tmp_prms, df, apx=apx, f_low=f_low, **kwargs)

        # Adding Microlensing effects
        if round(Mlz) == 0:
            return fd_hp, fd_hc
        else:
            fs = fd_hp.sample_frequencies
            wfs = np.array([pnt_lens_cy.w_of_f(f, Mlz) for f in fs])
            
            Ff = self.pnt_Ff_lookup_table(fs=fs, Mlz=Mlz, yl=yl)
            lfd_hp = Ff*fd_hp
            lfd_hp = pycbc.types.FrequencySeries(lfd_hp, delta_f = df)

            lfd_hc = Ff*fd_hc
            lfd_hc = pycbc.types.FrequencySeries(lfd_hc, delta_f = df)
            return lfd_hp, lfd_hc
        
    def objective_func_ML_4D(self, x, *args):
        """
        Objective function for the maximisation/minimsation.

        """

        x[0], x[1] = self.dom_chirp(x[0]), self.dom_q(x[1])
        m1, m2 = mchirp_q_to_m1m2(x[0], x[1])
        if (m1 < 3.5 or m2 < 3.5):
            return 1e4
        
        x[2], x[3] = self.wrap_reflective(x[2], 10, 1e4), self.wrap_reflective(x[3], 0.01, 3)

        gen_prms = m1, m2, x[2], x[3]

        signal, f_low, f_high, apx, kwargs = args
        df_lw = signal.delta_f

        try:
            ml_template = self.wf_ml_gen_fd(gen_prms, df = df_lw, f_low=f_low, apx=apx, **kwargs)[0]
            ml_template.start_time = signal.start_time
            return np.log(1-match_wfs_fd(signal, ml_template, f_low=f_low, f_high=f_high, psd=None)[0])
        except ValueError:
            return 1e4

    
class FF_ML_6D_aligned_spin(PointLensML, FF_4D_aligned_spin):   
    """
    Functions for `wf_model` == 'ML_6D'.

    """
    
    def gen_seed_prm_ML_6D(self, chirp_mass=25, q=1, chi_1=0, chi_2=0, Mlz=500, y_lens=1, 
                           sigma_Mlz=10, sigma_y=0.5, sigma_mchirp=0.5, sigma_q=0.2, sigma_chi=0.2):
        """
        Generates seed point in 6D [m1, m2, s_1z, s_2z, Mlz, y_lens] for match maximisation; uses reasonable initial bounds.

        """ 

        mchirp = np.random.normal(chirp_mass, sigma_mchirp, 1)[0]
        q = np.random.normal(q, sigma_q, 1)[0]
        sz1 = np.random.normal(chi_1, sigma_chi, 1)[0]
        sz2 = np.random.normal(chi_2, sigma_chi, 1)[0]
        Mlz = np.random.normal(Mlz, sigma_Mlz, 1)[0]
        y_lens = np.random.normal(y_lens, sigma_y, 1)[0]
        return [self.dom_chirp(mchirp), self.dom_q(q), self.dom_sp(sz1), self.dom_sp(sz2),\
                self.wrap_reflective(Mlz, 10, 1e4), self.wrap_reflective(y_lens, 0.01, 3)]    

    def gen_seed_near_best_fit_ML_6D(self, x, sigma_Mlz=10, sigma_y=0.5, sigma_mchirp = 0.5, sigma_q = 0.1, sigma_chi = 0.1):
        chirp_mass, q, sz1, sz2, Mlz, y_lens = x

        mchirp = np.random.normal(chirp_mass, sigma_mchirp, 1)[0]
        q = np.random.normal(q, sigma_q, 1)[0]
        sz1 = np.random.normal(sz1, sigma_chi, 1)[0]
        sz2 = np.random.normal(sz2, sigma_chi, 1)[0]
        Mlz = np.random.normal(Mlz, sigma_Mlz, 1)[0]
        y_lens = np.random.normal(y_lens, sigma_y, 1)[0]
        x_near = [self.dom_chirp(mchirp), self.dom_q(q), self.dom_sp(sz1), self.dom_sp(sz2),\
                 self.wrap_reflective(Mlz, 10, 1e4), self.wrap_reflective(y_lens, 0.01, 3)]
        return x_near
        
    def wf_ml_gen_fd(self, prms, df, f_low=20, apx="IMRPhenomXPHM", **kwargs): 

        m1, m2, sz1, sz2, Mlz, yl = prms
        tmp_prms = m1, m2, sz1, sz2
        fd_hp, fd_hc  = self.wf_gen_fd(tmp_prms, df, apx=apx, f_low=f_low, **kwargs)

        # Adding Microlensing effects
        if round(Mlz) == 0:
            return fd_hp, fd_hc
        else:
            fs = fd_hp.sample_frequencies
            wfs = np.array([pnt_lens_cy.w_of_f(f, Mlz) for f in fs])
            
            Ff = self.pnt_Ff_lookup_table(fs=fs, Mlz=Mlz, yl=yl)
            lfd_hp = Ff*fd_hp
            lfd_hp = pycbc.types.FrequencySeries(lfd_hp, delta_f = df)

            lfd_hc = Ff*fd_hc
            lfd_hc = pycbc.types.FrequencySeries(lfd_hc, delta_f = df)
            return lfd_hp, lfd_hc     
        
        
    def objective_func_ML_6D(self, x, *args):
        """
        Objective function for the maximisation/minimsation.

        """

        x[0], x[1] = self.dom_chirp(x[0]), self.dom_q(x[1])
        m1, m2 = mchirp_q_to_m1m2(x[0], x[1])
        if (m1 < 3.5 or m2 < 3.5):  # if mchirp and q doesn't lead to reasonable binary masses, they will be avoided
            return 1e4

        x[2], x[3] = self.dom_sp(x[2]), self.dom_sp(x[3])
        x[4], x[5] = self.wrap_reflective(x[4], 10, 1e4), self.wrap_reflective(x[5], 0.01, 3)

        gen_prms = m1, m2, x[2], x[3], x[4], x[5]

        signal, f_low, f_high, apx, kwargs = args
        df_lw = signal.delta_f
    
        try:
            ml_template = self.wf_ml_gen_fd(gen_prms, df = df_lw, f_low=f_low, apx=apx, **kwargs)[0]
            ml_template.start_time = signal.start_time
            return np.log(1-match_wfs_fd(signal, ml_template, f_low=f_low, f_high=f_high, psd=None)[0])
        except ValueError:
            return 1e4

############################################################################################################################################################################
############################################################################################################################################################################

def compute_fitting_factor(signal, wf_model = '2D', apx="IMRPhenomXPHM", psd=None, f_low=20, f_high=None, n_iters=['default'], **kwargs):
    """
    Function to compute the fitting factor (max. match) over unlensed template bank using Nelder-Mead minimation algorithm. 
    The PSD used is aLIGOZeroDetHighPower (almost equivalent to O4). Option to change PSD will come in future versions of the code.

    Parameters
    ----------
    signal : pycbc.types.TimeSeries
        WF for which FF will be evaluated.
    wf_model : {'2D', '4D', 'EC_3D', 'EC_5D', 'ML_4D', 'ML_6D'}, optional
        WF model to use for recovery. Default = '2D'.
        * '2D' represents 2D zero spin recovery in {chirp_mass, mass_ratio}.
        * '4D' represents 4D aligned spin recovery in {chirp_mass, mass_ratio, spin1z, spin2z}.
        * 'EC_3D' represents 3D eccentric zero spin recovery in {chirp_mass, mass_ratio, eccentricity}.
        * 'EC_5D' represents 5D eccentric aligned spin recovery in {eccentricity, chirp_mass, mass_ratio, spin1z, spin2z}.
        * 'ML_4D' represents 4D microlensed zero spin recovery in {chirp_mass, mass_ratio, Mlz, y_lens}.        
        * 'ML_6D' represents 6D microlensed aligned spin recovery in {redshifted_lens_mass, impact_parameter, chirp_mass, mass_ratio, spin1z, spin2z}.

    apx : str, optional
        Name of LAL WF approximant to use for recovery. Default="IMRPhenomXPHM".
    psd : str, optional
        Path to PSD file. Default = None.
    f_low : ({20., float}), optional 
        Starting frequency for waveform generation (in Hz). 
    f_high : ({None., float}), optional 
        Maximum frequency for matched filter computations (in Hz). 
    n_iters : {'default', int}, optional
        Number of iterations with different seed points for FF computation. Default = ['default'] which is equivalent to 10].
    
    Returns
    -------
    numpy.array, 
        An array containing the combined result from all the cores in the form [FF, best_matched_WF_parameters] for each iteration.
        The array is sorted in descending order, so the best match value will be the 0th element in the array.

    """    
    
    keyword_args = dict(Mtot=np.random.uniform(7, 200), q=0.05 + (1 - 0.05)*np.random.power(2, 1), 
                        spin1z=0, spin2z=0,
                        spin1x=0, spin1y=0, spin2x=0, spin2y=0, 
                        Mlz=10**np.random.uniform(1, 4), y_lens=0.01 + (3 - 0.01)*np.random.power(2, 1), 
                        ecc=sp.stats.loguniform.rvs(1e-3, .5, 1),
                        coa_phase = 0, inclination = 0, f_ref = f_low,
                        sigma_mchirp=1, sigma_q=0.2, sigma_chi=0.2,
                        max_wait_per_iter=None, default_value=0)
    keyword_args.update(chirp_mass=mtot_q_to_mchirp(keyword_args['Mtot'], keyword_args['q']))
    keyword_args.update(kwargs)
    
    if wf_model == '2D':
        FF_UL = FF_2D_zero_spin()
        gen_prms = FF_UL.gen_seed_prm_2D
        gen_seed_near_best_fit = FF_UL.gen_seed_near_best_fit_2D
        objective_func = FF_UL.objective_func_2D
    elif wf_model == '4D':
        FF_UL = FF_4D_aligned_spin()
        gen_prms = FF_UL.gen_seed_prm_4D
        gen_seed_near_best_fit = FF_UL.gen_seed_near_best_fit_4D
        objective_func = FF_UL.objective_func_4D
    elif wf_model == 'EC_3D':
        FF_EC = FF_EC_3D_zero_spin()
        gen_prms = FF_EC.gen_seed_prm_EC_3D
        gen_seed_near_best_fit = FF_EC.gen_seed_near_best_fit_EC_3D
        objective_func = FF_EC.objective_func_EC_3D
    elif wf_model == 'EC_5D':
        FF_EC = FF_EC_5D_aligned_spin()
        gen_prms = FF_EC.gen_seed_prm_EC_5D
        gen_seed_near_best_fit = FF_EC.gen_seed_near_best_fit_EC_5D
        objective_func = FF_EC.objective_func_EC_5D
    elif wf_model == 'ML_4D':
        FF_ML = FF_ML_4D_zero_spin()
        gen_prms = FF_ML.gen_seed_prm_ML_4D
        gen_seed_near_best_fit = FF_ML.gen_seed_near_best_fit_ML_4D
        objective_func = FF_ML.objective_func_ML_4D       
    elif wf_model == 'ML_6D':
        FF_ML = FF_ML_6D_aligned_spin()
        gen_prms = FF_ML.gen_seed_prm_ML_6D
        gen_seed_near_best_fit = FF_ML.gen_seed_near_best_fit_ML_6D
        objective_func = FF_ML.objective_func_ML_6D         
    else:
        raise Exception("Allowed values for the keyword argument 'wf_model' = ['2D', '4D', 'EC_3D', EC_5D, 'ML_4D', 'ML_6D']. But 'wf_model = %s' provided instead."%(wf_model) )
    
    res_prms = []

    Mtot, q, sz_1, sz_2 = keyword_args['Mtot'], keyword_args['q'], keyword_args['spin1z'], keyword_args['spin2z']
    comp_masses = mtot_q_to_m1m2(Mtot, q)
    Mlz, y_lens, ecc = keyword_args['Mlz'], keyword_args['y_lens'], keyword_args['ecc'] 
    sigma_mchirp, sigma_q, sigma_chi = keyword_args['sigma_mchirp'], keyword_args['sigma_q'], keyword_args['sigma_chi']
    
    if n_iters == ['default']:
        n_iters = 10

    for i in range(n_iters):
        if wf_model == '2D':
            tmp_best_val = gen_prms(chirp_mass=m1m2_to_mchirp(comp_masses[0], comp_masses[1]), q=q, 
                                    sigma_mchirp=sigma_mchirp, sigma_q=sigma_q)
        elif wf_model == '4D':
            tmp_best_val = gen_prms(chirp_mass=m1m2_to_mchirp(comp_masses[0], comp_masses[1]), q=q, chi_1=sz_1, chi_2=sz_2,
                                    sigma_mchirp=sigma_mchirp, sigma_q=sigma_q, sigma_chi=sigma_chi)
        elif wf_model == 'EC_3D':
            tmp_best_val = gen_prms(chirp_mass=m1m2_to_mchirp(comp_masses[0], comp_masses[1]), q=q,
                                    sigma_mchirp=sigma_mchirp, sigma_q=sigma_q, ecc=ecc)
        elif wf_model == 'EC_5D':
            tmp_best_val = gen_prms(chirp_mass=m1m2_to_mchirp(comp_masses[0], comp_masses[1]), q=q, chi_1=sz_1, chi_2=sz_2,
                                    sigma_mchirp=sigma_mchirp, sigma_q=sigma_q, sigma_chi=sigma_chi, ecc=ecc)
        elif wf_model == 'ML_4D':
            tmp_best_val = gen_prms(chirp_mass=m1m2_to_mchirp(comp_masses[0], comp_masses[1]), q=q, 
                                    sigma_mchirp=sigma_mchirp, sigma_q=sigma_q, Mlz=Mlz, y_lens=y_lens)
        elif wf_model == 'ML_6D':
            tmp_best_val = gen_prms(chirp_mass=m1m2_to_mchirp(comp_masses[0], comp_masses[1]), q=q, chi_1=sz_1, chi_2=sz_2,
                                    sigma_mchirp=sigma_mchirp, sigma_q=sigma_q, sigma_chi=sigma_chi, Mlz=Mlz, y_lens=y_lens)
        signal_copy = deepcopy(signal)
        
        res = minimize(objective_func, tmp_best_val, args=(signal_copy, f_low, f_high, apx, keyword_args), method='Nelder-Mead', 
                        options={'disp':True, 'adaptive':True, 'xatol':1e-4})
        
        FF_val = 1-np.power(10, objective_func(res.x, signal_copy, f_low, f_high, apx, keyword_args))
        res_prms.append([FF_val, list(res.x)])

    result = min(res_prms, key=lambda x: x[0])
    
    return result