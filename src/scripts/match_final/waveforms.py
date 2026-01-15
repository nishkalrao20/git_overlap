import lal
import pycbc
import pycbc.waveform
import pycbc.detector
import pycbc.types
import lalsimulation
import numpy as np
import scipy.stats
import gwmat 
import gwmat.cythonized_point_lens as pnt_lens_cy
import gweat.teobresums.utils as ecc_gen

class PairsWaveformGeneration:
    """
    Generating PAIRS Waveforms.
    Includes Singles, Pairs, Eccentric (TEOBResumS), and Microlensed models.
    """

    def __init__(self, **kwargs):
        self.duration = kwargs.get('duration', 4.0)
        self.sampling_frequency = kwargs.get('sampling_frequency', 4096)
        self.f_lower = kwargs.get('f_lower', 20.0)
        self.f_ref = kwargs.get('f_ref', 50.0)
        self.f_high = kwargs.get('f_high', 1024.0)
        self.delta_f = kwargs.get('delta_f', 1.0 / self.duration)
        self.delta_t = kwargs.get('delta_t', 1.0 / self.sampling_frequency)
        
        self.Ff_grid = None
        self.ys_grid = None
        self.ws_grid = None

    def wf_len_mod_start(self, wf, extra=1, **prms):
        """
        Taken from GWMAT. Function to modify the starting of a WF so that it starts on an integer GPS time (in sec).
        """      
        olen = len(wf)   
        diff = wf.sample_times[0]-np.floor(wf.sample_times[0])  
        dlen = round(self.sampling_frequency*(extra+diff))
        wf_strain = np.concatenate((np.zeros(dlen), wf))
        t0 = wf.sample_times[0]
        dt = wf.delta_t
        n = dlen
        tnn = t0-(n+1)*dt
        wf_stime = np.concatenate((np.arange(t0-dt,tnn,-dt)[::-1], np.array(wf.sample_times)))
        nwf = pycbc.types.TimeSeries(wf_strain, delta_t=wf.delta_t, epoch=wf_stime[0])
        return nwf

    def wf_len_mod_end(self, wf, extra=2, **prms): 
        """
        Taken from GWMAT. Function to modify the end of a WF so that it ends on an integer GPS time (in sec).
        """        
        olen = len(wf)   
        dt = abs(wf.sample_times[-1] - wf.sample_times[-2])
        diff = np.ceil(wf.sample_times[-1]) - (wf.sample_times[-1] + dt)   
        nlen = round(olen + self.sampling_frequency*(extra+diff))
        wf.resize(nlen)
        return wf    

    def make_len_power_of_2(self, wf):
        """
        Taken from GWMAT. Function to modify the length of a waveform so that its duration is a power of 2.
        """    
        dur = wf.duration  
        wf.resize( int(round(wf.sample_rate * np.power(2, np.ceil( np.log2( dur ) ) ))) )
        wf = self.cyclic_time_shift_of_WF(wf, rwrap = wf.duration - dur )
        return wf

    def cyclic_time_shift_of_WF(self, wf, rwrap=0.2):
        """
        Taken from GWMAT. Cyclic time shift preserving sample rate.
        """        
        if rwrap is not None and rwrap != 0:
            sn = abs(int(rwrap/wf.delta_t))
            cycles = int(sn/len(wf))

            cyclic_shifted_wf = wf.copy()
            sn_new = sn - int(cycles * len(wf))

            if rwrap > 0:
                epoch = wf.sample_times[0] - sn_new * wf.delta_t
                if sn_new != 0:
                    wf_arr = np.array(wf).copy()
                    tmp_wf_p1 = wf_arr[-sn_new:]
                    tmp_wf_p2 = wf_arr[:-sn_new] 
                    shft_wf_arr = np.concatenate(( tmp_wf_p1, tmp_wf_p2 ))
                    cyclic_shifted_wf = pycbc.types.TimeSeries(shft_wf_arr, delta_t = wf.delta_t, epoch = epoch)
            else:
                epoch = wf.sample_times[sn_new]
                if sn_new != 0:
                    wf_arr = np.array(wf).copy()
                    tmp_wf_p1 = wf_arr[sn_new:] 
                    tmp_wf_p2 = wf_arr[:sn_new]
                    shft_wf_arr = np.concatenate(( tmp_wf_p1, tmp_wf_p2 ))
                    cyclic_shifted_wf = pycbc.types.TimeSeries(shft_wf_arr, delta_t = wf.delta_t, epoch = epoch)  

            for i in range(cycles):        
                    epoch = epoch - np.sign(rwrap)*wf.duration
                    wf_arr = np.array(cyclic_shifted_wf)[:]
                    cyclic_shifted_wf = pycbc.types.TimeSeries(wf_arr, delta_t = wf.delta_t, epoch = epoch)

            assert len(cyclic_shifted_wf) == len(wf), 'Length mismatch: cyclic time shift added extra length to WF.'
            return cyclic_shifted_wf
        else:
            return wf  

    def jframe_to_l0frame(self, mass_1, mass_2, f_ref, phi_ref=0., theta_jn=0., phi_jl=0., a_1=0., a_2=0., tilt_1=0., tilt_2=0., phi_12=0., **kwargs):
        """
        Helper to convert J-frame to L0-frame.
        """
        inclination, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = \
            lalsimulation.SimInspiralTransformPrecessingNewInitialConditions(
                theta_jn, phi_jl, tilt_1, tilt_2, phi_12,
                a_1, a_2, mass_1*lal.MSUN_SI, mass_2*lal.MSUN_SI, f_ref,
                phi_ref)
        return {'inclination': inclination,
                'spin1x': spin1x, 'spin1y': spin1y, 'spin1z': spin1z,
                'spin2x': spin2x, 'spin2y': spin2y, 'spin2z': spin2z}

    def wf_td(self, injection_parameters, **kwargs):
        """
        Generate PyCBC time domain SINGLES waveforms.
        """
        lframe = self.jframe_to_l0frame(
            mass_1=injection_parameters['mass_1'], 
            mass_2=injection_parameters['mass_2'], 
            f_ref=self.f_ref, 
            theta_jn=injection_parameters['theta_jn'], 
            phi_jl=injection_parameters['phi_jl'], 
            a_1=injection_parameters['a_1'], 
            a_2=injection_parameters['a_2'], 
            tilt_1=injection_parameters['tilt_1'], 
            tilt_2=injection_parameters['tilt_2'], 
            phi_12=injection_parameters['phi_12']
        )    

        waveform_params = {
            'approximant': 'IMRPhenomXPHM',
            'mass1': injection_parameters['mass_1'],
            'mass2': injection_parameters['mass_2'],
            'spin1x': lframe['spin1x'], 'spin1y': lframe['spin1y'], 'spin1z': lframe['spin1z'],
            'spin2x': lframe['spin2x'], 'spin2y': lframe['spin2y'], 'spin2z': lframe['spin2z'],
            'distance': injection_parameters['luminosity_distance'],
            'inclination': lframe['inclination'],
            'coa_phase': injection_parameters['phase'],
            'f_lower': self.f_lower,
            'f_ref': self.f_ref,
            'delta_t': self.delta_t
        }

        hp, hc = pycbc.waveform.get_td_waveform(**waveform_params)
        hp.start_time += injection_parameters['geocent_time']
        hc.start_time += injection_parameters['geocent_time']

        from pycbc.waveform.utils import taper_timeseries
        det, ifo_signal = dict(), dict()
        for ifo in ['H1', 'L1', 'V1']:
            det[ifo] = pycbc.detector.Detector(ifo)
            ifo_signal[ifo] = det[ifo].project_wave(hp, hc, injection_parameters['ra'], injection_parameters['dec'], injection_parameters['psi'])
            ifo_signal[ifo] = taper_timeseries(ifo_signal[ifo], tapermethod='TAPER_STARTEND', return_lal=False)
            ifo_signal[ifo] = self.make_len_power_of_2(self.wf_len_mod_end(self.wf_len_mod_start(ifo_signal[ifo])))

        return {'H1': ifo_signal['H1'], 'L1': ifo_signal['L1'], 'V1': ifo_signal['V1']}

    def pairs_td(self, injection_parameters_a, injection_parameters_b, **kwargs):
        """
        Generate PyCBC time domain PAIRS waveforms.
        """
        
        lframe_a = self.jframe_to_l0frame(
            mass_1=injection_parameters_a['mass_1'], mass_2=injection_parameters_a['mass_2'], f_ref=self.f_ref, 
            theta_jn=injection_parameters_a['theta_jn'], phi_jl=injection_parameters_a['phi_jl'], 
            a_1=injection_parameters_a['a_1'], a_2=injection_parameters_a['a_2'], 
            tilt_1=injection_parameters_a['tilt_1'], tilt_2=injection_parameters_a['tilt_2'], phi_12=injection_parameters_a['phi_12']
        )   
        waveform_params_a = {
            'approximant': 'IMRPhenomXPHM',
            'mass1': injection_parameters_a['mass_1'], 'mass2': injection_parameters_a['mass_2'],
            'spin1x': lframe_a['spin1x'], 'spin1y': lframe_a['spin1y'], 'spin1z': lframe_a['spin1z'],
            'spin2x': lframe_a['spin2x'], 'spin2y': lframe_a['spin2y'], 'spin2z': lframe_a['spin2z'],
            'distance': injection_parameters_a['luminosity_distance'], 'inclination': lframe_a['inclination'],
            'coa_phase': injection_parameters_a['phase'], 'f_lower': self.f_lower, 'f_ref': self.f_ref, 'delta_t': self.delta_t
        }

        lframe_b = self.jframe_to_l0frame(
            mass_1=injection_parameters_b['mass_1'], mass_2=injection_parameters_b['mass_2'], f_ref=self.f_ref, 
            theta_jn=injection_parameters_b['theta_jn'], phi_jl=injection_parameters_b['phi_jl'], 
            a_1=injection_parameters_b['a_1'], a_2=injection_parameters_b['a_2'], 
            tilt_1=injection_parameters_b['tilt_1'], tilt_2=injection_parameters_b['tilt_2'], phi_12=injection_parameters_b['phi_12']
        )
        waveform_params_b = {
            'approximant': 'IMRPhenomXPHM',
            'mass1': injection_parameters_b['mass_1'], 'mass2': injection_parameters_b['mass_2'],
            'spin1x': lframe_b['spin1x'], 'spin1y': lframe_b['spin1y'], 'spin1z': lframe_b['spin1z'],
            'spin2x': lframe_b['spin2x'], 'spin2y': lframe_b['spin2y'], 'spin2z': lframe_b['spin2z'],
            'distance': injection_parameters_b['luminosity_distance'], 'inclination': lframe_b['inclination'],
            'coa_phase': injection_parameters_b['phase'], 'f_lower': self.f_lower, 'f_ref': self.f_ref, 'delta_t': self.delta_t
        }

        from pycbc.waveform.utils import taper_timeseries
        det, ifo_signal_a, ifo_signal_b = dict(), dict(), dict()

        for ifo in ['H1', 'L1', 'V1']:
            det[ifo] = pycbc.detector.Detector(ifo)

            hp_a, hc_a = pycbc.waveform.get_td_waveform(**waveform_params_a)
            hp_a.start_time += injection_parameters_a['geocent_time']
            hc_a.start_time += injection_parameters_a['geocent_time']
            ifo_signal_a[ifo] = det[ifo].project_wave(hp_a, hc_a, injection_parameters_a['ra'], injection_parameters_a['dec'], injection_parameters_a['psi'])
            ifo_signal_a[ifo] = taper_timeseries(ifo_signal_a[ifo], tapermethod='TAPER_STARTEND', return_lal=False)
            ifo_signal_a[ifo] = self.make_len_power_of_2(self.wf_len_mod_end(self.wf_len_mod_start(ifo_signal_a[ifo])))
            
            hp_b, hc_b = pycbc.waveform.get_td_waveform(**waveform_params_b)
            hp_b.start_time += injection_parameters_b['geocent_time']
            hc_b.start_time += injection_parameters_b['geocent_time']
            ifo_signal_b[ifo] = det[ifo].project_wave(hp_b, hc_b, injection_parameters_b['ra'], injection_parameters_b['dec'], injection_parameters_b['psi'])
            ifo_signal_b[ifo] = taper_timeseries(ifo_signal_b[ifo], tapermethod='TAPER_STARTEND', return_lal=False)
            ifo_signal_b[ifo] = self.make_len_power_of_2(self.wf_len_mod_end(self.wf_len_mod_start(ifo_signal_b[ifo])))

        ht_H1_a, ht_L1_a, ht_V1_a = ifo_signal_a['H1'], ifo_signal_a['L1'], ifo_signal_a['V1']
        ht_H1_b, ht_L1_b, ht_V1_b = ifo_signal_b['H1'], ifo_signal_b['L1'], ifo_signal_b['V1']

        delta_f = np.min([ht_H1_a.delta_f, ht_L1_a.delta_f, ht_V1_a.delta_f, ht_H1_b.delta_f, ht_L1_b.delta_f, ht_V1_b.delta_f])
        hf_a_H1, hf_a_L1, hf_a_V1 = ht_H1_a.to_frequencyseries(delta_f=delta_f), ht_L1_a.to_frequencyseries(delta_f=delta_f), ht_V1_a.to_frequencyseries(delta_f=delta_f)
        hf_b_H1, hf_b_L1, hf_b_V1 = ht_H1_b.to_frequencyseries(delta_f=delta_f), ht_L1_b.to_frequencyseries(delta_f=delta_f), ht_V1_b.to_frequencyseries(delta_f=delta_f)

        hf_H1 = hf_a_H1 + hf_b_H1 * np.exp(-1j * 2 * np.pi * (injection_parameters_b['geocent_time'] - injection_parameters_a['geocent_time']) * hf_b_H1.sample_frequencies)
        hf_L1 = hf_a_L1 + hf_b_L1 * np.exp(-1j * 2 * np.pi * (injection_parameters_b['geocent_time'] - injection_parameters_a['geocent_time']) * hf_b_L1.sample_frequencies)
        hf_V1 = hf_a_V1 + hf_b_V1 * np.exp(-1j * 2 * np.pi * (injection_parameters_b['geocent_time'] - injection_parameters_a['geocent_time']) * hf_b_V1.sample_frequencies)

        return {'H1': hf_H1.to_timeseries(), 'L1': hf_L1.to_timeseries(), 'V1': hf_V1.to_timeseries()}
    
    def import_lookup(self):
        import pickle
        if self.Ff_grid is None:
            with open('/home/nishkal.rao/git_overlap/src/data/point_lens_Ff_lookup_table_Geo_relErr_1p0.pkl', 'rb') as f:
                self.Ff_grid = pickle.load(f)
                self.ys_grid, self.ws_grid = self.y_w_grid_data(self.Ff_grid) 
        
    def y_w_grid_data(self, Ff_grid):
        ys_grid = np.array([Ff_grid[str(i)]['y'] for i in range(len(Ff_grid))])
        ws_grid = Ff_grid['0']['ws']
        return ys_grid, ws_grid

    def y_index(self, yl, ys_grid):
        return np.argmin(np.abs(ys_grid - yl))

    def w_index(self, w, ws_grid):
        return np.argmin(np.abs(ws_grid - w))

    def pnt_Ff_lookup_table(self, fs, Mlz, yl, ys_grid=None, ws_grid=None, extrapolate=True): 
        from scipy.interpolate import interp1d
        
        ys_grid, ws_grid = self.ys_grid, self.ws_grid
        wfs = np.array([pnt_lens_cy.w_of_f(f, Mlz) for f in fs])
        wc = pnt_lens_cy.w_cutoff_geometric_optics_tolerance_1p0(yl, warn=False)

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

        Ffs_2_geo = np.array([pnt_lens_cy.Fw_geometric_optics(w, yl) for w in wfs_2_geo])

        wfs_3 = wfs[wfs > np.max(ws_grid) ]
        Ffs_3 = np.array([pnt_lens_cy.Fw_geometric_optics(w, Mlz) for w in wfs_3])

        Ffs = np.concatenate((Ffs_1, Ffs_2_wave, Ffs_2_geo, Ffs_3))
        assert len(Ffs)==len(fs), 'len(Ffs) = {} does not match len(fs) = {}'.format(len(Ffs), len(fs))
        return Ffs

    def wf_ml_fd(self, injection_parameters, Ml_z, y, **kwargs):
        """
        Generate microlensed time domain waveforms for the injection parameters.
        Includes conjugate correction for LAL engineering convention.
        """       
        self.import_lookup()

        ht = self.wf_td(injection_parameters)
        hf_H1, hf_L1, hf_V1 = ht['H1'].to_frequencyseries(), ht['L1'].to_frequencyseries(), ht['V1'].to_frequencyseries()

        if round(Ml_z) == 0:
            return {'H1': hf_H1, 'L1': hf_L1, 'V1': hf_V1}
        
        else:
            Ff = self.pnt_Ff_lookup_table(ys_grid=self.ys_grid, ws_grid=self.ws_grid, fs=hf_H1.sample_frequencies, Mlz=Ml_z, yl=y)
            
            Ff_eng = np.conj(Ff)

            hfl_H1 = pycbc.types.FrequencySeries(Ff_eng*hf_H1, delta_f = hf_H1.delta_f)
            hfl_L1 = pycbc.types.FrequencySeries(Ff_eng*hf_L1, delta_f = hf_L1.delta_f)
            hfl_V1 = pycbc.types.FrequencySeries(Ff_eng*hf_V1, delta_f = hf_V1.delta_f)
            
            start_t = injection_parameters['geocent_time']-self.duration+2
            hfl_H1.start_time, hfl_L1.start_time, hfl_V1.start_time = start_t, start_t, start_t
            
            return {'H1': hfl_H1, 'L1': hfl_L1, 'V1': hfl_V1}

    def pairs_ml_fd(self, injection_parameters_a, injection_parameters_b, **kwargs):
        """
        Generate frequency domain PAIRS waveforms with Microlensing.
        """
        hf_a, hf_b = self.wf_ml_fd(injection_parameters=injection_parameters_a, Ml_z=0, y=0), self.wf_ml_fd(injection_parameters=injection_parameters_b, Ml_z=0, y=0)

        ht_H1_a, ht_L1_a, ht_V1_a = hf_a['H1'].to_timeseries(), hf_a['L1'].to_timeseries(), hf_a['V1'].to_timeseries()
        ht_H1_b, ht_L1_b, ht_V1_b = hf_b['H1'].to_timeseries(), hf_b['L1'].to_timeseries(), hf_b['V1'].to_timeseries()

        delta_f = np.min([ht_H1_a.delta_f, ht_L1_a.delta_f, ht_V1_a.delta_f, ht_H1_b.delta_f, ht_L1_b.delta_f, ht_V1_b.delta_f])
        hf_a_H1, hf_a_L1, hf_a_V1 = ht_H1_a.to_frequencyseries(delta_f=delta_f), ht_L1_a.to_frequencyseries(delta_f=delta_f), ht_V1_a.to_frequencyseries(delta_f=delta_f)
        hf_b_H1, hf_b_L1, hf_b_V1 = ht_H1_b.to_frequencyseries(delta_f=delta_f), ht_L1_b.to_frequencyseries(delta_f=delta_f), ht_V1_b.to_frequencyseries(delta_f=delta_f)

        hf_H1 = hf_a_H1 + hf_b_H1 * np.exp(-1j * 2 * np.pi * (injection_parameters_b['geocent_time'] - injection_parameters_a['geocent_time']) * hf_b_H1.sample_frequencies)
        hf_L1 = hf_a_L1 + hf_b_L1 * np.exp(-1j * 2 * np.pi * (injection_parameters_b['geocent_time'] - injection_parameters_a['geocent_time']) * hf_b_L1.sample_frequencies)
        hf_V1 = hf_a_V1 + hf_b_V1 * np.exp(-1j * 2 * np.pi * (injection_parameters_b['geocent_time'] - injection_parameters_a['geocent_time']) * hf_b_V1.sample_frequencies)

        hf_H1 = pycbc.types.FrequencySeries(hf_H1, delta_f = hf_a_H1.delta_f)
        hf_L1 = pycbc.types.FrequencySeries(hf_L1, delta_f = hf_a_L1.delta_f)
        hf_V1 = pycbc.types.FrequencySeries(hf_V1, delta_f = hf_a_V1.delta_f)
        
        return {'H1': hf_H1, 'L1': hf_L1, 'V1': hf_V1}

    def determine_time_shift(self, wf):
        """
        Computes time shift to align the peak of the waveform with its end time (t=0 relative in shifted frame).
        """
        wf_end_time = float(wf.end_time)
        peak_time = wf.sample_times[np.argmax(abs(np.asarray(wf)))]
        t_shift = wf_end_time - peak_time
        return t_shift

    def wf_ecc_td(self, injection_parameters, e, anomaly=0, **kwargs):
        """
        Generate PyCBC time domain eccentric waveforms (TEOBResumS).
        Includes robust peak alignment and tolerance settings.
        """
        
        lframe = self.jframe_to_l0frame(
            mass_1=injection_parameters['mass_1'], 
            mass_2=injection_parameters['mass_2'], 
            f_ref=self.f_ref, 
            theta_jn=injection_parameters['theta_jn'], 
            phi_jl=injection_parameters['phi_jl'], 
            a_1=injection_parameters['a_1'], 
            a_2=injection_parameters['a_2'], 
            tilt_1=injection_parameters['tilt_1'], 
            tilt_2=injection_parameters['tilt_2'], 
            phi_12=injection_parameters['phi_12']
        )    

        waveform_params = {
            'mass_1': injection_parameters['mass_1'],
            'mass_2': injection_parameters['mass_2'],
            'spin1x': lframe['spin1x'], 'spin1y': lframe['spin1y'], 'spin1z': lframe['spin1z'],
            'spin2x': lframe['spin2x'], 'spin2y': lframe['spin2y'], 'spin2z': lframe['spin2z'],
            'luminosity_distance': injection_parameters['luminosity_distance'],
            'inclination': lframe['inclination'],
            'coa_phase': injection_parameters['phase'],
            'f_lower': self.f_lower,
            'f_ref': self.f_ref,
            'delta_t': self.delta_t,
            'ecc': e,
            'anomaly': anomaly,
            "mode_array": [[2, 1], [2, 2], [3, 2], [3, 3], [4, 4]],
            "ecc_freq": 0,
            "initial_frequency": 10,
            "ode_abstol": 1e-8,
            "ode_reltol": 1e-7,
            "min_td_duration": 4.0,
        }
        
        waveform_params.update(kwargs)

        pars = ecc_gen.teobresums_pars_update(waveform_params)
        h = ecc_gen.teobresums_td_pure_polarized_wf_gen(**pars)
        hp_raw, hc_raw = h['hp'], h['hc']
        
        target_duration = max(waveform_params["min_td_duration"], hp_raw.duration)
        mod_wf_duration = target_duration * np.ceil(hp_raw.duration / target_duration) if target_duration > 0 else hp_raw.duration
        
        try:
             hp_raw = gwmat.injection.modify_signal_start_time(hp_raw, extra=mod_wf_duration - hp_raw.duration)
             hc_raw = gwmat.injection.modify_signal_start_time(hc_raw, extra=mod_wf_duration - hc_raw.duration)
        except (AttributeError, NameError):
             pass

        wf_complex = hp_raw - 1j * hc_raw
        t_shift = self.determine_time_shift(wf_complex)
        wf_shifted = self.cyclic_time_shift_of_WF(wf_complex, rwrap=t_shift)
        wf_shifted.start_time = 0
        
        hp = pycbc.types.TimeSeries(wf_shifted.real(), delta_t=self.delta_t)
        hc = pycbc.types.TimeSeries(-1 * wf_shifted.imag(), delta_t=self.delta_t)
        
        hp.start_time += injection_parameters['geocent_time']
        hc.start_time += injection_parameters['geocent_time']
            
        from pycbc.waveform.utils import taper_timeseries

        det, ifo_signal = dict(), dict()
        for ifo in ['H1', 'L1', 'V1']:
            det[ifo] = pycbc.detector.Detector(ifo)
            ifo_signal[ifo] = det[ifo].project_wave(hp, hc, injection_parameters['ra'], injection_parameters['dec'], injection_parameters['psi'])
            ifo_signal[ifo] = taper_timeseries(ifo_signal[ifo], tapermethod='TAPER_STARTEND', return_lal=False)
            ifo_signal[ifo] = self.make_len_power_of_2(self.wf_len_mod_end(self.wf_len_mod_start(ifo_signal[ifo])))

        return {'H1': ifo_signal['H1'], 'L1': ifo_signal['L1'], 'V1': ifo_signal['V1']}

    def pairs_ecc_td(self, injection_parameters_a, injection_parameters_b, **kwargs):
        """
        Generate PyCBC time domain PAIRS waveforms (Eccentric).
        """
        injection_parameters_c = injection_parameters_b.copy()
        injection_parameters_c['geocent_time'] = injection_parameters_a['geocent_time']
        
        e = kwargs.get('e', 0)
        anomaly = kwargs.get('anomaly', 0)

        ht_a = self.wf_ecc_td(injection_parameters_a, e=e, anomaly=anomaly, **kwargs)
        ht_b = self.wf_ecc_td(injection_parameters_c, e=e, anomaly=anomaly, **kwargs)

        ht = {}
        for det in ht_a.keys():
            max_len = max(len(ht_a[det]), len(ht_b[det]))
            ht_a[det].resize(max_len)
            ht_b[det].resize(max_len)

            time_diff = injection_parameters_b['geocent_time'] - injection_parameters_a['geocent_time']
            idx_shift = int(time_diff / ht_b[det].delta_t)
            
            ht_c_data = np.roll(ht_b[det]._data, idx_shift)
            ht_c = pycbc.types.TimeSeries(ht_c_data, delta_t = ht_b[det].delta_t)  
            ht_c.start_time = ht_a[det].start_time # Align epochs

            ht_pairs = ht_a[det]._data + ht_c._data
            ht_pairs = pycbc.types.TimeSeries(ht_pairs, delta_t = ht_a[det].delta_t)  
            ht_pairs.start_time = ht_a[det].start_time

            ht[det] = ht_pairs

        return ht
    
    def pairs_ecc_td(self, injection_parameters_a, injection_parameters_b, **kwargs):
        """
        Generate PyCBC time domain PAIRS waveforms (Eccentric).
        """
        e = kwargs.get('e', 0)
        anomaly = kwargs.get('anomaly', 0)

        ht_a = self.wf_ecc_td(injection_parameters_a, e=e, anomaly=anomaly, **kwargs)
        ht_b = self.wf_ecc_td(injection_parameters_b, e=e, anomaly=anomaly, **kwargs)

        ht_H1_a, ht_L1_a, ht_V1_a = ht_a['H1'], ht_a['L1'], ht_a['V1']
        ht_H1_b, ht_L1_b, ht_V1_b = ht_b['H1'], ht_b['L1'], ht_b['V1']

        delta_f = np.min([ht_H1_a.delta_f, ht_L1_a.delta_f, ht_V1_a.delta_f, ht_H1_b.delta_f, ht_L1_b.delta_f, ht_V1_b.delta_f])
        hf_a_H1, hf_a_L1, hf_a_V1 = ht_H1_a.to_frequencyseries(delta_f=delta_f), ht_L1_a.to_frequencyseries(delta_f=delta_f), ht_V1_a.to_frequencyseries(delta_f=delta_f)
        hf_b_H1, hf_b_L1, hf_b_V1 = ht_H1_b.to_frequencyseries(delta_f=delta_f), ht_L1_b.to_frequencyseries(delta_f=delta_f), ht_V1_b.to_frequencyseries(delta_f=delta_f)

        hf_H1 = hf_a_H1 + hf_b_H1 * np.exp(-1j * 2 * np.pi * (injection_parameters_b['geocent_time'] - injection_parameters_a['geocent_time']) * hf_b_H1.sample_frequencies)
        hf_L1 = hf_a_L1 + hf_b_L1 * np.exp(-1j * 2 * np.pi * (injection_parameters_b['geocent_time'] - injection_parameters_a['geocent_time']) * hf_b_L1.sample_frequencies)
        hf_V1 = hf_a_V1 + hf_b_V1 * np.exp(-1j * 2 * np.pi * (injection_parameters_b['geocent_time'] - injection_parameters_a['geocent_time']) * hf_b_V1.sample_frequencies)

        hf_H1 = pycbc.types.FrequencySeries(hf_H1, delta_f = hf_a_H1.delta_f)
        hf_L1 = pycbc.types.FrequencySeries(hf_L1, delta_f = hf_a_L1.delta_f)
        hf_V1 = pycbc.types.FrequencySeries(hf_V1, delta_f = hf_a_V1.delta_f)
        
        return {'H1': hf_H1, 'L1': hf_L1, 'V1': hf_V1}