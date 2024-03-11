import lal
import pycbc
import lalsimulation
import numpy as np 

class PairsWaveformGeneration:
    """
    Generating PAIRS Waveforms.

    """

    def __init__(self, **kwargs):
        self.duration = kwargs.get('duration', 4.0)
        self.sampling_frequency = kwargs.get('sampling_frequency', 4096)
        self.f_lower = kwargs.get('f_lower', 20.0)
        self.f_ref = kwargs.get('f_ref', 50.0)
        self.f_high = kwargs.get('f_high', 1024.0)
        self.delta_f = kwargs.get('delta_f', 1.0 / self.duration)
        self.delta_t = kwargs.get('delta_t', 1.0 / self.sampling_frequency)

    def wf_len_mod_start(self, wf, extra=1, **prms):
        """
        Taken from GWMAT. Function to modify the starting of a WF so that it starts on an integer GPS time (in sec) + add extra length as specified by the user.

        Parameters
        ----------
        wf :  pycbc.types.TimeSeries
            WF whose length is to be modified.
        extra : int, optional
            Extra length to be added in the beginning after making the WF to start from an integer GPS time (in sec). Default = 1.

        Returns
        -------
        pycbc.types.timeseries.TimeSeries
            Modified waveform starting form an integer time.

        """      

        olen = len(wf)   
        diff = wf.sample_times[0]-np.floor(wf.sample_times[0])  
        #nlen = round(olen+sampling_frequency*(extra+diff))
        dlen = round(self.sampling_frequency*(extra+diff))
        wf_strain = np.concatenate((np.zeros(dlen), wf))
        t0 = wf.sample_times[0]
        dt = wf.delta_t
        n = dlen
        tnn = t0-(n+1)*dt
        wf_stime = np.concatenate((np.arange(t0-dt,tnn,-dt)[::-1], np.array(wf.sample_times)))
        nwf = pycbc.types.TimeSeries(wf_strain, delta_t=wf.delta_t, epoch=wf_stime[0])
        
        return nwf

    def wf_len_mod_end(self, wf, extra=2, **prms): #post_trig_duration
        """
        Taken from GWMAT. Function to modify the end of a WF so that it ends on an integer GPS time (in sec) + add extra length as specified by the user.

        Parameters
        ----------
        wf : pycbc.types.TimeSeries
            WF whose length is to be modified.
        extra : int, optional
            Extra length to be added towards the end after making the WF to end from an integer GPS time (in sec). 
            Default = 2, which makes sure post-trigger duration is of at least 2 seconds.

        Returns
        -------
        pycbc.types.timeseries.TimeSeries
            Modified waveform ending on an integer time.

        """        

        olen = len(wf)   
        dt = abs(wf.sample_times[-1] - wf.sample_times[-2])
        diff = np.ceil(wf.sample_times[-1]) - (wf.sample_times[-1] + dt)   #wf.sample_times[-1]-int(wf.sample_times[-1])  
        nlen = round(olen + self.sampling_frequency*(extra+diff))
        wf.resize(nlen)
        
        return wf    

    def make_len_power_of_2(self, wf):
        """
        Taken from GWMAT. Function to modify the length of a waveform so that its duration is a power of 2.

        Parameters
        ----------
        wf : pycbc.types.TimeSeries
            WF whose length is to be modified.
            Modified waveform with duration a power of 2.
        Returns
        -------
        pycbc.types.timeseries.TimeSeries
            Returns the waveform with length a power of 2.

        """    

        dur = wf.duration  
        wf.resize( int(round(wf.sample_rate * np.power(2, np.ceil( np.log2( dur ) ) ))) )
        wf = self.cyclic_time_shift_of_WF(wf, rwrap = wf.duration - dur )
        
        return wf

    def cyclic_time_shift_of_WF(self, wf, rwrap=0.2):
        """
        Taken from GWMAT. Inspired by PyCBC's function pycbc.types.TimeSeries.cyclic_time_shift(), 
            it shifts the data and timestamps in the time domain by a given number of seconds (rwrap). 
            Difference between this and PyCBCs function is that this function preserves the sample rate of the WFs while cyclically rotating, 
            but the time shift cannot be smaller than the intrinsic sample rate of the data, unlike PyCBc's function.
            To just change the time stamps, do ts.start_time += dt.
            Note that data will be cyclically rotated, so if you shift by 2
            seconds, the final 2 seconds of your data will now be at the
            beginning of the data set.

        Parameters
        ----------
        wf : pycbc.types.TimeSeries
            The waveform for cyclic rotation.
        rwrap : float, optional
            Amount of time to shift the vector. Default = 0.2.

        Returns
        -------
        pycbc.types.TimeSeries
            The time shifted time series.

        """        

        # This function does cyclic time shift of a WF.
        # It is similar to PYCBC's "cyclic_time_shift" except for the fact that it also preserves the Sample Rate of the original WF.
        if rwrap is not None and rwrap != 0:
            sn = abs(int(rwrap/wf.delta_t))     # number of elements to be shifted 
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

    def wf_td(self, injection_parameters, **kwargs):
        """
        Generate PyCBC time domain SINGLES waveforms for the injection parameters, and returns injection projections of a signal onto the Hanford, Livingston, Virgo detectors
        
        """
        
        def jframe_to_l0frame(mass_1, mass_2, f_ref, phi_ref=0., theta_jn=0., phi_jl=0., a_1=0., a_2=0., tilt_1=0., tilt_2=0., phi_12=0., **kwargs):  
            """
            [Inherited from PyCBC and lalsimulation.]
                Function to convert J-frame coordinates (which Bilby uses for PE) to L0-frame coordinates (that Pycbc uses for waveform generation).
                J stands for the total angular momentum while L0 stands for the orbital angular momentum.
            """ 

            inclination, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = \
                lalsimulation.SimInspiralTransformPrecessingNewInitialConditions(
                    theta_jn, phi_jl, tilt_1, tilt_2, phi_12,
                    a_1, a_2, mass_1*lal.MSUN_SI, mass_2*lal.MSUN_SI, f_ref,
                    phi_ref)
            out_dict = {'inclination': inclination,
                        'spin1x': spin1x,
                        'spin1y': spin1y,
                        'spin1z': spin1z,
                        'spin2x': spin2x,
                        'spin2y': spin2y,
                        'spin2z': spin2z}
            return out_dict

        lframe = jframe_to_l0frame(mass_1=injection_parameters['mass_1'], 
                                mass_2=injection_parameters['mass_2'], 
                                f_ref=self.f_ref, 
                                theta_jn=injection_parameters['theta_jn'], 
                                phi_jl=injection_parameters['phi_jl'], 
                                a_1=injection_parameters['a_1'], 
                                a_2=injection_parameters['a_2'], 
                                tilt_1=injection_parameters['tilt_1'], 
                                tilt_2=injection_parameters['tilt_2'], 
                                phi_12=injection_parameters['phi_12'])    

        waveform_params = {
            'approximant': 'IMRPhenomXPHM',
            'mass1': injection_parameters['mass_1'],
            'mass2': injection_parameters['mass_2'],
            'spin1x': lframe['spin1x'],
            'spin1y': lframe['spin1y'],
            'spin1z': lframe['spin1z'],
            'spin2x': lframe['spin2x'],
            'spin2y': lframe['spin2y'],
            'spin2z': lframe['spin2z'],
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

        ht_H1, ht_L1, ht_V1 = ifo_signal['H1'], ifo_signal['L1'], ifo_signal['V1']
        ht = {'H1': ht_H1, 'L1': ht_L1, 'V1': ht_V1}

        return ht

    def pairs_td(self, injection_parameters_a, injection_parameters_b, **kwargs):
        """
        Generate PyCBC time domain PAIRS waveforms for the injection parameters, and returns injection projections of a signal onto the Hanford, Livingston, Virgo detectors
        
        """
        def jframe_to_l0frame(mass_1, mass_2, f_ref, phi_ref=0., theta_jn=0., phi_jl=0., a_1=0., a_2=0., tilt_1=0., tilt_2=0., phi_12=0., **kwargs):  
            """
            [Inherited from PyCBC and lalsimulation.]
                Function to convert J-frame coordinates (which Bilby uses for PE) to L0-frame coordinates (that Pycbc uses for waveform generation).
                J stands for the total angular momentum while L0 stands for the orbital angular momentum.
            """ 

            inclination, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = \
                lalsimulation.SimInspiralTransformPrecessingNewInitialConditions(
                    theta_jn, phi_jl, tilt_1, tilt_2, phi_12,
                    a_1, a_2, mass_1*lal.MSUN_SI, mass_2*lal.MSUN_SI, f_ref,
                    phi_ref)
            out_dict = {'inclination': inclination,
                        'spin1x': spin1x,
                        'spin1y': spin1y,
                        'spin1z': spin1z,
                        'spin2x': spin2x,
                        'spin2y': spin2y,
                        'spin2z': spin2z}
            return out_dict
        
        lframe_a = jframe_to_l0frame(mass_1=injection_parameters_a['mass_1'], 
                                    mass_2=injection_parameters_a['mass_2'], 
                                    f_ref=self.f_ref, 
                                    theta_jn=injection_parameters_a['theta_jn'], 
                                    phi_jl=injection_parameters_a['phi_jl'], 
                                    a_1=injection_parameters_a['a_1'], 
                                    a_2=injection_parameters_a['a_2'], 
                                    tilt_1=injection_parameters_a['tilt_1'], 
                                    tilt_2=injection_parameters_a['tilt_2'], 
                                    phi_12=injection_parameters_a['phi_12'])   

        waveform_params_a = {
            'approximant': 'IMRPhenomXPHM',
            'mass1': injection_parameters_a['mass_1'],
            'mass2': injection_parameters_a['mass_2'],
            'spin1x': lframe_a['spin1x'],
            'spin1y': lframe_a['spin1y'],
            'spin1z': lframe_a['spin1z'],
            'spin2x': lframe_a['spin2x'],
            'spin2y': lframe_a['spin2y'],
            'spin2z': lframe_a['spin2z'],
            'distance': injection_parameters_a['luminosity_distance'],
            'inclination': lframe_a['inclination'],
            'coa_phase': injection_parameters_a['phase'],
            'f_lower': self.f_lower,
            'f_ref': self.f_ref,
            'delta_t': self.delta_t
        }

        lframe_b = jframe_to_l0frame(mass_1=injection_parameters_b['mass_1'],
                                    mass_2=injection_parameters_b['mass_2'],
                                    f_ref=self.f_ref,
                                    theta_jn=injection_parameters_b['theta_jn'],
                                    phi_jl=injection_parameters_b['phi_jl'],
                                    a_1=injection_parameters_b['a_1'],
                                    a_2=injection_parameters_b['a_2'],
                                    tilt_1=injection_parameters_b['tilt_1'],
                                    tilt_2=injection_parameters_b['tilt_2'],
                                    phi_12=injection_parameters_b['phi_12'])

        waveform_params_b = {
            'approximant': 'IMRPhenomXPHM',
            'mass1': injection_parameters_b['mass_1'],
            'mass2': injection_parameters_b['mass_2'],
            'spin1x': lframe_b['spin1x'],
            'spin1y': lframe_b['spin1y'],
            'spin1z': lframe_b['spin1z'],
            'spin2x': lframe_b['spin2x'],
            'spin2y': lframe_b['spin2y'],
            'spin2z': lframe_b['spin2z'],
            'distance': injection_parameters_b['luminosity_distance'],
            'inclination': lframe_b['inclination'],
            'coa_phase': injection_parameters_b['phase'],
            'f_lower': self.f_lower,
            'f_ref': self.f_ref,
            'delta_t': self.delta_t
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
        hf_b_H1, hf_b_L1, hf_b_V1 = ht_H1_a.to_frequencyseries(delta_f=delta_f), ht_L1_a.to_frequencyseries(delta_f=delta_f), ht_V1_b.to_frequencyseries(delta_f=delta_f)

        hf_H1 = hf_a_H1 + hf_b_H1 * np.exp(-1j * 2 * np.pi * (injection_parameters_b['geocent_time'] - injection_parameters_a['geocent_time']) * hf_b_H1.sample_frequencies)
        hf_L1 = hf_a_L1 + hf_b_L1 * np.exp(-1j * 2 * np.pi * (injection_parameters_b['geocent_time'] - injection_parameters_a['geocent_time']) * hf_b_L1.sample_frequencies)
        hf_V1 = hf_a_V1 + hf_b_V1 * np.exp(-1j * 2 * np.pi * (injection_parameters_b['geocent_time'] - injection_parameters_a['geocent_time']) * hf_b_V1.sample_frequencies)

        ht_H1, ht_L1, ht_V1 = hf_H1.to_timeseries(), hf_L1.to_timeseries(), hf_V1.to_timeseries()
        ht = {'H1': ht_H1, 'L1': ht_L1, 'V1': ht_V1}

        return ht
    
    def import_lookup(self):
        import pickle
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
        import sys
        sys.path.append("/home/nishkal.rao/GWMAT/pnt_Ff_lookup_table/src/cythonized_pnt_lens_class")   
        import cythonized_pnt_lens_class as pnt_lens_cy  

        from scipy.interpolate import interp1d
        
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

    def wf_ml_fd(self, injection_parameters, Ml_z, y, **kwargs):
        """
        Generate microlensed time domain waveforms for the injection parameters, and returns injection projections of a signal onto the Hanford, Livingston, Virgo detectors (using GWMAT)
        
        """       
        self.import_lookup()

        ht = self.wf_td(injection_parameters)
        hf_H1, hf_L1, hf_V1 = ht['H1'].to_frequencyseries(), ht['L1'].to_frequencyseries(), ht['V1'].to_frequencyseries()

        if round(Ml_z) == 0:
            return {'H1': hf_H1, 'L1': hf_L1, 'V1': hf_V1}
        
        else:
            Ff = self.pnt_Ff_lookup_table(ys_grid=self.ys_grid, ws_grid=self.ws_grid, fs=hf_H1.sample_frequencies, Mlz=Ml_z, yl=y)

            hfl_H1, hfl_L1, hfl_V1 = pycbc.types.FrequencySeries(Ff*hf_H1, delta_f = hf_H1.delta_f), pycbc.types.FrequencySeries(Ff*hf_L1, delta_f = hf_L1.delta_f), pycbc.types.FrequencySeries(Ff*hf_V1, delta_f = hf_V1.delta_f)
            hfl_H1.start_time, hfl_L1.start_time, hfl_V1.start_time = injection_parameters['geocent_time']-self.duration+2, injection_parameters['geocent_time']-self.duration+2, injection_parameters['geocent_time']-self.duration+2
            hf_lens = {'H1': hfl_H1, 'L1': hfl_L1, 'V1': hfl_V1}

            return hf_lens

    def pairs_ml_fd(self, injection_parameters_a, injection_parameters_b, **kwargs):
        """
        Generate frequency domain PAIRS waveforms for the injection parameters, and returns injection projections of a signal onto the Hanford, Livingston, Virgo detectors
        
        """

        hf_a, hf_b = self.wf_ml_fd(injection_parameters=injection_parameters_a, Ml_z=0, y=0), self.wf_ml_fd(injection_parameters=injection_parameters_b, Ml_z=0, y=0)

        ht_H1_a, ht_L1_a, ht_V1_a = hf_a['H1'].to_timeseries(), hf_a['L1'].to_timeseries(), hf_a['V1'].to_timeseries()
        ht_H1_b, ht_L1_b, ht_V1_b = hf_b['H1'].to_timeseries(), hf_b['L1'].to_timeseries(), hf_b['V1'].to_timeseries()

        delta_f = np.min([ht_H1_a.delta_f, ht_L1_a.delta_f, ht_V1_a.delta_f, ht_H1_b.delta_f, ht_L1_b.delta_f, ht_V1_b.delta_f])
        hf_a_H1, hf_a_L1, hf_a_V1 = ht_H1_a.to_frequencyseries(delta_f=delta_f), ht_L1_a.to_frequencyseries(delta_f=delta_f), ht_V1_a.to_frequencyseries(delta_f=delta_f)
        hf_b_H1, hf_b_L1, hf_b_V1 = ht_H1_a.to_frequencyseries(delta_f=delta_f), ht_L1_a.to_frequencyseries(delta_f=delta_f), ht_V1_b.to_frequencyseries(delta_f=delta_f)

        hf_H1 = hf_a_H1 + hf_b_H1 * np.exp(-1j * 2 * np.pi * (injection_parameters_b['geocent_time'] - injection_parameters_a['geocent_time']) * hf_b_H1.sample_frequencies)
        hf_L1 = hf_a_L1 + hf_b_L1 * np.exp(-1j * 2 * np.pi * (injection_parameters_b['geocent_time'] - injection_parameters_a['geocent_time']) * hf_b_L1.sample_frequencies)
        hf_V1 = hf_a_V1 + hf_b_V1 * np.exp(-1j * 2 * np.pi * (injection_parameters_b['geocent_time'] - injection_parameters_a['geocent_time']) * hf_b_V1.sample_frequencies)

        hf_H1, hf_L1, hf_V1 = pycbc.types.FrequencySeries(hf_H1, delta_f = hf_a_H1.delta_f), pycbc.types.FrequencySeries(hf_L1, delta_f = hf_a_L1.delta_f), pycbc.types.FrequencySeries(hf_V1, delta_f = hf_a_V1.delta_f)
        hf = {'H1': hf_H1, 'L1': hf_L1, 'V1': hf_V1}

        return hf

    def wf_ecc_td(self, injection_parameters, e, **kwargs):
        """
        Generate PyCBC time domain eccentric waveforms for the injection parameters, and returns injection projections of a signal onto the Hanford, Livingston, Virgo detectors
        """

        import sys
        sys.path.append('/home/nishkal.rao/gweat/src/')
        import TEOBResumS_utils as ecc_gen

        def jframe_to_l0frame(mass_1, mass_2, f_ref, phi_ref=0., theta_jn=0., phi_jl=0., a_1=0., a_2=0., tilt_1=0., tilt_2=0., phi_12=0., **kwargs):  
            """
            [Inherited from PyCBC and lalsimulation.]
                Function to convert J-frame coordinates (which Bilby uses for PE) to L0-frame coordinates (that Pycbc uses for waveform generation).
                J stands for the total angular momentum while L0 stands for the orbital angular momentum.
            """ 

            inclination, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = \
                lalsimulation.SimInspiralTransformPrecessingNewInitialConditions(
                    theta_jn, phi_jl, tilt_1, tilt_2, phi_12,
                    a_1, a_2, mass_1*lal.MSUN_SI, mass_2*lal.MSUN_SI, f_ref,
                    phi_ref)
            out_dict = {'inclination': inclination,
                        'spin1x': spin1x,
                        'spin1y': spin1y,
                        'spin1z': spin1z,
                        'spin2x': spin2x,
                        'spin2y': spin2y,
                        'spin2z': spin2z}
            return out_dict
        
        lframe = jframe_to_l0frame(mass_1=injection_parameters['mass_1'], 
                                mass_2=injection_parameters['mass_2'], 
                                f_ref=self.f_ref, 
                                theta_jn=injection_parameters['theta_jn'], 
                                phi_jl=injection_parameters['phi_jl'], 
                                a_1=injection_parameters['a_1'], 
                                a_2=injection_parameters['a_2'], 
                                tilt_1=injection_parameters['tilt_1'], 
                                tilt_2=injection_parameters['tilt_2'], 
                                phi_12=injection_parameters['phi_12'])    

        waveform_params = {
            'approximant': 'IMRPhenomXPHM',
            'mass_1': injection_parameters['mass_1'],
            'mass_2': injection_parameters['mass_2'],
            'spin1x': lframe['spin1x'],
            'spin1y': lframe['spin1y'],
            'spin1z': lframe['spin1z'],
            'spin2x': lframe['spin2x'],
            'spin2y': lframe['spin2y'],
            'spin2z': lframe['spin2z'],
            'luminosity_distance': injection_parameters['luminosity_distance'],
            'inclination': lframe['inclination'],
            'coa_phase': injection_parameters['phase'],
            'f_lower': self.f_lower,
            'f_ref': self.f_ref,
            'delta_t': self.delta_t,
            'ecc': e
        }

        pars = ecc_gen.teobresums_pars_update(waveform_params)
        h = ecc_gen.teobresums_td_pure_polarized_wf_gen(**pars)
        hp, hc = pycbc.types.TimeSeries(h['hp'], delta_t = h['hp'].delta_t), pycbc.types.TimeSeries(h['hc'], delta_t = h['hc'].delta_t)
        hp.start_time += injection_parameters['geocent_time']
        hc.start_time += injection_parameters['geocent_time']
            
        from pycbc.waveform.utils import taper_timeseries

        det, ifo_signal = dict(), dict()
        for ifo in ['H1', 'L1', 'V1']:
            det[ifo] = pycbc.detector.Detector(ifo)
            ifo_signal[ifo] = det[ifo].project_wave(hp, hc, injection_parameters['ra'], injection_parameters['dec'], injection_parameters['psi'])
            ifo_signal[ifo] = taper_timeseries(ifo_signal[ifo], tapermethod='TAPER_STARTEND', return_lal=False)
            ifo_signal[ifo] = self.make_len_power_of_2(self.wf_len_mod_end(self.wf_len_mod_start(ifo_signal[ifo])))

        ht_H1, ht_L1, ht_V1 = ifo_signal['H1'], ifo_signal['L1'], ifo_signal['V1']

        ht_H1.start_time += injection_parameters['geocent_time']-ht_H1.sample_times[np.argmax(ht_H1)]
        ht_L1.start_time += injection_parameters['geocent_time']-ht_L1.sample_times[np.argmax(ht_L1)]
        ht_V1.start_time += injection_parameters['geocent_time']-ht_V1.sample_times[np.argmax(ht_V1)]
        
        ht = {'H1': ht_H1, 'L1': ht_L1, 'V1': ht_V1}

        return ht

    def pairs_ecc_td(self, injection_parameters_a, injection_parameters_b, **kwargs):
        """
        Generate PyCBC time domain PAIRS waveforms for the injection parameters, and returns injection projections of a signal onto the Hanford, Livingston, Virgo detectors
        """

        injection_parameters_c = injection_parameters_b.copy()
        injection_parameters_c['geocent_time'] = injection_parameters_a['geocent_time']
        
        ht_a, ht_b = self.wf_ecc_td(injection_parameters_a, 0), self.wf_ecc_td(injection_parameters_c, 0)

        ht = {}
        for det in ht_a.keys():
            ht_a[det].resize(max(len(ht_a[det]), len(ht_b[det])))
            ht_b[det].resize(max(len(ht_a[det]), len(ht_b[det])))

            ht_c = np.roll(ht_b[det]._data, int((injection_parameters_b['geocent_time']-injection_parameters_a['geocent_time'])/(ht_b[det].delta_t)-(np.argmax(ht_b[det])-np.argmax(ht_a[det]))))
            ht_c = pycbc.types.TimeSeries(ht_c, delta_t = ht_b[det].delta_t)  
            ht_c.start_time = ht_b[det].start_time

            ht_pairs = ht_a[det]._data + ht_c._data
            ht_pairs = pycbc.types.TimeSeries(ht_pairs, delta_t = ht_a[det].delta_t)  
            ht_pairs.start_time = ht_a[det].start_time

            ht[det] = ht_pairs

        return ht

##############################################################################################################################################################################
##############################################################################################################################################################################