#!/home/nishkal.rao/miniconda3/envs/gw_analysis/bin/python

"""
A Bilby source file for performing parameter estimations with
TOBResumS waveform model, including eccentricity.

It assumes TEOBResumS and relevant libraries are installed
before using this source file.

To speed up WF evaluation, the tolerance values for the ODE
integrators are loosened below the default values.
For reference, see: O'Shea. & Kumar (2021): https://arxiv.org/pdf/2107.07981.pdf
Also, see their implementation of Bilby source model:
https://github.com/osheamonn/eobresum_bilby/blob/main/eob_resum_bilby_layer.py

Since EOB performs an ODE integration, we can't control the actual
length of the result, which means the df of the FT won't match with
df of frequency_array. To resolve this, we set modify the duration of
the generated WF to be an integral multiple of the analysis_duration.
This makes the frequency resolution of the waveform an integral multiple
of the required frequency spacing by the Bilby. This allows us to sub-sample
to the required frequency array.

Author: Anuj Mishra
Contributors: Anirudh S. Nemmani, Akash Maurya.
"""

import numpy as np
import gwmat
from gwmat.bilby_custom_FD_source_models import microlensing_source
import gweat.teobresums.utils as ecc_gen


def determine_time_shift(wf):
    """
    Computes the time shift required to align the
    peak of the waveform with its end time.
    """
    wf_end_time = float(wf.end_time)
    peak_time = wf.sample_times[np.argmax(abs(np.asarray(wf)))]
    t_shift = wf_end_time - peak_time
    return t_shift


def next_power_of_2(num):
    """
    Returns the next power of 2 greater
    than or equal to the given number.
    """
    return 1 if num == 0 else 2 ** np.ceil(np.log2(num))


# TD TEOBResumS WF generator
def eccentric_teobresums_binary_black_hole_TD(
    mass_1,
    mass_2,
    luminosity_distance,
    chi_1x,
    chi_1y,
    chi_1z,
    chi_2x,
    chi_2y,
    chi_2z,
    inclination,
    phase,
    ecc,
    anomaly,
    **kwargs,
):
    """
    This is a wrapper function to call TOBResumS WF generator
    through the utils source file `TEOBResumS_utils.py`.
    It returns tapered TD plus and cross polarized WFs.

    Note: This is not a TD source model for Bilby.

    Parameters
    ----------
    * mass_1 : float
        The mass of the primary component object in the binary
        (in solar masses).
    * mass_2 : float
        The mass of the secondary component object in the binary
        (in solar masses).
    * chi1x : float
        The x component of the first binary component. Default = 0.
    * chi1y : float
        The y component of the first binary component. Default = 0.
    * chi1z : float
        The z component of the first binary component. Default = 0.
    * chi2x : float
        The x component of the second binary component. Default = 0.
    * chi2y : float
        The y component of the second binary component. Default = 0.
    * chi2z : float
        The z component of the second binary component. Default = 0.
    * ecc : float
        Eccentricity of the binary at a reference frequency of f_start.
    * luminosity_distance : ({100.,float}), optional
        Luminosity distance to the binary (in Mpc).
    * inclination : float
        Inclination (rad), defined as the angle between the
        orbital angular momentum J(or, L) and the line-of-sight
        at the reference frequency. Default = 0.
    * phase : ({0.,float}), optional
        Coalesence phase of the binary (in rad).
    * kwargs: dictionary
        * f_start : ({10., float}), optional
            Reference frequency for defining eccentricity.
            This is also the starting frequency of the (2, 2)
            mode for waveform generation (in Hz).
            When use_geometric_units = 0, units is Hz,
            else it is in geometric units.
        * ecc_freq : ({2, int}), optional
             Use periastron (0), average (1) or apastron (2) frequency
             for initial condition computation. Default = 2.
        * sample_rate : ({4096, int}), optional
            Sample rate for the TEOBResumS WF generation.
            It should be more than the "sampling-frequency" used by
            Bilby for PE. Ideally, twice the value works fine.
        * mode_array : ({[[2,2]], 2D list}), optional
            Mode array for the WF generation.
        * ode_abstol : ({1e-8, float}), optional
            Absolute numerical tolerance for ODE compuations used in TEOBResumS.
            Default=1e-8, which is more than the original default of
            1e-13 for speeding up WF evaluations.
        * ode_reltol : ({1e-7, float}), optional
            Relative numerical tolerance for ODE compuations used in TEOBResumS.
            Default=1e-7, which is more than the original default of
            1e-11 for speeding up WF evaluations.
        * min_td_duration : int, optional
            Minimum duration of time-domain WF. The larger it is,
            the better is the frequency resolution.
            Default is min_td_duration = analysis_duration passed in Bilby.
            If passing manually, ensure that min_td_duration > analysis_duration.
    Returns
    -------
    Dictionary:
        * plus: A tapered PyCBC Timeseries object.
            The plus polarized WF.
        * cross: A tapered PyCBC Timeseries object.
            The cross polarized WF.

    """

    waveform_kwargs = {
        "f_start": 10,
        "sample_rate": 4096,
        "mode_array": [[2, 1], [2, 2], [3, 2], [3, 3], [4, 4]],
        "ecc_freq": 0,
        "ode_abstol": 1e-8,
        "ode_reltol": 1e-7,
        "min_td_duration": 4,
    }
    waveform_kwargs.update(kwargs)
    waveform_kwargs["mode_array"] = [
        [int(a[0]), int(a[1])] for a in waveform_kwargs["mode_array"]
    ]

    # https://bitbucket.org/eob_ihes/teobresums/wiki/Conventions,%20parameters%20and%20output
    pars = {
        "mass_1": mass_1,
        "mass_2": mass_2,
        "chi1x": chi_1x,
        "chi1y": chi_1y,
        "chi1z": chi_1z,
        "chi2x": chi_2x,
        "chi2y": chi_2y,
        "chi2z": chi_2z,
        "chi1": chi_1z,
        "chi2": chi_2z,
        "luminosity_distance": luminosity_distance,
        "inclination": inclination,
        "coa_phase": phase,
        "srate_interp": waveform_kwargs["sample_rate"],
        "initial_frequency": waveform_kwargs["f_start"],
        "ecc": ecc,
        "anomaly": anomaly,
        "ecc_freq": waveform_kwargs["ecc_freq"],
        "ode_abstol": waveform_kwargs["ode_abstol"],
        "ode_reltol": waveform_kwargs["ode_reltol"],
    }

    pars.update(waveform_kwargs)
    wfs_res = ecc_gen.teobresums_td_pure_polarized_wf_gen(**pars)
    hp, hc = wfs_res["hp"], wfs_res["hc"]

    # modify duration
    mod_wf_duration = waveform_kwargs["min_td_duration"] * np.ceil(
        hp.duration / waveform_kwargs["min_td_duration"]
    )
    hp = gwmat.injection.modify_signal_start_time(
        hp, extra=mod_wf_duration - hp.duration
    )
    hc = gwmat.injection.modify_signal_start_time(
        hc, extra=mod_wf_duration - hc.duration
    )

    # shifting the peak of the WF to t=0.
    wf = hp - 1j * hc
    wf = gwmat.gw_utils.cyclic_time_shift_of_wf(wf, rwrap=determine_time_shift(wf))
    wf.start_time = 0
    hp, hc = wf.real(), -1 * wf.imag()

    return {"plus": hp, "cross": hc}


# Eccentric aligned spin TEOBResumS FD Source model for Bilby
def eccentric_aligned_spin_teobresums_binary_black_hole_FD(
    frequency_array,
    mass_1,
    mass_2,
    luminosity_distance,
    chi_1z,
    chi_2z,
    theta_jn,
    phase,
    ecc,
    anomaly,
    **kwargs,
):
    """
    This is a Bilby Frequency Domain Source model for performing
    parameter estimations with TOBResumS eccentric aligned-spin model.

    It returns FD plus and cross polarized WFs
    at the required frequency_array.

    The arguments in the kwargs can be updated through
    `waveform-arguments-dict` in the config ini file of
    bilby-pipe. For example, the default values are:
    waveform-arguments-dict = {f_start=10, ecc_freq=0, sample_rate=2048,
    mode_array=[[2, 1], [2, 2], [3, 2], [3, 3], [4, 4]],
    ode_abstol=1e-8, ode_reltol=1e-7}

    Parameters
    ----------
    * frequency_array : float
        Array of frequency values where the WF will be evaluated.
    * mass_1 : float
        The mass of the primary component object in the binary (in solar masses).
    * mass_2 : float
        The mass of the secondary component object in the binary (in solar masses).
    * chi1z : float
        The z component of the first binary component. Default = 0.
    * chi2z : float
        The z component of the second binary component. Default = 0.
    * ecc : float
        Eccentricity of the binary at a reference frequency of f_start.
    * luminosity_distance : ({100.,float}), optional
        Luminosity distance to the binary (in Mpc).
    * theta_jn : float
        Inclination (rad), defined as the angle between the orbital
        angular momentum J(or, L) and the line-of-sight at the
        reference frequency. Default = 0.
    * phase : ({0.,float}), optional
        Coalesence phase of the binary (in rad).
    * kwargs: dictionary
        * f_start : ({10., float}), optional
            Reference frequency for defining eccentricity.
            This is also the starting frequency of the
            (2, 2) mode for waveform generation (in Hz).
        * ecc_freq : ({0, int}), optional
             Use periastron (0), average (1) or apastron (2)
             frequency for initial condition computation.
             Default = 2.
        * sample_rate : ({4096, int}), optional
            Sample rate for the TEOBResumS WF generation.
            It should be more than the "sampling-frequency"
            used by Bilby for PE. Ideally, twice the value works fine.
        * mode_array : ({[[2,2]], 2D list}), optional
            Mode array for the WF generation.
        * ode_abstol : ({1e-8, float}), optional
            Absolute numerical tolerance for ODE
            compuations used in TEOBResumS.
            Default=1e-8, which is more than the original
            default of 1e-13 for speeding up WF evaluations.
        * ode_reltol : ({1e-7, float}), optional
            Relative numerical tolerance for ODE
            compuations used in TEOBResumS.
            Default=1e-7, which is more than the original
            default of 1e-11 for speeding up WF evaluations.
        * min_td_duration : int, optional
            Minimum duration of time-domain WF.
            The larger it is, the better is the frequency resolution.
            Default is min_td_duration = analysis_duration passed in Bilby.
            If passing manually, ensure that min_td_duration > analysis_duration.

    Returns
    -------
    Dictionary:
        * plus: A numpy array.
            Strain values of the plus polarized WF in Frequency Domain..
        * cross: A numpy array.
            Strain values of the cross polarized WF in Frequency Domain.

    """
    waveform_kwargs = {
        "reference_frequency": 20.0,
        "minimum_frequency": 20.0,
        "maximum_frequency": frequency_array[-1],
        "f_start": 10,
        "sample_rate": 4096,
        "mode_array": [[2, 1], [2, 2], [3, 2], [3, 3], [4, 4]],
        "ecc_freq": 0,
        "ode_abstol": 1e-8,
        "ode_reltol": 1e-7,
        "min_td_duration": 1 / (frequency_array[1] - frequency_array[0]),
        "verbose": False,
    }
    waveform_kwargs.update(kwargs)
    waveform_kwargs["mode_array"] = [
        [int(a[0]), int(a[1])] for a in waveform_kwargs["mode_array"]
    ]  # Bilby treats them as string so converting them to int manually.
    try:
        # generate TD TEOBResumS WF
        chi_1x = chi_1y = chi_2x = chi_2y = 0.0  # for aligned spin case
        wfs_res = eccentric_teobresums_binary_black_hole_TD(
            mass_1,
            mass_2,
            luminosity_distance,
            chi_1x,
            chi_1y,
            chi_1z,
            chi_2x,
            chi_2y,
            chi_2z,
            theta_jn,
            phase,
            ecc,
            anomaly,
            **waveform_kwargs,
        )

        res = {}
        for k, wf in wfs_res.items():
            ## converting TD WF -> FD WF
            fd_wf = wf.to_frequencyseries(delta_f=wf.delta_f)
            ## sub-resampling from the full freqeuncy array to match what Bilby requires
            df = frequency_array[1] - frequency_array[0]
            delta_f_ratio = df / fd_wf.delta_f

            # ensure that delta_f_ratio is an integer.
            assert (
                abs(delta_f_ratio - np.floor(delta_f_ratio)) < 1e-7
            ), f"The required delta_f {df:.3f} is not a multiple \
of TEOBResumS output delta_f {fd_wf.delta_f:.5f}."  ## may not be strictly zero when duration is not a power of 2 because of precision issues.
            # sub-sampling
            new_fd_wf = fd_wf[:: round(delta_f_ratio)][: len(frequency_array)]
            # ensure the sub-sampled frequency array is same as what Bilby requires.
            assert np.allclose(
                frequency_array, new_fd_wf.sample_frequencies.data
            ), "Bilby frequency array does not match the TEOBResumS output frequency array."
            # assign the waveform data to res
            fd_wf_arr = new_fd_wf.data
            res[k] = fd_wf_arr
        return res
    except Exception as e:
        failed_pars = dict(
            m1=mass_1,
            m2=mass_2,
            dL=luminosity_distance,
            chi1z=chi_1z,
            chi2z=chi_2z,
            incl=theta_jn,
            phase=phase,
            ecc=ecc,
            anomaly=anomaly,
        )
        if waveform_kwargs["verbose"]:
            print(f"[Waveform ERROR] {e} for pars:{failed_pars}; --returning None.")
        else:
            print(f"[Waveform ERROR] {e}; --returning None.")
        return None


# Eccentric aligned spin TEOBResumS FD Source model for Bilby
def eccentric_precessing_teobresums_binary_black_hole_FD(
    frequency_array,
    mass_1,
    mass_2,
    luminosity_distance,
    a_1,
    tilt_1,
    phi_12,
    a_2,
    tilt_2,
    phi_jl,
    theta_jn,
    phase,
    ecc,
    anomaly,
    **kwargs,
):
    """
    This is a Bilby Frequency Domain Source model for performing
    parameter estimations with TOBResumS eccentric aligned-spin model.

    It returns FD plus and cross polarized WFs
    at the required frequency_array.

    The arguments in the kwargs can be updated through
    `waveform-arguments-dict` in the config ini file of
    bilby-pipe. For example, the default values are:
    waveform-arguments-dict = {f_start=10, ecc_freq=0, sample_rate=2048,
    mode_array=[[2, 1], [2, 2], [3, 2], [3, 3], [4, 4]],
    ode_abstol=1e-8, ode_reltol=1e-7}

    Parameters
    ----------
    * frequency_array : float
        Array of frequency values where the WF will be evaluated.
    * mass_1 : float
        The mass of the primary component object in the binary (in solar masses).
    * mass_2 : float
        The mass of the secondary component object in the binary (in solar masses).
    * chi1z : float
        The z component of the first binary component. Default = 0.
    * chi2z : float
        The z component of the second binary component. Default = 0.
    * ecc : float
        Eccentricity of the binary at a reference frequency of f_start.
    * luminosity_distance : ({100.,float}), optional
        Luminosity distance to the binary (in Mpc).
    * theta_jn : float
        Inclination (rad), defined as the angle between the orbital
        angular momentum J(or, L) and the line-of-sight at the
        reference frequency. Default = 0.
    * phase : ({0.,float}), optional
        Coalesence phase of the binary (in rad).
    * kwargs: dictionary
        * f_start : ({10., float}), optional
            Reference frequency for defining eccentricity.
            This is also the starting frequency of the
            (2, 2) mode for waveform generation (in Hz).
        * ecc_freq : ({0, int}), optional
             Use periastron (0), average (1) or apastron (2)
             frequency for initial condition computation.
             Default = 2.
        * sample_rate : ({4096, int}), optional
            Sample rate for the TEOBResumS WF generation.
            It should be more than the "sampling-frequency"
            used by Bilby for PE. Ideally, twice the value works fine.
        * mode_array : ({[[2,2]], 2D list}), optional
            Mode array for the WF generation.
        * ode_abstol : ({1e-8, float}), optional
            Absolute numerical tolerance for ODE
            compuations used in TEOBResumS.
            Default=1e-8, which is more than the original
            default of 1e-13 for speeding up WF evaluations.
        * ode_reltol : ({1e-7, float}), optional
            Relative numerical tolerance for ODE
            compuations used in TEOBResumS.
            Default=1e-7, which is more than the original
            default of 1e-11 for speeding up WF evaluations.
        * min_td_duration : int, optional
            Minimum duration of time-domain WF.
            The larger it is, the better is the frequency resolution.
            Default is min_td_duration = analysis_duration passed in Bilby.
            If passing manually, ensure that min_td_duration > analysis_duration.

    Returns
    -------
    Dictionary:
        * plus: A numpy array.
            Strain values of the plus polarized WF in Frequency Domain..
        * cross: A numpy array.
            Strain values of the cross polarized WF in Frequency Domain.

    """
    waveform_kwargs = {
        "reference_frequency": 20.0,
        "minimum_frequency": 20.0,
        "maximum_frequency": frequency_array[-1],
        "f_start": 10,
        "sample_rate": 4096,
        "mode_array": [[2, 1], [2, 2], [3, 2], [3, 3], [4, 4]],
        "ecc_freq": 0,
        "ode_abstol": 1e-8,
        "ode_reltol": 1e-7,
        "min_td_duration": 1 / (frequency_array[1] - frequency_array[0]),
        "verbose": False,
    }
    waveform_kwargs.update(kwargs)
    waveform_kwargs["mode_array"] = [
        [int(a[0]), int(a[1])] for a in waveform_kwargs["mode_array"]
    ]  # Bilby treats them as string so converting them to int manually.
    try:
        # Convert J-Frame params to L-frame params
        lframe_params = gwmat.gw_utils.jframe_to_l0frame(
            mass_1=mass_1,
            mass_2=mass_2,
            f_ref=waveform_kwargs["reference_frequency"],
            phi_ref=phase,
            theta_jn=theta_jn,
            phi_jl=phi_jl,
            a_1=a_1,
            a_2=a_2,
            tilt_1=tilt_1,
            tilt_2=tilt_2,
            phi_12=phi_12,
        )

        # generate TD TEOBResumS WF
        wfs_res = eccentric_teobresums_binary_black_hole_TD(
            mass_1,
            mass_2,
            luminosity_distance,
            lframe_params["spin1x"],
            lframe_params["spin1y"],
            lframe_params["spin1z"],
            lframe_params["spin2x"],
            lframe_params["spin2y"],
            lframe_params["spin2z"],
            lframe_params["inclination"],
            phase,
            ecc,
            anomaly,
            **waveform_kwargs,
        )

        res = {}
        for k, wf in wfs_res.items():
            ## converting TD WF -> FD WF
            fd_wf = wf.to_frequencyseries(delta_f=wf.delta_f)
            ## sub-resampling from the full freqeuncy array to match what Bilby requires
            df = frequency_array[1] - frequency_array[0]
            delta_f_ratio = df / fd_wf.delta_f

            # ensure that delta_f_ratio is an integer.
            assert (
                abs(delta_f_ratio - np.floor(delta_f_ratio)) < 1e-7
            ), f"The required delta_f {df:.3f} is not a multiple \
of TEOBResumS output delta_f {fd_wf.delta_f:.5f}."  ## may not be strictly zero when duration is not a power of 2 because of precision issues.
            # sub-sampling
            new_fd_wf = fd_wf[:: round(delta_f_ratio)][: len(frequency_array)]
            # ensure the sub-sampled frequency array is same as what Bilby requires.
            assert np.allclose(
                frequency_array, new_fd_wf.sample_frequencies.data
            ), "Bilby frequency array does not match the TEOBResumS output frequency array."
            # assign the waveform data to res
            fd_wf_arr = new_fd_wf.data
            res[k] = fd_wf_arr
        return res
    except Exception as e:
        failed_pars = dict(
            m1=mass_1,
            m2=mass_2,
            dL=luminosity_distance,
            a_1=a_1,
            tilt_1=tilt_1,
            phi_12=phi_12,
            a_2=a_2,
            tilt_2=tilt_2,
            phi_jl=phi_jl,
            theta_jn=theta_jn,
            phase=phase,
            ecc=ecc,
            anomaly=anomaly,
        )
        if waveform_kwargs["verbose"]:
            print(f"[Waveform ERROR] {e} for pars:{failed_pars}; --returning None.")
        else:
            print(f"[Waveform ERROR] {e}; --returning None.")
        return None


## FD Microlensed TEOBResumS Source model for Bilby
# Define a global variable to hold the lookup table
LOOKUP_TABLE = None


def eccentric_aligned_spin_teobresums_point_lens_microlensing_binary_black_hole(
    frequency_array,
    mass_1,
    mass_2,
    luminosity_distance,
    chi_1z,
    chi_2z,
    theta_jn,
    phase,
    ecc,
    anomaly,
    Log_Mlz,
    yl,
    **kwargs,
):
    """
    This is a Bilby Frequency Domain Source model for performing parameter estimations
    with TOBResumS waveforms, including eccentricity and microlensing.

    It returns FD plus and cross polarized WFs
    interpolated at the required frequency_array.

    The arguments in the kwargs can be updated through `waveform-arguments-dict`
    in the config ini file of bilby-pipe. For example, the default values are:
    waveform-arguments-dict = {f_start=10, ecc_freq=0, sample_rate=2048,
    mode_array=[[2, 1], [2, 2], [3, 2], [3, 3], [4, 4]],
    ode_abstol=1e-8, ode_reltol=1e-7}

    Parameters
    ----------
    * frequency_array : float
        Array of frequency values where the WF will be evaluated.
    * mass_1 : float
        The mass of the primary component object in the binary (in solar masses).
    * mass_2 : float
        The mass of the secondary component object in the binary (in solar masses).
    * chi1z : float
        The z component of the first binary component. Default = 0.
    * chi2z : float
        The z component of the second binary component. Default = 0.
    * ecc : float
        Eccentricity of the binary at a reference frequency of f_start.
    * Log_Mlz : float
        Redshifted Mass of the point-lens in log10 scale (in solar masses).
    * yl : float
        The dimensionless impact parameter between the lens and the source.
    * luminosity_distance : ({100.,float}), optional
        Luminosity distance to the binary (in Mpc).
    * theta_jn : float
        Inclination (rad), defined as the angle between the orbital
        angular momentum J(or, L) and the line-of-sight at the
        reference frequency. Default = 0.
    * phase : ({0.,float}), optional
        Coalesence phase of the binary (in rad).
    * kwargs: dictionary
        * f_start : ({10., float}), optional
            Reference frequency for defining eccentricity.
            This is also the starting frequency of the
            (2, 2) mode for waveform generation (in Hz).
        * ecc_freq : ({0, int}), optional
             Use periastron (0), average (1) or apastron (2)
             frequency for initial condition computation.
             Default = 2.
        * sample_rate : ({4096, int}), optional
            Sample rate for the TEOBResumS WF generation.
            It should be more than the "sampling-frequency"
            used by Bilby for PE. Ideally, twice the value works fine.
        * mode_array : ({[[2,2]], 2D list}), optional
            Mode array for the WF generation.
        * ode_abstol : ({1e-8, float}), optional
            Absolute numerical tolerance for ODE
            compuations used in TEOBResumS.
            Default=1e-8, which is more than the original
            default of 1e-13 for speeding up WF evaluations.
        * ode_reltol : ({1e-7, float}), optional
            Relative numerical tolerance for ODE
            compuations used in TEOBResumS.
            Default=1e-7, which is more than the original
            default of 1e-11 for speeding up WF evaluations.
        * min_td_duration : int, optional
            Minimum duration of time-domain WF.
            The larger it is, the better is the frequency resolution.
            Default is min_td_duration = analysis_duration passed in Bilby.
            If passing manually, ensure that min_td_duration > analysis_duration.
        * lookup_table_path: str
            Path to the lookup table containing pre-computed
            microlensing amplification factor values.

    Returns
    -------
    Dictionary:
        * plus: A numpy array.
            Strain values of the plus polarized WF in Frequency Domain..
        * cross: A numpy array.
            Strain values of the cross polarized WF in Frequency Domain.

    """
    # Load lookup table if no already loaded.
    global LOOKUP_TABLE
    if LOOKUP_TABLE is None:
        LOOKUP_TABLE = microlensing_source.load_lookup_table(
            lookup_table_path=kwargs["lookup_table_path"]
        )
    Ff_grid, ys_grid, ws_grid = LOOKUP_TABLE

    try:
        # get unlensed TEOBResumS WF
        ecc_ul_fd = eccentric_aligned_spin_teobresums_binary_black_hole_FD(
            frequency_array,
            mass_1,
            mass_2,
            luminosity_distance,
            chi_1z,
            chi_2z,
            theta_jn,
            phase,
            ecc,
            anomaly,
            **kwargs,
        )

        # add microlensing effects
        if Log_Mlz < -3:
            return ecc_ul_fd

        Mlz = np.power(10.0, Log_Mlz)
        Ff = microlensing_source.point_lens_Ff_lookup_table(
            fs=frequency_array,
            Mlz=Mlz,
            yl=yl,
            Ff_grid=Ff_grid,
            ys_grid=ys_grid,
            ws_grid=ws_grid,
        )
        # Since LAL waveforms are defined using the engineering convention of the Fourier
        # transform, while F(f) follows the physics convention by default, we instead need
        # to modify the waveform using the complex conjugate of F(f).
        Ff_eng = np.conj(Ff)
        lhp = Ff_eng * ecc_ul_fd["plus"]
        lhc = Ff_eng * ecc_ul_fd["cross"]
        return {"plus": lhp, "cross": lhc}
    except Exception as e:
        print(f"[Waveform ERROR] {e} -- returning None.")
        return None
