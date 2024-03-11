# Generate BBH Injections (SINGLES/ PAIRS) from a population and inject into detectors (H1, L1, V1)

import bilby
import pycbc
import pickle
import deepdish
import pycbc.psd
import pycbc.types
import gwpopulation
import pycbc.waveform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcdefaults()
plt.rcParams.update({"text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 16,
    "axes.formatter.use_mathtext": True,
    "axes.formatter.limits": (-3, 3)
}) 

# Reading in Hyperposterior Samples from the GWTC-3 catalog from https://zenodo.org/record/5655785

result = bilby.core.result.read_in_result('git_overlap/src/data/GWTC-3-population-data/analyses/PowerLawPeak/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json')

# Selecting the hyper-priors with maximum (log_likelihood + log_prior): MAP

n_map = np.argmax(result.posterior['log_likelihood']+result.posterior['log_prior'])
PP_params = {
    "alpha": result.posterior['alpha'][n_map],
    "beta": result.posterior['beta'][n_map],
    "mmin": 5,
    "mmax": 100,
    "lam": result.posterior['lam'][n_map],
    "mpp": result.posterior['mpp'][n_map],
    "sigpp": result.posterior['sigpp'][n_map],
    "delta_m": result.posterior['delta_m'][n_map],
    "mu_chi": result.posterior['mu_chi'][n_map],
    "sigma_chi": result.posterior['sigma_chi'][n_map],
    "xi_spin": result.posterior['xi_spin'][n_map],
    "sigma_spin": result.posterior['sigma_spin'][n_map],
    "lamb": result.posterior['lamb'][n_map],
    "amax": result.posterior['amax'][n_map]
}

# Using GWPOP models for interpolating the parameters

num_samples, num_inject = int(1e5), int(1e5)

p, _ = gwpopulation.conversions.convert_to_beta_parameters(PP_params)

mass = np.linspace(5, 100, num = num_samples)
q = np.linspace(0, 1, num = num_samples)
z = np.linspace(1e-3, 2.3, num = num_samples)
cos_tilt_1 = np.linspace(-1, 1, num = num_samples)
cos_tilt_2 = np.linspace(-1, 1, num = num_samples) 
a_1 = np.linspace(0, 0.8, num = num_samples)
a_2 = np.linspace(0, 0.8, num = num_samples)

mass_model = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution(mmin=5, mmax=100)   # Power Law + Peak model for the source mass
p_mass = mass_model.p_m1(
    dataset = pd.DataFrame(dict(mass_1 = mass)),
    alpha = p["alpha"],
    mmin = p["mmin"],
    mmax = p["mmax"],
    lam = p["lam"],
    mpp = p["mpp"],
    sigpp = p["sigpp"],
    delta_m = p["delta_m"]
)
p_q = mass_model.p_q(  
    dataset = pd.DataFrame(dict(mass_ratio = q, mass_1 = mass)),
    beta = p["beta"],
    mmin = p["mmin"],
    delta_m = p["delta_m"]
)
p_z = gwpopulation.models.redshift.PowerLawRedshift(z_max = 2.3).probability(   # Redshifted distance
    dataset = pd.DataFrame(dict(redshift = z)), lamb = p["lamb"]
)
p_a = gwpopulation.models.spin.iid_spin_magnitude_beta(
    dataset = pd.DataFrame(dict(a_1 = a_1, a_2 = a_2)),
    amax = p["amax"],
    alpha_chi = p["alpha_chi"],
    beta_chi = p["beta_chi"]
)
p_cos_tilt_1 = gwpopulation.models.spin.truncnorm(
    xx = cos_tilt_1, mu = 1, sigma = p["sigma_spin"], high = 1, low = -1
)
p_cos_tilt_2 = gwpopulation.models.spin.truncnorm(
    xx = cos_tilt_2, mu = 1, sigma = p["sigma_spin"], high = 1, low = -1
)

# Generating the priors dictionary for the injection parameters

priors = bilby.prior.PriorDict(
    dict(
        mass_1_source = bilby.core.prior.Interped( #Primary mass of the binary in source frame in solar masses
            mass,
            p_mass,
            minimum = 5,
            maximum = 100,
            name = "mass_1_source",
            latex_label = "$m_{1}_s$",
        ),
        mass_ratio = bilby.core.prior.Interped( #Mass ratio of the primary and secondary binaries
            q,
            p_q,
            minimum = 0,
            maximum = 1,
            name = "mass_ratio",
            latex_label = "$q$",
        ),
        a_1 = bilby.core.prior.Interped( #The spin magnitude of the first binary component
            a_1,
            p_a,
            minimum = 0,
            maximum = 1,
            name = "a_1",
            latex_label = "$a_1$",
        ),
        a_2 = bilby.core.prior.Interped( #The spin magnitude of the second binary component
            a_2,
            p_a,
            minimum = 0,
            maximum = 1,
            name = " a_2",
            latex_label = "$a_2$",
        ),
        redshift = bilby.core.prior.Interped( #Redshift parameter
            z,
            p_z,
            minimum = 0,
            maximum = 2.3,
            name = "redshift",
            latex_label = "$z$",
        ),
        cos_tilt_1 = bilby.core.prior.Interped( #The cosine of the angle between the spin of the first binary component and the orbital angular momentum vector
            cos_tilt_1,
            p_cos_tilt_1,
            minimum = -1,
            maximum = 1,
            name = "cos_tilt_1",
            latex_label = "$\\cos\\theta_1$",
        ),
        cos_tilt_2 = bilby.core.prior.Interped( #The cosine of the angle between the spin of the second binary component and the orbital angular momentum vector
            cos_tilt_2,
            p_cos_tilt_2,
            minimum = -1,
            maximum = 1,
            name = "cos_tilt_2",
            latex_label = "$\\cos\\theta_2$",
        ),
        phi_12 = bilby.core.prior.Uniform( #The azimuthal angle between the spin of the heavier object and the spin of the lighter object, in radians
            name = "phi_12",
            minimum = 0,
            maximum = 2*np.pi,
            boundary = "periodic",
            latex_label = "$\\phi_{12}$",
        ),
        phi_jl = bilby.core.prior.Uniform( #The azimuthal angle between the total angular momentum vector and the orbital angular momentum vector, in radians
            name = "phi_jl",
            minimum = 0,
            maximum = 2*np.pi,
            boundary = "periodic",
            latex_label = "$\\phi_{jl}$",
        ),
        cos_theta_jn = bilby.core.prior.Uniform( #The angle between the total angular momentum vector and the line of sight, in radians
            name = "cos_theta_jn",
            minimum = -1,
            maximum = 1,
            boundary = "periodic",
            latex_label = "$\\cos\\theta_{jn}$",
        ),  
        ra = bilby.core.prior.Uniform( #The right ascension of the binary system, in radians
            name = "ra", 
            minimum = 0, 
            maximum = 2*np.pi, 
            boundary = "periodic",
            latex_label = "$\\alpha$",
        ),
        dec = bilby.core.prior.Cosine( #The declination of the binary system, in radians.
            name = "dec", 
            latex_label = "$\\delta$",
        ),
        psi = bilby.core.prior.Uniform( #The polarization angle of the gravitational wave.
            name = "psi",
            minimum = 0,
            maximum = np.pi,
            boundary = "periodic",
            latex_label = "$\\psi$",
        ),
        phase = bilby.core.prior.Uniform( #The phase of the gravitational wave at coalescence
            name = "phase",
            minimum = 0,
            maximum = 2*np.pi,
            boundary = "periodic",
            latex_label = "$\\phi_{c}$",
        ),
        incl = bilby.core.prior.Uniform( #The angle between the orbital angular momentum vector and the line of sight vector, in radians
            name = "incl",
            minimum = 0,
            maximum = np.pi,
            boundary = "periodic",
            latex_label = "$\\iota$",
        ),
        cos_theta_zn = bilby.core.prior.Uniform( #The angle between the line of sight and the normal to the orbital plane, in radians
            name = "cos_theta_zn",
            minimum = -1,
            maximum = 1,
            boundary = "periodic",
            latex_label = "$\\cos\\theta_{zn}$",
        )
    )
)

# Creating an injection data frame and saving the files

injections = pd.DataFrame(priors.sample(num_inject)).to_dict("list")

injections["mass_1"] = injections["mass_1_source"] * (1 + np.array(injections.get("redshift", [])))   # Detector Frame Mass
injections["mass_2"] = injections["mass_1"] * injections["mass_ratio"]
injections["luminosity_distance"] = bilby.gw.conversion.redshift_to_luminosity_distance(injections.get("redshift", []))/4   # Luminosity Distance

injections["tilt_1"] = np.arccos(injections["cos_tilt_1"])
injections["tilt_2"] = np.arccos(injections["cos_tilt_2"])
injections['theta_jn'] = np.arccos(injections["cos_theta_jn"])
injections['theta_zn'] = np.arccos(injections["cos_theta_zn"])

injections["geocent_time"] = 1200000000

pd.DataFrame(injections).to_hdf('git_overlap/src/output/injections/injections.hdf5', key='injections')

# Setting some constants

delta_f = 1
duration = 100
minimum_frequency = 20
reference_frequency = 50   # Most sensitive frequency
maximum_frequency = 1024
sampling_frequency = 2048

def psd_gen(det):
    """Reading the PSD files for the detectors"""

    if det == 'H1':
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt', sampling_frequency, delta_f, minimum_frequency, is_asd_file=True)
    if det == 'L1':
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt', sampling_frequency, delta_f, minimum_frequency, is_asd_file=True)
    if det == 'V1':
        psd = pycbc.psd.read.from_txt('git_overlap/src/psds/O3-V1_sensitivity_strain_asd.txt', sampling_frequency, delta_f, minimum_frequency, is_asd_file=True)

    return psd

# Selecting SNRs > 8

def snr(mass_1, mass_2, a_1, a_2, luminosity_distance, ra, dec, psi, incl, phase, geocent_time):
    """Evaluates the matched filter signal-to-noise ratio (through PyCBC) for the set of waveforms generated by the injection parameters"""

    hp, hc = pycbc.waveform.get_fd_waveform(approximant = 'IMRPhenomPv2', mass1 = mass_1, mass2 = mass_2, distance = luminosity_distance, spin1x = a_1, spin2x = a_2, 
                                            inclination = incl, coa_phase = phase, delta_f = delta_f, f_lower = minimum_frequency, f_final = maximum_frequency)    # Generate waveform 
     
    snrs = dict() 
    for det in ['H1', 'L1', 'V1']:

        det_obj = pycbc.detector.Detector(det)   # Loading the detector

        Fp, Fc = det_obj.antenna_pattern(ra, dec, psi, geocent_time)   # Antenna Patterns
        h = Fp*hp + Fc*hc   # h = F+*h+ + Fx*hx

        snrs[det] = pycbc.filter.matchedfilter.sigma(h, psd = psd_gen(det), low_frequency_cutoff = minimum_frequency, high_frequency_cutoff = maximum_frequency)   # Matched Filter SNR
    
    return snrs['H1'], snrs['L1'], snrs['V1']

injection = dict(deepdish.io.load('git_overlap/src/output/injections/injections.hdf5')['injections'])   # Loading the injections
for key, val in injection.items():   # Setting the variables
    exec(key + '=val')
mchirp = (mass_1*mass_2)**(3/5)/(mass_1+mass_2)**(1/5)

# Determining injection parameters with matched filter SNR > SNR threshold = 8

snr_H1, snr_L1, snr_V1 = np.vectorize(snr)(mass_1, mass_2, a_1, a_2, luminosity_distance, ra, dec, psi, incl, phase, geocent_time)   # Calculating the SNR
snr_det = np.sqrt(snr_H1**2 + snr_L1**2 + snr_V1**2)
det_idx = np.where(snr_det >= 8)[0]   # Setting SNR threshold
np.savetxt('git_overlap/src/output/injections/snr.csv', np.column_stack((det_idx, snr_det[det_idx])), delimiter=',')

idx = np.random.choice(det_idx, 2, replace = False) 

injection_parameters_a = {}
for key, val in injection.items():
    injection_parameters_a[key] = val[idx[0]]   # Updating the variables

injection_parameters_b = {}
for key, val in injection.items():
    injection_parameters_b[key] = val[idx[1]]   # Updating the variables
injection_parameters_b['geocent_time'] += np.random.uniform(0,0.5)   # Strong Bias

start_time = injection_parameters_a['geocent_time']-duration+2

# Bilby's WaveformGenerator object to generate BBH waveforms

waveform_generator_a = bilby.gw.WaveformGenerator(duration = duration, sampling_frequency = sampling_frequency, start_time = start_time,
                                                  frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole, 
                                                  waveform_arguments = {'waveform_approximant': 'IMRPhenomPv2', 'reference_frequency': reference_frequency, 'minimum_frequency': minimum_frequency})

waveform_generator_b = bilby.gw.WaveformGenerator(duration=duration, sampling_frequency = sampling_frequency, start_time = start_time,
                                                  frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole, 
                                                  waveform_arguments = {'waveform_approximant': 'IMRPhenomPv2', 'reference_frequency': reference_frequency, 'minimum_frequency': minimum_frequency})

# Initializing the detectors as bilby interferometer objects with zero noise, and with GPS time around the geocenter time of the GW signal

ifos, ifos_a, ifos_b = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1']), bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1']), bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])   # Initialize Detectors

for det in [ifos, ifos_a, ifos_b]:
    for ifo in det:
        ifo.minimum_frequency, ifo.maximum_frequency  = minimum_frequency, sampling_frequency/2
    det.set_strain_data_from_zero_noise(sampling_frequency = sampling_frequency, duration = duration, start_time = start_time)

# Injecting the SINGLES GW signal into H1, L1, and V1 using bilby, and saving the parameters

ifos_a.inject_signal(waveform_generator = waveform_generator_a, parameters = injection_parameters_a)    # SINGLES A
with open('git_overlap/src/output/injections/Waveform A Meta Data.pkl', 'wb') as file:
    pickle.dump(ifos_a.meta_data, file) 

ifos_b.inject_signal(waveform_generator = waveform_generator_b, parameters = injection_parameters_b)    # SINGLES B
with open('git_overlap/src/output/injections/Waveform B Meta Data.pkl', 'wb') as file:
    pickle.dump(ifos_b.meta_data, file)

# Injecting the PAIRS GW signal into H1, L1, and V1 using bilby

ifos.inject_signal(waveform_generator = waveform_generator_a, parameters = injection_parameters_a)    # PAIRS (A)
ifos.inject_signal(waveform_generator = waveform_generator_b, parameters = injection_parameters_b)    # PAIRS (B)

# Extrapolating the strain data in the time domain

H1_strain_a, L1_strain_a, V1_strain_a = ifos_a[0].time_domain_strain, ifos_a[1].time_domain_strain, ifos_a[2].time_domain_strain    # SINGLES A
H1_strain_b, L1_strain_b, V1_strain_b = ifos_b[0].time_domain_strain, ifos_b[1].time_domain_strain, ifos_b[2].time_domain_strain    # SINGLES B
H1_strain, L1_strain, V1_strain = ifos[0].time_domain_strain, ifos[1].time_domain_strain, ifos[2].time_domain_strain    # PAIRS

# Generating PyCBC TimeSeries from the strain array

ht_a_H1, ht_a_L1, ht_a_V1 = pycbc.types.TimeSeries(H1_strain_a, delta_t = 1/sampling_frequency), pycbc.types.TimeSeries(L1_strain_a, delta_t = 1/sampling_frequency), pycbc.types.TimeSeries(V1_strain_a, delta_t = 1/sampling_frequency)    # SINGLES A
ht_b_H1, ht_b_L1, ht_b_V1 = pycbc.types.TimeSeries(H1_strain_b, delta_t = 1/sampling_frequency), pycbc.types.TimeSeries(L1_strain_b, delta_t = 1/sampling_frequency), pycbc.types.TimeSeries(V1_strain_b, delta_t = 1/sampling_frequency)    # SINGLES B
ht_H1, ht_L1, ht_V1 = pycbc.types.TimeSeries(H1_strain, delta_t = 1/sampling_frequency), pycbc.types.TimeSeries(L1_strain, delta_t = 1/sampling_frequency), pycbc.types.TimeSeries(V1_strain, delta_t = 1/sampling_frequency)    # PAIRS

# Setting the start times to the geocenter time and creating the dictionary of waveforms 

ht_a_H1.start_time, ht_a_L1.start_time, ht_a_V1.start_time = start_time, start_time, start_time    # SINGLES A
ht_b_H1.start_time, ht_b_L1.start_time, ht_b_V1.start_time = start_time, start_time, start_time    # SINGLES B
ht_H1.start_time, ht_L1.start_time, ht_V1.start_time = start_time, start_time, start_time    # PAIRS

ht_a, ht_b, ht = {'H1': ht_a_H1, 'L1': ht_a_L1, 'V1': ht_a_V1}, {'H1': ht_b_H1, 'L1': ht_b_L1, 'V1': ht_b_V1}, {'H1': ht_H1, 'L1': ht_L1, 'V1': ht_V1}

# Matplotlib rcParams

plt.style.use('default')
plt.rcParams.update({"text.usetex": True,
    "font.family": "sans-serif",
    "axes.formatter.use_mathtext": True,
    "axes.formatter.limits": (-3, 3)
})

# Plotting the waveforms

fig, ax = plt.subplots(1,3, figsize = (18, 5), sharex = True)
fig.suptitle('\\textbf{WAVEFORM INJECTIONS}')

ax[0].plot(ht_a['H1'].sample_times, ht_a['H1'], 'g-', linewidth=1, label='$\\rm{SINGLES_A}$')
ax[0].plot(ht_b['H1'].sample_times, ht_b['H1'], 'm-', linewidth=1, label='$\\rm{SINGLES_B}$')
ax[0].plot(ht['H1'].sample_times, ht['H1'], 'b--', label='$\\rm{PAIRS}$')
ax[0].set_xlabel('Time $[s]$')
ax[0].set_ylabel('Strain $h$')
ax[0].set_xlim(start_time+duration-2.4, start_time+duration-1.7)
ax[0].legend() 
ax[0].grid(True)
ax[0].set_title('H1')

ax[1].plot(ht_a['L1'].sample_times, ht_a['L1'], 'g-', linewidth=1, label='$\\rm{SINGLES_A}$')
ax[1].plot(ht_b['L1'].sample_times, ht_b['L1'], 'm-', linewidth=1, label='$\\rm{SINGLES_B}$')
ax[1].plot(ht['L1'].sample_times, ht['L1'], 'b--', label='$\\rm{PAIRS}$')
ax[1].set_xlabel('Time $[s]$')
ax[1].set_ylabel('Strain $h$')
ax[1].set_xlim(start_time+duration-2.4, start_time+duration-1.7)
ax[1].legend() 
ax[1].grid(True)
ax[1].set_title('L1')

ax[2].plot(ht_a['V1'].sample_times, ht_a['V1'], 'g-', linewidth=1, label='$\\rm{SINGLES_A}$')
ax[2].plot(ht_b['V1'].sample_times, ht_b['V1'], 'm-', linewidth=1, label='$\\rm{SINGLES_B}$')
ax[2].plot(ht['H1'].sample_times, ht['V1'], 'b--', label='$\\rm{PAIRS}$')
ax[2].set_xlabel('Time $[s]$')
ax[2].set_ylabel('Strain $h$')
ax[2].set_xlim(start_time+duration-2.4, start_time+duration-1.7)
ax[2].legend() 
ax[2].grid(True)
ax[2].set_title('V1')

fig.set_tight_layout(True)
plt.savefig('git_overlap/src/output/injections/INJECTIONS.png')
plt.close()