# Simulate BBH SINGLES Injections

import bilby
import pycbc
import pickle
import deepdish
import pycbc.psd
import numpy as np
import gwpopulation
import pandas as pd
import pycbc.waveform
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'text.usetex' : True})

# Reading in Hyperposterior Samples from the GWTC-3 catalog from https://zenodo.org/record/5655785

result=bilby.core.result.read_in_result('git_overlap/src/data/GWTC-3-population-data/analyses/PowerLawPeak/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json')
alpha=result.posterior['alpha']
beta=result.posterior['beta']
lam=result.posterior['lam']
mpp=result.posterior['mpp']
sigpp=result.posterior['sigpp']
delta_m=result.posterior['delta_m']
mu_chi=result.posterior['mu_chi']
sigma_chi=result.posterior['sigma_chi']
xi_spin=result.posterior['xi_spin']
sigma_spin=result.posterior['sigma_spin']
lamb=result.posterior['lamb']
amax=result.posterior['amax']

# Selecting the hyper-priors with maximum (log_likelihood + log_prior): MAP

n_map=np.argmax(result.posterior['log_likelihood']+result.posterior['log_prior'])
PP_params = {
    "alpha": alpha[n_map],
    "beta": beta[n_map],
    "mmin": 5,
    "mmax": 100,
    "lam": lam[n_map],
    "mpp": mpp[n_map],
    "sigpp": sigpp[n_map],
    "delta_m": delta_m[n_map],
    "mu_chi": mu_chi[n_map],
    "sigma_chi": sigma_chi[n_map],
    "xi_spin": xi_spin[n_map],
    "sigma_spin": sigma_spin[n_map],
    "lamb": lamb[n_map],
    "amax": amax[n_map]
}

# GWPOP models

p, _ = gwpopulation.conversions.convert_to_beta_parameters(PP_params)

num_samples = 10000
mass = np.linspace(5, 100, num=num_samples)
q = np.linspace(0, 1, num=num_samples)
z = np.linspace(1e-3, 2.3, num=num_samples)
cos_tilt_1 = np.linspace(-1, 1, num=num_samples)
cos_tilt_2 = np.linspace(-1, 1, num=num_samples) 
a_1 = np.linspace(0, 0.8, num=num_samples)
a_2 = np.linspace(0, 0.8, num=num_samples)

mass_model = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution(mmin=5, mmax=100)   # Power Law + Peak model for the source mass
p_mass = mass_model.p_m1(
    dataset=pd.DataFrame(dict(mass_1=mass)),
    alpha=p["alpha"],
    mmin=p["mmin"],
    mmax=p["mmax"],
    lam=p["lam"],
    mpp=p["mpp"],
    sigpp=p["sigpp"],
    delta_m=p["delta_m"]
)
p_q = mass_model.p_q(  
    dataset=pd.DataFrame(dict(mass_ratio=q, mass_1=mass)),
    beta=p["beta"],
    mmin=p["mmin"],
    delta_m=p["delta_m"]
)
p_z = gwpopulation.models.redshift.PowerLawRedshift(z_max=2.3).probability(   # Redshifted distance
    dataset=pd.DataFrame(dict(redshift=z)), lamb=p["lamb"]
)
p_a = gwpopulation.models.spin.iid_spin_magnitude_beta(
    dataset=pd.DataFrame(dict(a_1=a_1, a_2=a_2)),
    amax=p["amax"],
    alpha_chi=p["alpha_chi"],
    beta_chi=p["beta_chi"]
)
p_cos_tilt_1 = gwpopulation.models.spin.truncnorm(
    xx=cos_tilt_1, mu=1, sigma=p["sigma_spin"], high=1, low=-1
)
p_cos_tilt_2 = gwpopulation.models.spin.truncnorm(
    xx=cos_tilt_2, mu=1, sigma=p["sigma_spin"], high=1, low=-1
)

# Generating the priors dictionary

priors = bilby.prior.PriorDict(
    dict(
        mass_1_source=bilby.core.prior.Interped( #Primary mass of the binary in source frame
            mass,
            p_mass,
            minimum=5,
            maximum=100,
            name="mass_1_source",
            latex_label="$m_{1}$",
        ),
        mass_ratio=bilby.core.prior.Interped( #Mass ratio of the binaries
            q,
            p_q,
            minimum=0,
            maximum=1,
            name="mass_ratio",
            latex_label="$q$",
        ),
        redshift=bilby.core.prior.Interped( #Redshifted distance
            z,
            p_z,
            minimum=0,
            maximum=2.3,
            name="redshift",
            latex_label="$pred_z$",
        ),
        psi=bilby.core.prior.Uniform( #The polarization angle as defined with respect to the total angular momentum
            name="psi", 
            minimum=0, 
            maximum=np.pi, 
            boundary="periodic",
            latex_label="$\\psi_j$",
        ),
        a_1=bilby.core.prior.Interped( #The spin magnitude on the larger object
            a_1,
            p_a,
            minimum=0,
            maximum=1,
            name="a_1",
            latex_label="$a_1$",
        ),
        a_2=bilby.core.prior.Interped( #The spin magnitude on the secondary object
            a_2,
            p_a,
            minimum=0,
            maximum=1,
            name="a_2",
            latex_label="$a_2$",
        ),
        cos_tilt_1=bilby.core.prior.Interped( #The angle between the total orbital angular momentum and the primary spin
            cos_tilt_1,
            p_cos_tilt_1,
            minimum=-1,
            maximum=1,
            name="cos_tilt_1",
            latex_label="$\\cos\ \\mathrm{tilt}_1$",
        ),
        cos_tilt_2=bilby.core.prior.Interped( #The angle between the total orbital angular momentum and the secondary spin
            cos_tilt_2,
            p_cos_tilt_2,
            minimum=-1,
            maximum=1,
            name="cos_tilt_2",
            latex_label="$\\cos\ \\mathrm{tilt}_2$",
        ),
        phi_12=bilby.core.prior.Uniform( #The angle between the primary spin and the secondary spin
            name="phi_12",
            minimum=0,
            maximum=2 * np.pi,
            boundary="periodic",
            latex_label="$\\phi_{12}$",
        ),
        cos_theta_jn=bilby.core.prior.Uniform( #The angle between the total orbital angular momentum and the line of sight
            name="cos_theta_jn",
            minimum=-1,
            maximum=1,
            boundary="periodic",
            latex_label="$\\cos\\theta_{jn}$",
        ),
        ra=bilby.core.prior.Uniform( #The right ascension of the source
            name="ra", 
            minimum=0, 
            maximum=2 * np.pi, 
            boundary="periodic",
            latex_label="$ra$",
        ),
        dec=bilby.core.prior.Cosine( #The declination of the source
            name="dec", 
            latex_label="$dec$",
        ),
        phi_jl=bilby.core.prior.Uniform( #The precession phase
            name="phi_jl",
            minimum=0,
            maximum=2 * np.pi,
            boundary="periodic",
            latex_label="$\\phi_{jl}$",
        ),
        incl=bilby.core.prior.Uniform( #The source inclination angle
            name="incl", 
            minimum=0, 
            maximum=2 * np.pi, 
            boundary="periodic",
            latex_label="$\\iota$",  
        ),
        phase=bilby.core.prior.Uniform( #The coalescence phase
            name="phase", 
            minimum=0, 
            maximum=2 * np.pi, 
            boundary="periodic",
            latex_label="$\\phi$",  
        )
    )
)

# Creating an injection data frame

num_inject = int(5e3)

injections = pd.DataFrame(priors.sample(num_inject)).to_dict("list")
injections["mass_1"] = injections["mass_1_source"] * (1 + np.array(injections.get("redshift", [])))   # Detector Frame Mass
injections["mass_2"] = injections["mass_1"] * injections["mass_ratio"]
injections["luminosity_distance"] = bilby.gw.conversion.redshift_to_luminosity_distance(injections.get("redshift", []))   # Luminosity Distance
injections["tilt_1"] = np.arccos(injections["cos_tilt_1"])
injections["tilt_2"] = np.arccos(injections["cos_tilt_2"])
injections['theta_jn'] = np.arccos(injections["cos_theta_jn"])
injections["geocent_time"] = 1200000000

samples = pd.DataFrame(injections)
samples.to_hdf('git_overlap/src/output/overlap_injection/injections.hdf5', key='injections')

delta_f = 1
duration = 1000   # 10 days
minimum_frequency = 20
reference_frequency = 50    # Most sensitive frequency
maximum_frequency = 1024
sampling_frequency = int(maximum_frequency/delta_f)

# Reading the PSD files

psd_H1 = pycbc.psd.read.from_txt('git_overlap/src/psds/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt', sampling_frequency, delta_f, minimum_frequency, is_asd_file=True)
psd_L1 = pycbc.psd.read.from_txt('git_overlap/src/psds/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt', sampling_frequency, delta_f, minimum_frequency, is_asd_file=True)
psd_V1 = pycbc.psd.read.from_txt('git_overlap/src/psds/O3-V1_sensitivity_strain_asd.txt', sampling_frequency, 1, minimum_frequency, is_asd_file=True)

# Selecting SNRs > 8

def snr(mass_1, mass_2, a_1, a_2, luminosity_distance, ra, dec, psi, incl, phase, geocent_time):
    
    hp, hc = pycbc.waveform.get_fd_waveform(approximant='IMRPhenomPv2', mass1=mass_1, mass2=mass_2, distance=luminosity_distance, spin1x=a_1, spin2x=a_2, 
                                            inclination=incl, coa_phase=phase, delta_f=delta_f, f_lower=minimum_frequency, f_final=maximum_frequency)    # Generate waveform 
 
    snr = np.zeros(3)   # Initializing SNR array
    psd_list = [psd_H1, psd_L1, psd_V1]   # Importing the PSDs
    det_obj_list = [pycbc.detector.Detector('H1'), pycbc.detector.Detector('L1'), pycbc.detector.Detector('V1')]   # Setting the detectors
    
    for i, det_obj in enumerate(det_obj_list):
        Fp, Fc = det_obj.antenna_pattern(ra, dec, psi, geocent_time)   # Antenna Patterns
        h = Fp*hp + Fc*hc   # h = F+*h+ + Fx*hx
        snr[i] = pycbc.filter.matchedfilter.sigma(h, psd=psd_list[i], low_frequency_cutoff=minimum_frequency, high_frequency_cutoff=maximum_frequency)   # Matched Filter SNR
    return snr[0], snr[1], snr[2]

injection = dict(deepdish.io.load('git_overlap/src/output/overlap_injection/injections.hdf5')['injections'])   # Loading the injections

for key, val in injection.items():   # Setting the variables
    exec(key + '=val')

snr_H1, snr_L1, snr_V1 = np.vectorize(snr)(mass_1, mass_2, a_1, a_2, luminosity_distance, ra, dec, psi, incl, phase, geocent_time)   # Calculating the SNR
det_idx = np.where(np.sqrt(snr_H1**2 + snr_L1**2 + snr_V1**2) >= 8)[0]   # Setting SNR threshold

# Generating injections for SINGLES A & B

injection_parameters_a = {}
for key, val in injection.items():
    injection_parameters_a[key] = val[det_idx[np.random.randint(len(det_idx))]]   # Updating the variables

injection_parameters_b = {}
for key, val in injection.items():
    injection_parameters_b[key] = val[det_idx[np.random.randint(len(det_idx))]]   # Updating the variables

# WaveformGenerator object to generate BBH waveforms

waveform_generator_a = bilby.gw.WaveformGenerator(duration=duration, sampling_frequency=sampling_frequency, 
                                                  start_time=injection_parameters_a['geocent_time']- duration + 2,
                                                  frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole, 
                                                  waveform_arguments={'waveform_approximant': 'IMRPhenomPv2',
                                                                      'reference_frequency': reference_frequency,
                                                                      'minimum_frequency': minimum_frequency})

waveform_generator_b = bilby.gw.WaveformGenerator(duration=duration, sampling_frequency=sampling_frequency, 
                                                  start_time=injection_parameters_b['geocent_time'] - duration + 2,
                                                  frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole, 
                                                  waveform_arguments={'waveform_approximant': 'IMRPhenomPv2',
                                                                      'reference_frequency': reference_frequency,
                                                                      'minimum_frequency': minimum_frequency})

ifos_a, ifos_b = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1']), bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])   # Initialize Detectors

# Bilby Interferometer object with GPS time around the geocenter time of the GW signal. 
# Gaussian noise background with the PSD being their design sensitivity.

for ifo in ifos_a:
    ifo.minimum_frequency, ifo.maximum_frequency = minimum_frequency, sampling_frequency/2
ifos_a.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency, duration=duration, start_time=injection_parameters_a['geocent_time']- duration + 2)

for ifo in ifos_b:
    ifo.minimum_frequency, ifo.maximum_frequency  = minimum_frequency, sampling_frequency/2
ifos_b.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency, duration=duration, start_time=injection_parameters_b['geocent_time']  - duration + 2)

# Injecting GW signal into H1, L1, and V1

ifos_a.inject_signal(waveform_generator=waveform_generator_a, parameters=injection_parameters_a)
with open('git_overlap/src/output/overlap_injection/Waveform A Meta Data.pkl', 'wb') as file:
    pickle.dump(ifos_a.meta_data, file) 

ifos_b.inject_signal(waveform_generator=waveform_generator_b, parameters=injection_parameters_b)
with open('git_overlap/src/output/overlap_injection/Waveform B Meta Data.pkl', 'wb') as file:
    pickle.dump(ifos_b.meta_data, file) 

# Examining the signals in the time domain

polarizations_td_a, polarizations_td_b = waveform_generator_a.time_domain_strain(injection_parameters_a), waveform_generator_b.time_domain_strain(injection_parameters_b)
plus_td_a, cross_td_a = np.roll(polarizations_td_a['plus'], int(sampling_frequency * 2)), np.roll(polarizations_td_a['cross'], int(sampling_frequency * 2))
plus_td_b, cross_td_b = np.roll(polarizations_td_b['plus'], int(sampling_frequency * 2)), np.roll(polarizations_td_b['cross'], int(sampling_frequency * 2))
time_a, time_b = np.linspace(0, duration, len(plus_td_a)), np.linspace(0, duration, len(plus_td_b))

# Plotting the waveforms

fig, ax = plt.subplots(1,2, figsize=(12, 6))
fig.suptitle('\\textbf{SINGLES A and B}')
ax[0].plot(time_a, plus_td_a, label='$h_{+}$')
ax[0].plot(time_a, cross_td_a, label='$h_{\\times}$')
ax[0].set_xlim(1.5, 2.1)
ax[0].set_xlabel('Time $t$')
ax[0].set_ylabel('Strain $h$')
ax[0].set_title('SINGLES A')
ax[0].legend()
ax[1].plot(time_b, plus_td_b, label='$h_{+}$')
ax[1].plot(time_b, cross_td_b, label='$h_{\\times}$')
ax[1].set_xlim(1.5, 2.1)
ax[1].set_xlabel('Time $t$')
ax[1].set_ylabel('Strain $h$')
ax[1].set_title('SINGLES B')
ax[1].legend()
plt.savefig('git_overlap/src/output/overlap_injection/POLARIZATIONS: SINGLES A & B.png')
plt.close()