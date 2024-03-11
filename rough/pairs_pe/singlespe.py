import math
import bilby
import scipy
import pycbc
import pickle
import deepdish
import gwpopulation

import numpy as np
import pandas as pd
import scipy.signal as sig
import matplotlib.pyplot as plt

result=bilby.core.result.read_in_result('Parameter Estimation_Overlap/Overlap_Injection/GWTC-3-population-data/analyses/PowerLawPeak/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json')
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

params = PP_params.copy()
p, _ = gwpopulation.conversions.convert_to_beta_parameters(params)

num_samples = 10000
mass = np.linspace(5, 100, num=num_samples)
q = np.linspace(0, 1, num=num_samples)
z = np.linspace(1e-3, 2.3, num=num_samples)
cos_tilt_1 = np.linspace(-1, 1, num=num_samples)
cos_tilt_2 = np.linspace(-1, 1, num=num_samples) 
a_1 = np.linspace(0, 0.8, num=num_samples)
a_2 = np.linspace(0, 0.8, num=num_samples)

mass_model = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution(mmin=5, mmax=100)
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
p_z = gwpopulation.models.redshift.PowerLawRedshift(z_max=2.3).probability(
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

priors = bilby.prior.PriorDict(
    dict(
        mass_1_source=bilby.core.prior.Interped( #Primary mass of the bianry in source frame
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

duration = 1000
minimum_frequency = 20
sampling_frequency = 4096

waveform_arguments = {
    'waveform_approximant': 'IMRPhenomPv2', #'SEOBNRv4PHM',
    'reference_frequency': 50,
    'minimum_frequency': 2
}

waveform_metadata_a=pickle.load(open('Parameter Estimation_Overlap/Overlap_Injection/Output/Waveform A Meta Data.pkl', 'rb'))

waveform_generator_a = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency, start_time=waveform_metadata_a['H1']['parameters']['geocent_time']- duration + 2,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=waveform_arguments)

ifos_a = bilby.gw.detector.InterferometerList(['H1', 'L1'])

for ifo in ifos_a:
    ifo.minimum_frequency = minimum_frequency
    ifo.maximum_frequency = sampling_frequency/2
    ifos_a.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=waveform_metadata_a['H1']['parameters']['geocent_time']- duration + 2)
ifos_a.inject_signal(waveform_generator=waveform_generator_a, parameters=waveform_metadata_a['H1']['parameters'])

waveform_metadata_b=pickle.load(open('Parameter Estimation_Overlap/Overlap_Injection/Output/Waveform B Meta Data.pkl', 'rb'))

waveform_generator_b = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency, start_time=waveform_metadata_b['H1']['parameters']['geocent_time']- duration + 2,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=waveform_arguments)

ifos_b = bilby.gw.detector.InterferometerList(['H1', 'L1'])

for ifo in ifos_b:
    ifo.minimum_frequency = minimum_frequency
    ifo.maximum_frequency = sampling_frequency/2
    ifos_b.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=waveform_metadata_b['H1']['parameters']['geocent_time']- duration + 2)
ifos_b.inject_signal(waveform_generator=waveform_generator_b, parameters=waveform_metadata_b['H1']['parameters']);

priors['geocent_time']=waveform_metadata_a['H1']['parameters']['geocent_time']
likelihood_a = bilby.gw.GravitationalWaveTransient(interferometers=ifos_a, 
                                                   waveform_generator=waveform_generator_a)

result_a = bilby.run_sampler(
    likelihood_a, priors, sampler='dynesty', outdir='Parameter Estimation_Overlap/Overlap_PE/Singles_PE_A', label='PE',
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters
)
result_a.plot_corner()
plt.show()
plt.close()

priors['geocent_time']=waveform_metadata_b['H1']['parameters']['geocent_time']
likelihood_b = bilby.gw.GravitationalWaveTransient(interferometers=ifos_b, 
                                                   waveform_generator=waveform_generator_b)

result_b = bilby.run_sampler(
    likelihood_b, priors, sampler='dynesty', outdir='Parameter Estimation_Overlap/Overlap_PE/Singles_PE_B', label='PE',
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters
)
result_b.plot_corner()
plt.show() 
plt.close()
