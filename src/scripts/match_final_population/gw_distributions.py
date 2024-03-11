
import sys
import bilby
import pycbc
import pickle
import deepdish
import gwpopulation
import numpy as np
import pandas as pd

# Reading in Hyperposterior Samples from the GWTC-3 catalog from https://zenodo.org/record/5655785

result = bilby.core.result.read_in_result('/home/nishkal.rao/git_overlap/src/data/GWTC-3-population-data/analyses/PowerLawPeak/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json')

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

num_samples, num_inject = int(2e6), int(2e6)

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
pd.DataFrame(injections).to_hdf('/home/nishkal.rao/git_overlap/src/output/match_final_population/injections/injections.hdf5', key='injections')
