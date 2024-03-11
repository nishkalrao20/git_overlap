#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("/data/gravwav/bgadre/microlensing/src/gwmat/src/")

import py_lgw
lgw = py_lgw.lensed_wf_gen()
import time
import logging
import os
import argparse

import h5py
import numpy as np
import lalsimulation as lalsim
import pycbc
import pycbc.filter as pf
import pycbc.inject as _inject
import pycbc.pnutils as pnu
import pycbc.psd as pp
import pycbc.waveform as pw
import pyswarms as ps
import pyswarms.backend.topology as topo
from pyswarms.single.global_best import GlobalBestPSO
from scipy.optimize import differential_evolution, minimize


def get_microlense_waveform(mass1=30, mass2=20,
                          spin1x=0., spin1y=0., spin1z=0.,
                          spin2x=0., spin2y=0., spin2z=0.,
                          coa_phase=0., inclination=0., distance=1,
                          delta_f=1./32., delta_t=0.5/1024., f_lower=20., approximant='ML', wf_approximant='IMRPhenomXAS', mass_lens=None, impact_parameter=None,
                          **kwargs):

    lens_prms = dict(m_lens=mass_lens if mass_lens else 0., y_lens=impact_parameter if impact_parameter else 1., z_lens=0)
    cbc_params = dict(mass_1=mass1, mass_2=mass2, spin1z=spin1z, spin2z=spin2z, inclination=inclination, coa_phase=coa_phase, luminosity_distance=distance, f_start=f_lower, f_ref=kwargs.get('f_ref', f_lower), delta_t=delta_t, delta_f=delta_f, wf_approximant=wf_approximant)
    print(cbc_params, lens_prms)
    res = lgw.lensed_pure_polarized_wf_gen(**{**lens_prms, **cbc_params})
    for k in res:
        print(k, res[k])
    return res['lensed_FD_WF_hp'], res['lensed_FD_WF_hc']

pw.add_custom_waveform('ML', get_microlense_waveform, 'frequency')

#  if __name__ == '__main__':
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--injection-fraction-range',
                    default='0/1',
                    help='Optional, analyze only a certain range of the '
                    'injections. Format PART/NUM_PARTS')

opts = parser.parse_args()

## FIXME: Should be argparsed!
verbose = True
pycbc.init_logging(verbose)
disp = True  ## This sets verbose option for optimizer output
polish = True  ## This sets verbose option for optimizer output

ninj_write_out = 1  ## We will write output after every 10 injections.
injection_fraction_range = opts.injection_fraction_range
injection_file = 'align_ecc_SEOBHM_injection.hdf'

inj_approx = 'ML'
inj_wf_approx = 'IMRPhenomXAS'
rec_approx = 'IMRPhenomXAS'
outfile = '{}_ff_SEOBNRv4E_vs_v4_ROM_part{}.txt'

fmin = 20  ## Hz
fmin_match = 20.
fmax = 2048
inj_fmax = 2048
delta_t = 0.5 / fmax
mr_dur = 2.
postlen = int(mr_dur / delta_t)  ## WE add 2 sec to allow
taper = 'startend'
ifo = 'H1'
scale = 1. / pycbc.DYN_RANGE_FAC

m1_min, m1_max = 2.5, 55
m2_min, m2_max = 2.5, 55
s1_min, s1_max = -0.99, 0.99
s2_min, s2_max = -0.99, 0.99

dur = pnu._get_imr_duration(m1_min, m2_min, s1_max, s2_max, fmin,
                            'SPAtmplt') + mr_dur
dur = np.ceil(
    np.log2(dur)) if np.log2(dur) - int(np.log2(dur)) > 0 else np.log2(dur)
seg_len = 2 * 2**dur

## We assume m2 <= m1 unless specified otherwise
## Figure out the bound

mc_min, eta_max = pnu.mass1_mass2_to_mchirp_eta(m1_min, m2_min)
mc_max, _ = pnu.mass1_mass2_to_mchirp_eta(m1_max, m2_max)
_, eta_min = pnu.mass1_mass2_to_mtotal_eta(80, 3)

ub = [mc_max, eta_max, s1_max, s2_max]
lb = [mc_min, eta_min, s1_min, s2_min]

bounds = np.asanyarray([(l, u) for l, u in zip(lb, ub)])

tlen = int(seg_len / delta_t)
flen = tlen // 2 + 1
delta_f = 1. / seg_len

# psd = pp.aLIGOaLIGODesignSensitivityT1800044(flen, delta_f, 15.)
psd = pp.aLIGOAdVO3LowT1800545(flen, delta_f, 15.)
psd.data[-1] = psd.data[-2]
psd.data /= scale**2

kmin, kmax = pf.get_cutoff_indices(20., fmax, delta_f, tlen)
# print('psd.data')
# print(psd.data[kmin:])

##########################


def funcmin(rec_param, stilde, psd, f_min, f_max, approx, snorm):
    assert stilde.delta_f - psd.delta_f < 1e-6, f"psd and stilde to not have same df values to be compatible as {stilde.delta_f}, {psd.delta_f}"

    mchirp, eta, s1zr, s2zr = rec_param[0], rec_param[1], rec_param[
        2], rec_param[3]
    m1r, m2r = pnu.mchirp_eta_to_mass1_mass2(mchirp, eta)
    try:
        hrp, _ = gen_waveform(rec_param, f_min, f_max, psd.delta_f, approx)

    except RuntimeError:
        print(m1r, m2r, s1zr, s2zr)
        raise RuntimeError(
            "Waveform generation failed for the above parameters")

    maxsnr, maxtid = pf.match(hrp,
                              stilde,
                              psd=psd,
                              low_frequency_cutoff=f_min,
                              high_frequency_cutoff=f_max,
                              v2_norm=snorm)
    funcmin.max_id = maxtid  ## Just to check time delay with injected time if needed
    # print(maxsnr, maxtid)
    return 1 - maxsnr


def gen_waveform(rec_param,
                 f_min,
                 f_max,
                 delta,
                 approx,
                 tlen=None,
                 taper=None):
    '''we are using chirp mass and symmetric mass ratio as search parameters.
    We always return FS and not TS.
    '''
    mchirp, eta, s1zr, s2zr = rec_param[0], rec_param[1], rec_param[
        2], rec_param[3]
    m1r, m2r = pnu.mchirp_eta_to_mass1_mass2(mchirp, eta)

    if approx in pw.fd_approximants():
        hrp, hrc = pw.get_fd_waveform(approximant=approx,
                                      mass1=m1r,
                                      mass2=m2r,
                                      spin1z=s1zr,
                                      spin2z=s2zr,
                                      delta_f=delta,
                                      f_lower=fmin,
                                      f_final=fmax,
                                      distance=scale)

    elif approx in pw.fd_approximants():
        hrp, hrc = pw.get_fd_waveform(approximant=approx,
                                      mass1=m1r,
                                      mass2=m2r,
                                      spin1z=s1zr,
                                      spin2z=s2zr,
                                      delta_f=delta,
                                      f_lower=fmin,
                                      distance=scale,
                                      f_final=fmax)

        ## Remember, always taper before resizing
        if taper:
            hrp = pw.taper_timeseries(hp, taper)
            hrc = pw.taper_timeseries(hc, taper)
        if tlen:
            hrp.resize(tlen)
            hrc.resize(tlen)
    else:
        raise ValueError(
            "Approximant {} not found. Invalid approxiamnts!".format(approx))

    return pf.make_frequency_series(hrp), pf.make_frequency_series(hrc)


def create_injection(inj, ifo, inj_fmin, inj_fmax=None, approx=None, taper=None, distance_scale=1, **kwargs):
    if approx:
        inj.approximant = approx
    if taper:
        inj.taper = taper

    if inj_fmax is not None:
        idelta_t = 0.5 / inj_fmax
    else:
        idelta_t = delta_t
    strain = injections.make_strain_from_inj_object(inj, idelta_t, ifo, inj_fmin,
                                                    distance_scale)
    if strain:
        strain = pf.resample_to_delta_t(strain, delta_t, method='ldas')
        strain.resize(tlen - postlen)
        strain.prepend_zeros(postlen)
        stilde = strain.to_frequencyseries()
        return stilde
    else:
        return None


def parse_injection_range(num_inj, rangestr):
    part = int(rangestr.split('/')[0])
    pieces = int(rangestr.split('/')[1])
    tmin = num_inj * part // pieces
    tmax = num_inj * (part + 1) // pieces
    return tmin, tmax


logging.info("Reading injections")
# injections = _inject.InjectionSet(injection_file)
# inj_table = injections.table
#
# part = injection_fraction_range.split('/')[0]
#
# num_injections = len(inj_table)
#
# if 'simulation_id' not in inj_table:
#     inj_table = inj_table.add_fields(np.arange(num_injections),
#                                      'simulation_id')
# if 'taper' not in inj_table:
#     inj_table = inj_table.add_fields(np.repeat([taper], num_injections),
#                                      'taper')
#
# logging.info(
#     "Reading injections... Deciding which to be used based on range fraction")
# imin, imax = parse_injection_range(num_injections, injection_fraction_range)
# inj_table = inj_table[imin:imax]
# logging.info("We will be doing injections from {} to {}".format(
#     inj_table[0].simulation_id, inj_table[-1].simulation_id))

logging.info("Preparing optimizers")
### Maybe these options can also be made into argparse with defaults
## NM_opts
opts_search = {
    'maxiter': 5000,
    'maxfev': 15000,
    'xatol': 1e-12,
    'fatol': 1e-12,
    'adaptive': True
}

## L-BFGS-B opts
opts_polish = {
    # 'maxfun': 1000,
    # 'maxiter': 500,
    'ftol': 1e-12,
    'gtol': 1e-12,
    'maxfun': 10,
    'maxiter': 5,
}

## Diff evol
maxiter = 40
popsize = 15
atol = 1e-7
tol = 1e-5

## PSO
# instatiate the optimizer
options_pso = {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 5, 'p': 2}
max_iter = 300
swarm_size = 25
optimizer = GlobalBestPSO(n_particles=swarm_size,
                          dimensions=len(ub),
                          options=options_pso,
                          bounds=(lb, ub),
                          ftol=1e-12,
                          ftol_iter=10)

## collect results:
## results are [best_fit, match, maxiter, maxfev]
# res_nm = []
res_der2b = []
# res_deb = []
# res_pso = []

logging.info("Starting injection loop")
# for num, inj in enumerate(inj_table):
for num, inj in enumerate([1]):
    # logging.info(f"Running injection num {inj.simulation_id}")
    # mchirp, eta = pnu.mass1_mass2_to_mchirp_eta(inj.mass1, inj.mass2)
    # inj_params = [mchirp, eta, inj.spin1z, inj.spin2z]
    #
    # inj_fmin = lalsim.EOBHighestInitialFreq(inj.mass1 + inj.mass2)
    # inj_fmin = fmin if inj_fmin > fmin + 1 else inj_fmin
    #
    # stilde = create_injection(inj, ifo, inj_fmin, inj_fmax, inj_approx, taper, scale)

    m1, m2 = 30, 10.
    s1x, s1y, s1z = 0., 0., 0.7
    s2x, s2y, s2z = 0., 0., 0.
    approx = 'ML'
    # approx = 'IMRPhenomXAS'

    params = dict(mass1=m1,
                mass2=m2,
                spin1x=s1x,
                spin1y=s1y,
                spin1z=s1z,
                spin2x=s2x,
                spin2y=s2y,
                spin2z=s2z,
                coa_phase=0.,
                inclination=0.,
                distance=1.,
                delta_t=delta_t,
                delta_f=delta_f,
                approximant=approx,
                f_lower=fmin,
                f_final=fmax
        )
    print(params)

    stilde, _ = pw.get_fd_waveform(**params, wf_approximant='IMRPhenomXAS', mass_lens=1e4, impact_parameter=0.15)
    # stilde, _ = pw.get_fd_waveform(**params)
    print(stilde.data.size, kmin, kmax)
    print(psd.delta_f, stilde.delta_f)

    # print(stilde)
    # print(stilde.data[kmin:kmax])
    # if stilde:
        # print(stilde.data[kmin:kmax])
        # print(4*delta_f * np.sum(stilde.data[kmin:kmax]*stilde.data[kmin:kmax].conj()/psd.data[kmin:kmax]))
    ht = stilde[kmin:kmax]
    # snorm = ht.weighted_inner(stilde[kmin:kmax], psd[kmin:kmax]) * 4 * psd.delta_f
    snorm = ht.weighted_inner(ht, psd[kmin:kmax]) * 4 * psd.delta_f
    # snorm = pf.sigmasq(stilde, psd, fmin+1, fmax-1)
    # print(f'snorm={snorm}')

    rec_params = np.random.uniform(lb, ub)

    # logging.info("NM for injection {}".format(inj.simulation_id))
    # res = minimize(funcmin,
    #                rec_params,
    #                args=(stilde, psd, fmin, fmax, rec_approx, snorm),
    #                method='Nelder-Mead',
    #                bounds=[[l, h] for l, h in zip(lb, ub)],
    #                tol=1e-12,
    #                options=opts_search)
    # print(res)
    #
    # res_nm.append([inj.simulation_id] + res.x.tolist() +
    #               [1 - res.fun, res.nit, res.nfev])
    #
    # logging.info("DiffEvol r2b1bin for injection {}".format(inj.simulation_id))
    logging.info("DiffEvol r2b1bin for injection")
    result = differential_evolution(funcmin,
                                    bounds=bounds,
                                    args=(stilde, psd, fmin_match, fmax, rec_approx,
                                        snorm),
                                    strategy='best1bin',
                                    maxiter=maxiter,
                                    atol=atol,
                                    disp=True,
                                    polish=polish,
                                    popsize=popsize,
                                    tol=tol,
                                    mutation=0.5,
                                    recombination=0.7,
                                    seed=num)
    print(result)

    # res_der2b.append([inj.simulation_id] + result.x.tolist() +
    #                 [1 - result.fun, result.nit, result.nfev])

    # print(res_der2b)
    # logging.info("PSO for injection {}".format(inj.simulation_id))
    # cost, pos = optimizer.optimize(ps.cost(funcmin),
    #                                max_iter,
    #                                stilde=stilde,
    #                                psd=psd,
    #                                f_min=fmin,
    #                                f_max=fmax,
    #                                approx=rec_approx,
    #                                snorm=snorm,
    #                                verbose=disp)
    #
    # coarse_params = pos
    # print(cost, pos)
    # res = minimize(funcmin,
    #                coarse_params,
    #                args=(stilde, psd, fmin, fmax, rec_approx, snorm),
    #                method='L-BFGS-B',
    #                bounds=[[l, h] for l, h in zip(lb, ub)],
    #                tol=1e-12,
    #                options=opts_polish)
    # print(res)
    #
    # res_pso.append([inj.simulation_id] + res.x.tolist() + [
    #     1 - res.fun, res.nit + len(optimizer.cost_history), res.nfev +
    #     len(optimizer.cost_history) * optimizer.swarm.n_particles
    # ])

