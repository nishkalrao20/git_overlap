import math
import scipy
import pycbc
import pickle
import pycbc.psd
import numpy as np
import pycbc.waveform
import matplotlib.pyplot as plt

N_iter = 1e4
delta_f = 1
low_frequency_cutoff = 20
high_frequency_cutoff = 1024
flen = int(high_frequency_cutoff / delta_f)

waveform_metadata_a=pickle.load(open('./Overlap_Injection/Output/Waveform A Meta Data.pkl', 'rb'))
for key,val in waveform_metadata_a['H1']['parameters'].items():
    exec(key +'_a' + '=val')
mchirp_a, eta_a = ((mass_1_a*mass_2_a)**(3/5))/((mass_1_a+mass_2_a)**(1/5)), (mass_1_a*mass_2_a)/((mass_1_a+mass_2_a)**2)

waveform_metadata_b=pickle.load(open('./Overlap_Injection/Output/Waveform B Meta Data.pkl', 'rb'))
for key,val in waveform_metadata_b['H1']['parameters'].items():
    exec(key +'_b' + '=val')
mchirp_b, eta_b = ((mass_1_b*mass_2_b)**(3/5))/((mass_1_b+mass_2_b)**(1/5)), (mass_1_b*mass_2_b)/((mass_1_b+mass_2_b)**2)

psd = pycbc.psd.read.from_txt('./Overlap_NM/O3-H1-C01_CLEAN_SUB60HZ-1262197260.0_sensitivity_strain_asd.txt', flen, delta_f, low_frequency_cutoff, is_asd_file=True)

def mismatch(hp, mchirp, eta, spin1z, spin2z, psd, distance, coa_phase):

    mtot = mchirp * np.power(eta, -3./5)
    fac = np.sqrt(1. - 4.*eta)
    m1, m2 = (mtot * (1. + fac) / 2., mtot * (1. - fac) / 2.)

    if math.isnan(m1) or math.isnan(m2) or math.isnan(spin1z) or math.isnan(spin2z) or m1 < 2 or m2 < 2 or m1/m2 < 1/18 or m1/m2 > 18 or m1+m2 > 800 or spin1z < -0.99 or spin1z > 0.99 or spin2z < -0.99 or spin2z > 0.99:
            log_mismatch = 1e6 
    
    else:
        hp_templ, hc_templ = pycbc.waveform.get_fd_waveform(approximant='IMRPhenomPv2', mass1=m1, mass2=m2, spin1z=spin1z, spin2z=spin2z, 
                        distance=distance, coa_phase=coa_phase, delta_f=delta_f, f_lower=low_frequency_cutoff)
        hp.resize(max(len(hp), len(hp_templ)))
        hp_templ.resize(max(len(hp), len(hp_templ)))
        match = pycbc.filter.matchedfilter.match(hp, hp_templ, low_frequency_cutoff=low_frequency_cutoff, psd=psd, high_frequency_cutoff=high_frequency_cutoff, v1_norm=None, v2_norm=None)[0]
        log_mismatch = np.log10(1-match)
    
    return log_mismatch 

def minimize_mismatch(fun_log_mismatch, mchirp_0, eta_0, spin1z_0, spin2z_0): 
   
   res = scipy.optimize.minimize(fun_log_mismatch, (mchirp_0, eta_0, spin1z_0, spin2z_0), method='Nelder-Mead', options={'adaptive':True}) 
   
   return res.fun 

def ff(mchirp, eta, spin1z, spin2z, psd, distance, coa_phase, N_iter):

    mtot = mchirp * np.power(eta, -3./5)
    fac = np.sqrt(1. - 4.*eta)
    m1, m2 = (mtot * (1. + fac) / 2., mtot * (1. - fac) / 2.)

    if math.isnan(m1) or math.isnan(m2) or math.isnan(spin1z) or math.isnan(spin2z) or m1 < 2 or m2 < 2 or m1/m2 < 1/18 or m1/m2 > 18 or m1+m2 > 800 or spin1z < -0.99 or spin1z > 0.99 or spin2z < -0.99 or spin2z > 0.99:
        log_mismatch = 1e6 

    else:    
        hp, hc = pycbc.waveform.get_fd_waveform(approximant='IMRPhenomPv2', mass1=m1, mass2=m2, spin1z=spin1z, spin2z=spin2z, 
                            distance=distance, coa_phase=coa_phase, delta_f=delta_f, f_lower=low_frequency_cutoff)
                        
        fun_log_mismatch = lambda x: mismatch(hp, x[0], x[1], x[2], x[3],  psd, distance, coa_phase)

        sigma_mc, sigma_eta, sigma_spin = 0.5, 0.1, 0.1

        mchirp_0 = scipy.stats.truncnorm.rvs(-3, 3, size=10*int(N_iter))*sigma_mc+mchirp
        eta_0 = scipy.stats.truncnorm.rvs(-3, 3, size=10*int(N_iter))*sigma_eta+eta
        spin1z_0 = scipy.stats.truncnorm.rvs(-3, 3, size=10*int(N_iter))*sigma_spin+spin1z
        spin2z_0 = scipy.stats.truncnorm.rvs(-3, 3, size=10*int(N_iter))*sigma_spin+spin2z

        idx = (mchirp_0>1) & (mchirp_0<200) & (eta_0>0.02) & (eta_0<0.25) & (spin1z_0>-0.99) & (spin1z_0<0.99) & (spin2z_0>-0.99) & (spin2z_0<0.99)

        mchirp_0 = np.random.choice(mchirp_0[idx], int(N_iter)-1)
        eta_0 = np.random.choice(eta_0[idx], int(N_iter)-1)
        spin1z_0 = np.random.choice(spin1z_0[idx], int(N_iter)-1)
        spin2z_0 = np.random.choice(spin2z_0[idx], int(N_iter)-1)
        
        log_mismatch_arr = np.vectorize(minimize_mismatch)(fun_log_mismatch, mchirp_0, eta_0, spin1z_0, spin2z_0)
        match_arr = 1 - np.power(10, log_mismatch_arr)
        log_mismatch = np.min(log_mismatch_arr)

    return log_mismatch, 1 - 10**log_mismatch, mchirp_0, eta_0, spin1z_0, spin2z_0, log_mismatch_arr, match_arr

log_mismatch_a, match_a, mchirp_af, eta_af, spin1z_af, spin2z_af, log_mismatch_arr_a, match_arr_a = np.vectorize(ff, otypes=[object], excluded={4,5,6,7})(mchirp_a, eta_a, a_1_a, a_2_a, psd, luminosity_distance_a, phase_a, N_iter)
log_mismatch_b, match_b, mchirp_bf, eta_bf, spin1z_bf, spin2z_bf, log_mismatch_arr_b, match_arr_b = np.vectorize(ff, otypes=[object], excluded={4,5,6,7})(mchirp_b, eta_b, a_1_b, a_2_b, psd, luminosity_distance_b, phase_b, N_iter)

mchirp_aff, eta_aff, spin1z_aff, spin2z_aff = mchirp_af[np.argmin(log_mismatch_arr_a)], eta_af[np.argmin(log_mismatch_arr_a)], spin1z_af[np.argmin(log_mismatch_arr_a)], spin2z_af[np.argmin(log_mismatch_arr_a)]
mchirp_bff, eta_bff, spin1z_bff, spin2z_bff = mchirp_bf[np.argmin(log_mismatch_arr_b)], eta_bf[np.argmin(log_mismatch_arr_b)], spin1z_bf[np.argmin(log_mismatch_arr_b)], spin2z_bf[np.argmin(log_mismatch_arr_b)]

mtotal_af, fac_af, mtotal_bf, fac_bf = mchirp_aff * eta_aff**(-3/5), np.sqrt(1 - 4*eta_aff), mchirp_bff * eta_bff**(-3/5), np.sqrt(1 - 4*eta_bff)
mass_1_af, mass_2_af, mass_1_bf, mass_2_bf = mtotal_af * (1 + fac_af) / 2, mtotal_af * (1 - fac_af) / 2, mtotal_bf * (1 + fac_bf) / 2, mtotal_bf * (1 - fac_bf) / 2
    
fig, ax = plt.subplots(2,2, figsize=(14, 12))

mc12_c = ax[0,0].scatter(mchirp_af, eta_af, c=match_arr_a, cmap='plasma')
ax[0,0].scatter(mchirp_a, eta_a, color = 'white', edgecolors='black')
ax[0,0].scatter(mchirp_aff, eta_aff, facecolors = 'None', edgecolors='black')
ax[0,0].text(1.001*mchirp_a, 1.001*eta_a,'A')
ax[0,0].text(1.001*mchirp_aff, 1.001*eta_aff,'F')
ax[0,0].set_xlabel('$\mathcal{M}}$')
ax[0,0].set_ylabel('${\eta}$')
ax[0,0].set_title('Mass Ratio ${\eta}$ $-$ Chirp Mass ${\mathcal{M}}$')
clb1 = plt.colorbar(mc12_c, ax=ax[0,0])
clb1.ax.set_title('$Match$')

s12_c = ax[0,1].scatter(spin1z_af, spin2z_af, c=match_arr_a, cmap='plasma')
ax[0,1].scatter(a_1_a, a_2_a, color = 'white', edgecolors='black')
ax[0,1].scatter(spin1z_aff, spin2z_aff, facecolors = 'None', edgecolors='black')
ax[0,1].text(1.001*a_1_a, 1.001*a_2_a,'A')
ax[0,1].text(1.001*spin1z_aff, 1.001*spin2z_aff,'F')
ax[0,1].set_xlabel('${s_z}_1$')
ax[0,1].set_ylabel('${s_z}_2$')
ax[0,1].set_title('Spins: ${s_z}_1-{s_z}_2$')
clb2 = plt.colorbar(s12_c, ax=ax[0,1])
clb2.ax.set_title('$Match$')

mc12_c = ax[1,0].scatter(mchirp_bf, eta_bf, c=match_arr_b, cmap='plasma')
ax[1,0].scatter(mchirp_b, eta_b, color = 'white', edgecolors='black')
ax[1,0].scatter(mchirp_bff, eta_bff, facecolors = 'None', edgecolors='black')
ax[1,0].text(1.001*mchirp_b, 1.001*eta_b,'B')
ax[1,0].text(1.001*mchirp_bff, 1.001*eta_bff,'F')
ax[1,0].set_xlabel('$\mathcal{M}}$')
ax[1,0].set_ylabel('${\eta}$')
ax[1,0].set_title('Mass Ratio ${\eta}$ $-$ Chirp Mass ${\mathcal{M}}$')
clb3 = plt.colorbar(mc12_c, ax=ax[1,0])
clb3.ax.set_title('$Match$')

s12_c = ax[1,1].scatter(spin1z_bf, spin2z_bf, c=match_arr_b, cmap='plasma')
ax[1,1].scatter(a_1_b, a_2_b, color = 'white', edgecolors='black')
ax[1,1].scatter(spin1z_bff, spin2z_bff, facecolors = 'None', edgecolors='black')
ax[1,1].text(1.001*a_1_b, 1.001*a_2_b,'B')
ax[1,1].text(1.001*spin1z_bff, 1.001*spin2z_bff,'F')
ax[1,1].set_xlabel('${s_z}_1$')
ax[1,1].set_ylabel('${s_z}_2$')
ax[1,1].set_title('Spins: ${s_z}_1-{s_z}_2$')
clb4 = plt.colorbar(s12_c, ax=ax[1,1])
clb4.ax.set_title('$Match$')

plt.suptitle('Singles: A & B; $\\log$ Mismatch$_A$: %s, Match$_A$: %s, $\\log$ Mismatch$_B$: %s, Match$_B$: %s'%(log_mismatch_a, match_a, log_mismatch_b, match_b))
fig.tight_layout()
plt.savefig('./Overlap_NM/Output/Singles_NM.png')
plt.close()

print('Injected - A: Chirp Mass: %s, Sym. Ratio: %s, Spin1z: %s, Spin2z: %s, Mass1: %s, Mass2: %s'%(np.round(mchirp_a, 2), np.round(eta_a, 2), np.round(a_1_a, 2), np.round(a_2_a, 2), np.round(mass_1_a, 2), np.round(mass_2_a, 2)))
print('Injected - B: Chirp Mass: %s, Sym. Ratio: %s, Spin1z %s, Spin2z: %s, Mass1 %s, Mass2: %s'%(np.round(mchirp_b, 2), np.round(eta_b, 2), np.round(a_1_b, 2), np.round(a_2_b, 2), np.round(mass_1_b, 2), np.round(mass_1_b, 2)))
print('Recovered - A: Chirp Mass: %s, Sym. Ratio: %s, Spin1z: %s, Spin2z: %s, Mass1: %s, Mass2: %s'%(np.round(mchirp_aff, 2), np.round(eta_aff, 2), np.round(spin1z_aff, 2), np.round(spin2z_aff, 2), np.round(mass_1_af, 2), np.round(mass_2_af, 2)))
print('Recovered - B: Chirp Mass: %s, Sym. Ratio: %s, Spin1z: %s, Spin2z: %s, Mass1: %s, Mass2: %s'%(np.round(mchirp_bff, 2), np.round(eta_bff, 2), np.round(spin1z_bff, 2), np.round(spin2z_bff, 2), np.round(mass_1_bf, 2), np.round(mass_2_bf, 2)))