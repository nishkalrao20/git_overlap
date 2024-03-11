import math
import scipy
import pycbc
import pickle
import pycbc.psd
import numpy as np
import pycbc.waveform
import matplotlib.pyplot as plt

N_iter = 1e4
delta_f = 1.0 / 4
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

def overlap_ff(mass_1_a, mass_2_a, spin1z_a, spin2z_a, distance_a, coa_phase_a, psd, mass_1_b, mass_2_b, spin1z_b, spin2z_b, distance_b, coa_phase_b, N_iter):

    mchirp_a, eta_a = ((mass_1_a*mass_2_a)**(3/5))/((mass_1_a+mass_2_a)**(1/5)), (mass_1_a*mass_2_a)/((mass_1_a+mass_2_a)**2)
    mchirp_b, eta_b = ((mass_1_b*mass_2_b)**(3/5))/((mass_1_b+mass_2_b)**(1/5)), (mass_1_b*mass_2_b)/((mass_1_b+mass_2_b)**2)

    if math.isnan(mass_1_a) or math.isnan(mass_1_b) or math.isnan(mass_2_a) or math.isnan(mass_2_a) or math.isnan(spin1z_a) or math.isnan(spin2z_a) or math.isnan(spin1z_b) or math.isnan(spin2z_b) or mass_1_a < 2 or  mass_1_b < 2 or mass_2_a < 2 or mass_2_b < 2 or mass_1_a/mass_2_a < 1/18 or mass_1_b/mass_2_b < 1/18 or mass_1_a/mass_2_a > 18 or mass_1_b/mass_2_b > 18 or mass_1_a+mass_2_a > 800 or mass_1_b+mass_2_b > 800 or spin1z_a < -0.99 or spin1z_a > 0.99 or spin1z_b < -0.99 or spin1z_b > 0.99 or spin2z_a < -0.99 or spin2z_a > 0.99 or spin2z_b < -0.99 or spin2z_b > 0.99:
        log_mismatch = 1e6 
            
    else:            
        hp_a, hc_a = pycbc.waveform.get_fd_waveform(approximant='IMRPhenomPv2', mass1=mass_1_a, mass2=mass_2_a, spin1z=spin1z_a, spin2z=spin2z_a, distance=luminosity_distance_a, coa_phase=phase_a, delta_f=delta_f, f_lower=low_frequency_cutoff)
        hp_b, hc_b = pycbc.waveform.get_fd_waveform(approximant='IMRPhenomPv2', mass1=mass_1_b, mass2=mass_2_b, spin1z=spin1z_b, spin2z=spin2z_b, distance=luminosity_distance_b, coa_phase=phase_b, delta_f=delta_f, f_lower=low_frequency_cutoff)
        
        hp_a.resize(max(len(hp_a), len(hp_b))), hp_b.resize(max(len(hp_a), len(hp_b)))
        hp = np.add(hp_a,hp_b)
        
        if waveform_metadata_a['H1']['optimal_SNR']>waveform_metadata_b['H1']['optimal_SNR']:
            fun_log_mismatch = lambda x: mismatch(hp, x[0], x[1], x[2], x[3], psd, distance_a, coa_phase_a)
        if waveform_metadata_b['H1']['optimal_SNR']>waveform_metadata_a['H1']['optimal_SNR']:
            fun_log_mismatch = lambda x: mismatch(hp, x[0], x[1], x[2], x[3], psd, distance_b, coa_phase_b)

        mchirp_0 = scipy.stats.uniform.rvs(size=10*int(N_iter))*(mchirp_a+mchirp_b)
        eta_0 = scipy.stats.uniform.rvs(size=10*int(N_iter))*(eta_a+eta_b)
        spin1z_0 = scipy.stats.uniform.rvs(size=10*int(N_iter))*(spin1z_a+spin1z_b)
        spin2z_0 = scipy.stats.uniform.rvs(size=10*int(N_iter))*(spin2z_a+spin2z_b)

        idx = (mchirp_0>1) & (mchirp_0<200) & (eta_0>0.02) & (eta_0<=0.25) & (spin1z_0>-0.99) & (spin1z_0<0.99) & (spin2z_0>-0.99) & (spin2z_0<0.99)

        mchirp_0 = np.append((mchirp_a+mchirp_b)/2, np.random.choice(mchirp_0[idx], int(N_iter)-1))
        eta_0 = np.append((eta_a+eta_b)/2, np.random.choice(eta_0[idx], int(N_iter)-1))
        spin1z_0 = np.append((spin1z_a+spin1z_b)/2, np.random.choice(spin1z_0[idx], int(N_iter)-1))
        spin2z_0 = np.append((spin2z_a+spin2z_b)/2, np.random.choice(spin2z_0[idx], int(N_iter)-1))

        log_mismatch_arr = np.vectorize(minimize_mismatch)(fun_log_mismatch, mchirp_0, eta_0, spin1z_0, spin2z_0)
        match_arr = 1 - np.power(10, log_mismatch_arr)
        log_mismatch = np.min(log_mismatch_arr)

    return log_mismatch, 1 - 10**log_mismatch, mchirp_0, eta_0, spin1z_0, spin2z_0, log_mismatch_arr, match_arr

log_mismatch, match, mchirp, eta, spin1z, spin2z, log_mismatch_arr, match_arr = np.vectorize(overlap_ff, otypes=[object], excluded={4,5,6,11,12,13})(mass_1_a, mass_2_a, a_1_a, a_2_a, luminosity_distance_a, phase_a, psd,
                                                                                                                                                    mass_1_b, mass_2_b, a_1_b, a_2_b, luminosity_distance_b, phase_b, N_iter)


mchirp_f, eta_f, spin1z_f, spin2z_f = mchirp[np.argmin(log_mismatch_arr)], eta[np.argmin(log_mismatch_arr)], spin1z[np.argmin(log_mismatch_arr)], spin2z[np.argmin(log_mismatch_arr)]
mtotal, fac = mchirp_f * eta_f**(-3/5), np.sqrt(1 - 4 * eta_f)
mass_1, mass_2 = mtotal * (1 + fac) / 2, mtotal * (1 - fac) / 2

fig, ax = plt.subplots(1,2, figsize=(14, 6))

mc = ax[0].scatter(mchirp, eta, c=match_arr, cmap='plasma')
ax[0].scatter([mchirp_a, mchirp_b], [eta_a, eta_b], color = 'white', edgecolors='black')
ax[0].scatter(mchirp_f, eta_f, facecolors = 'None', edgecolors='black')
ax[0].text(1.001*mchirp_a, 1.001*eta_a,'A')
ax[0].text(1.001*mchirp_b, 1.001*eta_b,'B')
ax[0].text(1.001*mchirp_f, 1.001*eta_f,'F')
ax[0].set_xlabel('$\mathcal{M}}$')
ax[0].set_ylabel('${\eta}$')
ax[0].set_title('Mass Ratio ${\eta}$ $-$ Chirp Mass ${\mathcal{M}}$')
clb1 = plt.colorbar(mc, ax=ax[0])
clb1.ax.set_title('$Match$')

s = ax[1].scatter(spin1z, spin2z, c=match_arr, cmap='plasma')
ax[1].scatter([a_1_a, a_1_b], [a_2_a, a_2_b], color = 'white', edgecolors='black')
ax[1].scatter(spin1z_f, spin2z_f, facecolors = 'None', edgecolors='black')
ax[1].text(1.001*a_1_a, 1.001*a_2_a,'A')
ax[1].text(1.001*a_1_b, 1.001*a_2_b,'B')
ax[1].text(1.001*spin1z_f, 1.001*spin2z_f,'F')
ax[1].set_xlabel('${s_z}_1$')
ax[1].set_ylabel('${s_z}_2$')
ax[1].set_title('Spins: ${s_z}_1-{s_z}_2$')
clb2 = plt.colorbar(s, ax=ax[1])
clb2.ax.set_title('$Match$')

plt.suptitle('Overlap: A & B; $\\log$ Mismatch: %s, Match: %s'%(log_mismatch, match))
fig.tight_layout()
plt.savefig('./Overlap_NM/Output/Overlap_NM.png')
plt.close()

print('Injected - A: Chirp Mass: %s, Sym. Ratio: %s, Spin1z: %s, Spin2z: %s, Mass1: %s, Mass2: %s'%(np.round(mchirp_a, 2), np.round(eta_a, 2), np.round(a_1_a, 2), np.round(a_2_a, 2), np.round(mass_1_a, 2), np.round(mass_2_a, 2)))
print('Injected - B: Chirp Mass: %s, Sym. Ratio: %s, Spin1z %s, Spin2z: %s, Mass1 %s, Mass2: %s'%(np.round(mchirp_b, 2), np.round(eta_b, 2), np.round(a_1_b, 2), np.round(a_2_b, 2), np.round(mass_1_b, 2), np.round(mass_1_b, 2)))
print('Recovered: Chirp Mass: %s, Sym. Ratio: %s, Spin1z: %s, Spin2z: %s, Mass1: %s, Mass2: %s'%(np.round(mchirp_f, 2), np.round(eta_f, 2), np.round(spin1z_f, 2), np.round(spin2z_f, 2), np.round(mass_1, 2), np.round(mass_2, 2)))