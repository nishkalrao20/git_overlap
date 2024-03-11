import math
import scipy
import pycbc
import pickle
import pycbc.psd
import numpy as np
import pycbc.waveform
import matplotlib.pyplot as plt

N_b = int(10)
N_iter = int(1e2)
delta_f = 1 / 4
low_frequency_cutoff = 20
high_frequency_cutoff = 1024
flen = int(high_frequency_cutoff / delta_f)

waveform_metadata_a=pickle.load(open('Overlap_NM/Output/Waveform Metadata/Waveform A Meta Data.pkl', 'rb'))
for key,val in waveform_metadata_a['H1']['parameters'].items():
    exec(key +'_a' + '=val')
mchirp_a, eta_a = np.power((mass_1_a*mass_2_a),3/5)/np.power((mass_1_a+mass_2_a),1/5), (mass_1_a*mass_2_a)/(np.power((mass_1_a*mass_2_a),2))

mass_1_b, mass_2_b, a_1_b, a_2_b, luminosity_distance_b, phase_b, delta_tb = [], [], [], [], [], [], []

for i in range(0,N_b):
    waveform_metadata_b=pickle.load(open('Overlap_NM/Output/Waveform Metadata/Waveform B Meta Data %s.pkl'%(i+1), 'rb'))
    mass_1_b, mass_2_b, a_1_b, a_2_b, luminosity_distance_b, phase_b, delta_tb = np.append(mass_1_b, waveform_metadata_b['H1']['parameters']['mass_1']), np.append(mass_2_b, waveform_metadata_b['H1']['parameters']['mass_2']), np.append(a_1_b, waveform_metadata_b['H1']['parameters']['a_1']), np.append(a_2_b, waveform_metadata_b['H1']['parameters']['a_2']), np.append(luminosity_distance_b, waveform_metadata_b['H1']['parameters']['luminosity_distance']), np.append(phase_b, waveform_metadata_b['H1']['parameters']['phase']), np.append(delta_tb, waveform_metadata_b['H1']['parameters']['delta_tb'])
mchirp_b, eta_b = np.power((mass_1_b*mass_2_b),3/5)/np.power((mass_1_b+mass_2_b),1/5), (mass_1_b*mass_2_b)/(np.power((mass_1_b*mass_2_b),2))

psd = pycbc.psd.read.from_txt('Overlap_NM/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt', flen, delta_f, low_frequency_cutoff, is_asd_file=True)

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

def overlap_ff(m_1_a, m_2_a, spin1z_a, spin2z_a, distance_a, coa_phase_a, psd, m_1_b, m_2_b, spin1z_b, spin2z_b, distance_b, coa_phase_b, N_iter, delta_b):

    mchirp_a, eta_a = ((m_1_a*m_2_a)**(3/5))/((m_1_a+m_2_a)**(1/5)), (m_1_a*m_2_a)/((m_1_a+m_2_a)**2)
    mchirp_b, eta_b = ((m_1_b*m_2_b)**(3/5))/((m_1_b+m_2_b)**(1/5)), (m_1_b*m_2_b)/((m_1_b+m_2_b)**2)

    if math.isnan(m_1_a) or math.isnan(m_1_b) or math.isnan(m_2_a) or math.isnan(m_2_a) or math.isnan(spin1z_a) or math.isnan(spin2z_a) or math.isnan(spin1z_b) or math.isnan(spin2z_b) or m_1_a < 2 or  m_1_b < 2 or m_2_a < 2 or m_2_b < 2 or m_1_a/m_2_a < 1/18 or m_1_b/m_2_b < 1/18 or m_1_a/m_2_a > 18 or m_1_b/m_2_b > 18 or m_1_a+m_2_a > 800 or m_1_b+m_2_b > 800 or spin1z_a < -0.99 or spin1z_a > 0.99 or spin1z_b < -0.99 or spin1z_b > 0.99 or spin2z_a < -0.99 or spin2z_a > 0.99 or spin2z_b < -0.99 or spin2z_b > 0.99:
        log_mismatch = 1e6 
            
    else:            
        hp_a, hc_a = pycbc.waveform.get_fd_waveform(approximant='IMRPhenomPv2', mass1=m_1_a, mass2=m_2_a, spin1z=spin1z_a, spin2z=spin2z_a, distance=distance_a, coa_phase=coa_phase_a, delta_f=delta_f, f_lower=low_frequency_cutoff)
        hp_b, hc_b = pycbc.waveform.get_fd_waveform(approximant='IMRPhenomPv2', mass1=m_1_b, mass2=m_2_b, spin1z=spin1z_b, spin2z=spin2z_b, distance=distance_b, coa_phase=coa_phase_b, delta_f=delta_f, f_lower=low_frequency_cutoff)
        
        hpt_a, hct_a = pycbc.types.TimeSeries(hp_a, delta_t=1/flen), pycbc.types.TimeSeries(hc_a, delta_t=1/flen)
        hpt_b, hct_b = pycbc.types.TimeSeries(hp_b, delta_t=1/flen, epoch=delta_b), pycbc.types.TimeSeries(hc_b, delta_t=1/flen, epoch=delta_b)

        hpt_b, hct_b = np.lib.pad(hpt_b,(int(delta_b*flen),0)), np.lib.pad(hct_b,(int(delta_b*flen),0))
        hpt_a, hct_a = np.lib.pad(hpt_a,(0,len(hpt_b)-len(hpt_a))), np.lib.pad(hct_a,(0,len(hpt_b)-len(hpt_a)))
        hpt, hct = np.add(hpt_a,hpt_b), np.add(hct_a,hct_b)

        hp, hc = pycbc.types.FrequencySeries(hpt, delta_f=delta_f), pycbc.types.FrequencySeries(hct, delta_f=delta_f)
        hpt, hct = pycbc.types.TimeSeries(hp, delta_t=1/flen), pycbc.types.TimeSeries(hc, delta_t=1/flen)
        
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

mchirp_f, eta_f, spin1z_f, spin2z_f, mchirp_suparr, eta_suparr, spin1z_suparr, spin2z_suparr, match_suparr = [], [], [], [], [], [], [], [], []

for j in range(len(delta_tb)):
    
    log_mismatch, match, mchirp, eta, spin1z, spin2z, log_mismatch_arr, match_arr = np.vectorize(overlap_ff, otypes=[object], excluded={4,5,6,11,12,13})(mass_1_a, mass_2_a, a_1_a, a_2_a, luminosity_distance_a, phase_a, psd, mass_1_b[j], mass_2_b[j], a_1_b[j], a_2_b[j], luminosity_distance_b[j], phase_b[j], N_iter, delta_tb[j])
    mchirp_f, eta_f, spin1z_f, spin2z_f = np.append(mchirp, mchirp[np.argmin(log_mismatch_arr)]), np.append(eta, eta[np.argmin(log_mismatch_arr)]), np.append(spin1z, spin1z[np.argmin(log_mismatch_arr)]), np.append(spin2z, spin2z[np.argmin(log_mismatch_arr)])
    mchirp_suparr, eta_suparr, spin1z_suparr, spin2z_suparr, match_suparr = np.append(mchirp_suparr, mchirp_f[j]), np.append(eta_suparr, eta_f[j]), np.append(spin1z_suparr, spin1z_f[j]), np.append(spin2z_suparr, spin2z_f[j]), np.append(match_suparr, match)
    
    mtotal, fac = mchirp_f * np.power(eta_f,(-3/5)), np.sqrt(1 - 4 * eta_f)
    mass_1, mass_2 = mtotal * (1 + fac) / 2, mtotal * (1 - fac) / 2

    fig, ax = plt.subplots(1,2, figsize=(14, 6))

    mc = ax[0].scatter(mchirp, eta, c=match_arr, cmap='plasma')
    ax[0].scatter([mchirp_a, mchirp_b[j]], [eta_a, eta_b[j]], color = 'white', edgecolors='black')
    ax[0].scatter(mchirp_f[j], eta_f[j], facecolors = 'None', edgecolors='black')
    ax[0].text(1.001*mchirp_a, 1.001*eta_a,'A')
    ax[0].text(1.001*mchirp_b[j], 1.001*eta_b[j],'B')
    ax[0].text(1.001*mchirp_f[j], 1.001*eta_f[j],'F')
    ax[0].set_xlabel('$\mathcal{M}}$')
    ax[0].set_ylabel('${\eta}$')
    ax[0].set_title('Mass Ratio ${\eta}$ $-$ Chirp Mass ${\mathcal{M}}$')
    clb1 = plt.colorbar(mc, ax=ax[0])
    clb1.ax.set_title('$Match$')

    s = ax[1].scatter(spin1z, spin2z, c=match_arr, cmap='plasma')
    ax[1].scatter([a_1_a, a_1_b[j]], [a_2_a, a_2_b[j]], color = 'white', edgecolors='black')
    ax[1].scatter(spin1z_f[j], spin2z_f[j], facecolors = 'None', edgecolors='black')
    ax[1].text(1.001*a_1_a, 1.001*a_2_a,'A')
    ax[1].text(1.001*a_1_b[j], 1.001*a_2_b[j],'B')
    ax[1].text(1.001*spin1z_f[j], 1.001*spin2z_f[j],'F')
    ax[1].set_xlabel('${s_z}_1$')
    ax[1].set_ylabel('${s_z}_2$')
    ax[1].set_title('Spins: ${s_z}_1-{s_z}_2$')
    clb2 = plt.colorbar(s, ax=ax[1])
    clb2.ax.set_title('$Match$')

    plt.suptitle('Overlap: A & B; $\\log$ Mismatch: %s, Match: %s at $\\Delta t=$ %s'%(log_mismatch, match, delta_tb[j]))
    fig.tight_layout()
    plt.savefig('./Overlap_NM/Output/Cummulative/Overlap_NM %s.png'%(j+1))
    plt.close()

fig, ax = plt.subplots(1,2, figsize=(14, 6))

mc = ax[0].scatter(delta_tb, mchirp_suparr, c=match_suparr, cmap='plasma')
ax[0].plot(mchirp_a*np.ones(len(delta_tb)), 'k')
ax[0].plot(mchirp_b*np.ones(len(delta_tb)), 'k')
ax[0].set_ylabel('$\mathcal{M}}$')
ax[0].set_xlabel('${\\Delta t}$')
ax[0].set_title('Recovered Chirp Mass ${\\mathcal{M}}$ $-$ Time Delay $\\Delta$ $t$')
clb1 = plt.colorbar(mc, ax=ax[0])
clb1.ax.set_title('$Match$')

s = ax[1].scatter(delta_tb, eta_suparr, c=match_suparr, cmap='plasma')
ax[1].plot(eta_a*np.ones(len(delta_tb)), 'k')
ax[1].plot(eta_b*np.ones(len(delta_tb)), 'k')
ax[1].set_ylabel('${\eta}$')
ax[1].set_xlabel('${\\Delta t}$')
ax[1].set_title('Recovered Symm Mass Ratio ${\\eta}$ $-$ Time Delay $\\Delta$ $t$')
clb2 = plt.colorbar(s, ax=ax[1])
clb2.ax.set_title('$Match$')

fig.tight_layout()
plt.savefig('./Overlap_NM/Output/Overlap_NM (Match).png')
plt.close()