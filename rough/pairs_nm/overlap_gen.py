import bilby
import pickle
import numpy as np

duration = 1000
minimum_frequency = 20
sampling_frequency = 4096

waveform_arguments = {
    'waveform_approximant': 'IMRPhenomPv2',
    'reference_frequency': 50,
    'minimum_frequency': 2
}

waveform_metadata_a=pickle.load(open('Overlap_Injection/Output/Waveform A Meta Data.pkl', 'rb'))

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

with open('Overlap_NM/Output/Waveform Metadata/Waveform A Meta Data.pkl', 'wb') as file:
    pickle.dump(ifos_a.meta_data, file)

N_b = int(1e2)

for i in range(0,N_b):

    delta_tb=np.random.uniform(0,1)
    waveform_metadata_b=pickle.load(open('Overlap_Injection/Output/Waveform B Meta Data.pkl', 'rb'))
    waveform_metadata_b['H1']['parameters']['geocent_time']=waveform_metadata_a['H1']['parameters']['geocent_time']+delta_tb

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
    ifos_b.inject_signal(waveform_generator=waveform_generator_b, parameters=waveform_metadata_b['H1']['parameters'])
    
    ifos_b.meta_data['H1']['parameters']['delta_tb']=delta_tb

    with open('Overlap_NM/Output/Waveform Metadata/Waveform B Meta Data %s.pkl'%(i+1), 'wb') as file:
        pickle.dump(ifos_b.meta_data, file)