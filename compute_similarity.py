import os
import numpy as np
from data.wav_folder import read_wav
from resemblyzer import preprocess_wav, VoiceEncoder 

vctr_dir = '/path/to/vctr_utt_path/'
target_dir = '/path/to/tgt_utt_path/'

sampling_rate = 24000
encoder = VoiceEncoder()

vctr_list = []
for vctr_path in os.listdir(vctr_dir):
    vctr_path = os.path.join(vctr_dir, vctr_path)

    vctr_wav, _ = read_wav(vctr_path, sr=sampling_rate)

    vctr_list.append(vctr_wav)

target_list = []
for target_path in os.listdir(target_dir):
    target_path = os.path.join(target_dir, target_path)
    target_wav, _ = read_wav(target_path, sr=sampling_rate)
    target_list.append(target_wav)

spk_embeds_vctr = np.array([encoder.embed_speaker(vctr_list)])
spk_embeds_target = np.array([encoder.embed_speaker(target_list)])

spk_sim_vctr = np.inner(spk_embeds_vctr, spk_embeds_target)

print('CVC:{}'.format(spk_sim_vctr))