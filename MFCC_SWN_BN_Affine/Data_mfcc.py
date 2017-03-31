# End-End Speech Recognition Application using Recurrent Neural Network
# Name: Rajkumar Conjeevaram Mohan
# Email: Rajkumar.Conjeevaram-Mohan14@imperial.ac.uk
#
# This program follows a simple recipe from Alex Graves et. al 2015 paper
# ( Towards End-End Speech Recognition using Recurrent Neural Networks )
# that incorporates Connectionist Temporal Classification Network with
# RNN to obtain ability to compute loss without prior pre-segmentation.

import sys
from os import listdir
from os.path import isfile, join, splitext
import numpy as np
from scipy.io import wavfile
import time
import threading
from python_speech_features import mfcc
from python_speech_features import logfbank
import gc

class Data:
    """Audio waveform pre-processing entity"""
    b_id = 0
    file_path = None
    frame_rate = None
    training_files = []
    test_files = []
    n_files = 0
    frame_overlap_flag = False
    ON = True
    charmap = None
    reverse_map = None
    noise_types = np.array(["_NOISE_","_INHALE_","_EHM_HMM_","_LAUGH_"])
    batch = None

    # For multi-threading purpose
    buffer_filled = False
    frames = None
    n_frames = None
    targ_indices = None
    targ_values = None
    targ_shape = None
    transcript = None


    def __init__(self,batch_size,
                 file_path,
                 filelist_filename,
                 frame_overlap_flag=False,
                 ms_to_sample=20,
                 overlap_ms=10,
                 mfcc_coeff=26):
        self.mfcc_coeff = mfcc_coeff
        self.batch = batch_size
        self.file_path = file_path
        self.frame_overlap_flag = frame_overlap_flag
        self.ms_to_sample = ms_to_sample
        self.overlap_ms = overlap_ms
        if not isfile(filelist_filename):
            self.files = [f for f in listdir(file_path) if isfile(join(file_path,f)) if f.endswith("wav")]
            np.save(filelist_filename,self.files)
        else:
            self.files = np.load(filelist_filename)
        #self.safe_training_files = self.files
        self.files = self.files
        self.n_files = len(self.files)

        # Assuming the whole batch would have the same properties
        # we sample the properties from an arbitrary file
        self.frame_rate, _ = wavfile.read(join(file_path,self.files[0]))
        self.frame_length = (self.frame_rate / 1000) * ms_to_sample # 15ms as per sample rate
        self.sides = 'onesided'
        if self.sides == 'twosided':
            self.retain_fft = self.frame_length
        else:
            self.retain_fft = (self.frame_length / 2) + 1
        self.overlap = self.frame_rate/1000 * self.overlap_ms
        self.default_window = np.hamming(self.frame_length + self.overlap)

        #Build char map for dense representation of transcripts
        chars = []
        # Add blank label finally
        chars.append(chr(32))

        for i in range(65,91):
            chars.append(chr(i))

        # Special characters like punctuations
        chars.append(chr(39))
        #for n in self.noise_types:
        #    chars.append(n)

        indices = range(len(chars))
        self.charmap = dict(zip(chars,indices))
        self.reverse_map = dict(zip(indices,chars))
        self.start_thread()

    def next_batch(self):
        while not self.buffer_filled:
            time.sleep(1)

        self.buffer_filled = False
        self.start_thread()

        return self.frames, self.transcript, self.targ_indices, \
               self.targ_value, self.targ_shape, self.n_frames

    def start_thread(self):
        thread = threading.Thread(target=self.process_batch,
                                  args=(self.file_path,  self.charmap,
                                        self.noise_types, self.frame_overlap_flag))
        thread.daemon = True
        thread.start()


    def process_batch(self,file_path,charmap,noise_types,frame_overlap_flag):
        time.sleep(0.2)
        if (self.b_id + self.batch) >= self.n_files:
            self.b_id = 0
        data = []
        transcript = []
        targ_len = []
        targ_indices = []
        for b in range(self.batch):

            # Training data
            _,temp_data = wavfile.read(join(file_path,self.files[self.b_id + b]))
            data.append(temp_data)

            # Target data
            target_file = self.files[self.b_id + b] + ".trn"
            f = open(join(file_path, target_file), 'r')
            temp = f.read()
            f.close()

            temp = self.rep_dense(temp, charmap, noise_types)
            targ_len.append(len(temp))
            # Indices made for SparseTensor
            temp_indices = np.zeros([len(temp),2]) # 2 -> 2d indices
            temp_indices[:,0] = b
            temp_indices[:,1] = range(len(temp))
            targ_indices.append(temp_indices)
            #------------------------------
            transcript.append(temp)

        self.targ_indices = np.concatenate(targ_indices,axis=0)
        # Value for SparseTensor
        self.targ_value = np.concatenate(transcript,axis=0)
        # Shape for SparseTensor
        self.targ_shape = [self.batch,np.max(targ_len)]

        # Passing in as matrix, so that split_frames can realise
        # whether batch of data is included
        self.frames, self.n_frames = self.get_frames(data,frame_overlap_flag)
        self.b_id += self.batch
        self.transcript = transcript
        self.buffer_filled = True
        gc.collect()

    def get_max_framecount(self,raw_signal,frame_length):
        n_frames = []
        for b in range(len(raw_signal)):
            signal = raw_signal[b]
            n_frames.append(np.ceil(float(len(signal))/frame_length))
        return n_frames


    def get_delta(self,frames):
        delta = np.zeros(frames.shape)
        frames = np.pad(frames,((1,1),(0,0)),'edge')
        for t in range(1,frames.shape[0]-2):
            delta[t,:] = frames[t+1,:] - frames[t-1,:]
        return delta

    def get_frames(self,raw_signal,frame_overlap_flag,limit=4522):
        # Employ splitting overlapping frames
        #-------------------------------------
        frames = []
        frame_size = []

        # Accumulate spectrograms for each data in the batch
        for b in range(self.batch):
            time_signal = raw_signal[b]

            # Compute the Mel-Frequency Cepstral Coefficients
            mfcc_data = None
            if frame_overlap_flag:
                mfcc_data = mfcc(time_signal, samplerate=self.frame_rate, winlen=float(self.ms_to_sample)/1000,
                                 winstep=float(self.overlap_ms)/1000, numcep = self.mfcc_coeff/2, nfilt=self.mfcc_coeff,
                                 nfft=self.frame_length, appendEnergy=True, winfunc=np.hamming)
            else:
                mfcc_data = mfcc(time_signal, samplerate=self.frame_rate, winlen=float(self.ms_to_sample)/1000,
                                 winstep=0, numcep = self.mfcc_coeff/2, nfilt=self.mfcc_coeff,
                                 nfft=self.frame_length, appendEnergy=True, winfunc=np.hamming)

            delta = self.get_delta(mfcc_data)
            ddelta = self.get_delta(mfcc_data)
            mfcc_data = np.concatenate([mfcc_data,delta,ddelta],axis=1)
            # When the computed spectrogram's time steps is different to others, then
            # its timestep dimension is padded to be concatenated with others
            # max_size refers to the maximum number of timesteps in the current mini-batch
            l_f = mfcc_data.shape[0]
            frame_size.append(l_f)
            frames.append(mfcc_data)


        # Reshape each data to maximum time step, and store it in matrix
        max_time = np.max(frame_size)
        for b in range(self.batch):
            frame = frames[b]
            l_f = frame.shape[0]
            if l_f < max_time:
                pad_size = int(np.abs(max_time - l_f))
                frames[b] = np.pad(frame, ((0,pad_size),(0,0)), 'constant', constant_values=(1e-5))

        frames = np.concatenate(frames,axis=1)
        # Reshape in order to make it time_steps x batch_size x frame_length
        frames = np.reshape(frames,newshape=[-1,self.batch,(self.mfcc_coeff/2)*3])
        return frames, frame_size

    def rep_dense(self,transcript,charmap,noise_types):
        dense = None
        try:
            transcript = transcript.split("\n")[0]
            temp = transcript.split(" ")
            if any(temp[0] == noise_types):
                #dense = [charmap[w] for w in temp if noise_types.__contains__(w)]
                # dense = charmap[transcript]
                dense = [charmap[" "]]
            else:
                dense = [charmap[l] for l in transcript if all(l != noise_types)]
        except:
            err = sys.exc_info()
            print(err)
            print(transcript)
            print("Catch me")
        return dense

# train_path = '../../../Data/OpenSLR/data_voip_en/train'
# data_train = Data(299, train_path,
#                      'train_list.npy',
#                       frame_overlap_flag=True,
#                       ms_to_sample=25)
# arr = data_train.next_batch()
# print("Finished")