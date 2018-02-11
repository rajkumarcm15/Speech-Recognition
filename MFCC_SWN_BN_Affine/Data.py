"""
End-End Speech Recognition Application using Recurrent Neural Network
Name: Rajkumar Conjeevaram Mohan
Email: Rajkumar.Conjeevaram-Mohan14@imperial.ac.uk

This program follows a simple recipe from Alex Graves et. al 2015 paper
( Towards End-End Speech Recognition using Recurrent Neural Networks )
that incorporates Connectionist Temporal Classification Network with
RNN to obtain ability to compute loss without prior pre-segmentation.

"""

from multiprocessing.pool import ThreadPool
from os.path import isfile, join
from python_speech_features import logfbank
from python_speech_features import mfcc
from scipy.io import wavfile
from matplotlib import mlab
from os import listdir
import numpy as np
import time
import sys
# import gc


class Data:
    """
    Audio signal processing entity that computes
    1. log-mel filter bank appended with energy, delta, and double delta
    2. Pure FFT Spectrogram
    3. Power Spectral Density
    """

    noise_types = np.array(["_NOISE_","_INHALE_","_EHM_HMM_","_LAUGH_"])
    frame_overlap_flag = False
    reverse_map = None
    frame_rate = None
    file_path = None
    charmap = None
    batch = None
    n_files = 0
    ON = True
    mode = 0
    b_id = 0

    # For multi-threading purpose
    thread = None

    def __init__(self, batch_size,
                 file_path,
                 filelist_filename,
                 mode,
                 mfcc_coeff = 26,
                 frame_overlap_flag=False,
                 ms_to_sample=25,
                 overlap_ms=10):

        self.batch = batch_size
        self.mfcc_coeff = mfcc_coeff
        self.mode = mode
        self.file_path = file_path
        self.frame_overlap_flag = frame_overlap_flag
        self.ms_to_sample = ms_to_sample
        self.overlap_ms = overlap_ms
        if not isfile(filelist_filename):
            self.files = [f for f in listdir(file_path) if isfile(join(file_path, f)) if f.endswith("wav")]
            np.save(filelist_filename, self.files)
        else:
            self.files = np.load(filelist_filename)
        self.safe_training_files = self.files
        self.files = self.files[:50]
        self.n_files = len(self.files)

        # Assuming the whole batch would have the same properties
        # we sample the properties from an arbitrary file
        self.frame_rate, _ = wavfile.read(join(file_path, self.files[0]))
        self.frame_length = (self.frame_rate / 1000) * ms_to_sample
        self.sides = 'onesided'
        if self.sides == 'twosided':
            self.retain_fft = self.frame_length
        else:
            self.retain_fft = (self.frame_length / 2) + 1
        self.overlap = self.frame_rate / 1000 * self.overlap_ms
        # self.default_window = np.hamming(self.frame_length + self.overlap)

        # Build char map for dense representation of transcripts
        chars = []
        # Add blank label - BUT THIS TIME ONLY FOR IDENTIFYING NOISE
        chars.append(chr(32))
        for i in range(65, 91):
            chars.append(chr(i))
        # Special characters like punctuations
        chars.append(chr(39))
        # Enable this when want the model to learn noise rep as well
        #for n in self.noise_types:
        #   chars.append(n)

        indices = range(len(chars))
        self.charmap = dict(zip(chars, indices))
        self.reverse_map = dict(zip(indices, chars))
        # self.start_thread()

    def next_batch(self):
        """
        Function that process the next mini-batch asynchronously, which
        is what should be called for any purpose with the feature data
        :param self: Instance of Data
        :return: frames
        """
        # data = self.thread.get()
        # self.start_thread()
        # return data
        return self.process_batch(self.file_path, self.charmap,
                                  self.noise_types, self.frame_overlap_flag)

    def start_thread(self):
        pool = ThreadPool(processes=1)
        self.thread = pool.apply_async(self.process_batch,(self.file_path, self.charmap,
                                       self.noise_types, self.frame_overlap_flag))

    def get_delta(self,frames):
        """
        Function used to compute the first and the second order
        temporal derivatives
        :param frames: 2D array that represents feature vectors of shape [frames,feature_size]
        :return: 2D array of derivative
        """
        delta = np.zeros(frames.shape)
        frames = np.pad(frames,((1,1),(0,0)),'edge')
        for t in range(1,frames.shape[0]-2):
            delta[t,:] = frames[t+1,:] - frames[t-1,:]
        return delta

    def get_featvec(self,time_signal,frame_overlap_flag,mode):
        """
        Computes the feature vectors for a given audio signal
        in time domain
        :param self: Instance of Data
        :param time_signal: 1D audio signal in time domain
        :param frame_overlap_flag: If set to true,
        then there will be overlap between successive frames
        :param mode: Specifies the type of feature vector
         1 = Spectrogram
         2 = Log Mel-Filter Bank
         3 = Mel-Frequency Cepstral Coefficients
        :return: feature vectors of shape [frames,features]
        """
        ms_to_sample = float(self.ms_to_sample) / 1000
        overlap_ms = float(self.overlap_ms) / 1000
        overlap = self.overlap
        if not frame_overlap_flag:
            overlap = 0
            overlap_ms = 0

        data = None
        if mode == 1:
            # Compute Spectrogram
            data, _, _ = mlab.specgram(time_signal, NFFT=self.frame_length, Fs=self.frame_rate,
                                       window=mlab.window_hanning, sides=self.sides, noverlap=overlap)
            data = data.T
            data = np.abs(data)**2
        elif mode == 2:
            # Compute Log Mel-filter bank
            data = logfbank(time_signal, samplerate=self.frame_rate, winlen=ms_to_sample,
                            winstep=overlap_ms, nfilt=self.mfcc_coeff, nfft=self.frame_length)
            delta = self.get_delta(data)
            ddelta = self.get_delta(delta)
            data = np.concatenate([data, delta, ddelta], axis=1)
        elif mode == 3:
            # Compute Mel-Frequency Cepstrum Coefficients
            # Previously nfft was set to self.frame_length, which is incorrect.
            # nfft is the analysis window, which is not the same as frame length
            # and the recommended value as per
            # http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
            # is 512, so it has been corrected to 512

            data = mfcc(time_signal, samplerate=self.frame_rate, winlen=ms_to_sample,
                        winstep=overlap_ms, numcep = self.mfcc_coeff/2, nfilt=self.mfcc_coeff,
                        nfft=512, appendEnergy=True, winfunc=np.hamming)
            #delta = self.get_delta(data)
            #ddelta = self.get_delta(delta)
            #data = np.concatenate([data, delta, ddelta], axis=1)
        #gc.collect()

        return data, data.shape[0]

    def arrays_to_tensor(self,arrays,seq_len):
        """
        Converts list of variable size arrays to a tensor
        of shape [max_time,batch_size,feature_size]
        :param self: Instance of Data
        :param arrays: List of arrays in which each element is a
        2D feature vectors of an audio clip
        :param seq_len: 1D array in which each element represent
        a length of time series, i.e. frames, in every data of
        batch. dim: [batch_size]
        :return: A 3D tensor of shape [max_time,batch_size,feature_size]
        """
        f_size = arrays[0].shape[1]
        max_time = np.max(seq_len)
        n_data = len(arrays)

        for i in range(n_data):
            e_i = seq_len[i]
            d = arrays[i]
            mat = np.zeros([max_time,f_size])
            mat[:e_i,:] = d
            arrays[i] = mat
            del mat, d, e_i

        arrays = np.concatenate(arrays,axis=1)
        arrays = np.reshape(arrays,[max_time,-1,f_size])
        #gc.collect()
        return arrays, max_time

    def get_train_targ_data(self, file_path, charmap, noise_types, frame_overlap_flag, s_i, e_i, th_id, data_thread):
        """
        This function was implemented to parallelise the loop over every
        data in the mini-batch
        :param self:
        :param file_path: base directory of the train/val/test data
        :param charmap: dictionary, whose keys and values represent
        characters, and its ids respectively.
        :param noise_types: 1D array that enlists the type of noise found in
         the data
        :param frame_overlap_flag:
        :param s_i: start index of the ids in self.files for the subset of current mini-batch
        :param e_i: end index of the ids in self.files for the subset of current mini-batch
        :param th_id: thread id
        :return: 2D frames, 1D sequence_length, 2D target_indices
        """
        #---------------------------------------------
        # Temporary parameters for the current routine
        # ---------------------------------------------
        targ_indices = []
        transcript = []
        targ_len = []
        nframes = []
        frames = []
        count = 0
        # ---------------------------------------------

        for f in range(s_i,e_i):
            # Training data
            # (self, time_signal, frame_overlap_flag, mode)
            temp_data, n_frames = self.get_featvec(wavfile.read(join(file_path, self.files[f]))[1],
                                                   frame_overlap_flag, self.mode)
            frames.append(temp_data)
            del temp_data
            nframes.append(n_frames)
            del n_frames
            # Target data
            target_file = self.files[f] + ".trn"
            file_ = open(join(file_path, target_file), 'r')
            temp = file_.read()
            file_.close()
            del file_

            temp = self.rep_dense(temp, charmap, noise_types)
            t_len = len(temp)
            targ_len.append(t_len)
            # Indices made for SparseTensor
            temp_indices = np.zeros([t_len, 2])  # 2 -> 2d indices
            b_id = (th_id*data_thread)+count
            # """
            # Debug------------------------------
            # """
            # print("\nb_id: %d, file: %s"%(b_id,join(file_path, self.files[f])))
            # print("transcript: %s \n"%temp)
            # """
            # End of debug-----------------------
            # """

            temp_indices[:, 0] = b_id
            temp_indices[:, 1] = range(t_len)
            targ_indices.append(temp_indices)
            # ------------------------------
            transcript.append(temp)
            count += 1
            #gc.collect()

        frames, max_time = self.arrays_to_tensor(frames,nframes)
        targ_indices = np.concatenate(targ_indices, axis=0)
        # Value for SparseTensor
        targ_values = np.concatenate(transcript, axis=0)
        # Shape for SparseTensor
        targ_shape = [frames.shape[1], np.max(targ_len)]
        #gc.collect()
        return frames,max_time,nframes,transcript,targ_indices,targ_values,targ_shape

    def append_tensor(self,tensors,new_tensor):

        if tensors == None:
            return new_tensor

        old_tsize = tensors.shape[0]
        new_tsize = new_tensor.shape[0]
        old_bsize = tensors.shape[1]
        new_bsize = new_tensor.shape[1]
        f_size = new_tensor.shape[2]

        if old_tsize == new_tsize:
            return np.concatenate([tensors,new_tensor],axis=1)
        elif old_tsize > new_tsize:
            tensor = np.zeros([old_tsize,old_bsize+new_bsize,f_size])
            tensor[:,:old_bsize,:] = tensors
            tensor[:new_tsize,old_bsize:old_bsize+new_bsize,:] = new_tensor
            #gc.collect()
            return tensor
        else:
            tensor = np.zeros([new_tsize,old_bsize+new_bsize,f_size])
            tensor[:old_tsize,:old_bsize,:] = tensors
            tensor[:,old_bsize:old_bsize+new_bsize,:] = new_tensor
            #gc.collect()
            return tensor

    def process_batch(self, file_path, charmap, noise_types, frame_overlap_flag):
        if (self.b_id + self.batch) >= self.n_files:
            self.b_id = 0

        n_threads = 2
        pool = ThreadPool(processes=n_threads)
        # data per thread
        data_thread = int(np.floor(float(self.batch)/n_threads))
        # remaining data for last thread
        rem_data_lth = self.batch - (data_thread*(n_threads-1))
        threads = []
        for p in range(n_threads):

            if p == n_threads-1:
                if (self.b_id+rem_data_lth) >= self.n_files:
                    self.b_id = 0
            else:
                if (self.b_id+data_thread) >= self.n_files:
                    self.b_id = 0

            s_i = self.b_id
            if rem_data_lth != 0 and p == n_threads-1:
                e_i = s_i + rem_data_lth
            else:
                e_i = s_i + data_thread
            # update the self.b_id
            self.b_id = e_i
            # """
            # Debug start---------------
            # """
            # print("s_i: %d, e_i: %d"%(s_i,e_i))
            # """
            # Debug end-----------------
            # """
            threads.append(pool.apply_async(self.get_train_targ_data,args=(file_path, charmap, noise_types,
                                                                           frame_overlap_flag, s_i, e_i,p,
                                                                           data_thread)))
        #gc.collect()
        #-----------------------------------------------------------
        # Variables for collapsing results from different processes
        #-----------------------------------------------------------
        frames = None
        seq_lens = []
        transcripts = []
        t_indices = []
        t_values = []
        t_shape = []
        # ----------------------------------------------------------
        for p in range(n_threads):
            data, _, nframes, \
            transcript, targ_indices, \
            targ_values, targ_shape = threads[p].get()
            # Erase the memory in threads[p]
            threads[p] = None

            frames = self.append_tensor(frames,data)
            seq_lens.append(nframes)
            t_shape.append([targ_shape])
            transcripts.append(transcript)
            t_indices.append(targ_indices)
            t_values.append(targ_values)
            del data,nframes,transcript,targ_indices,targ_values,targ_shape

        pool.close()
        pool.join()
        pool._join_exited_workers()
        t_indices = np.concatenate(t_indices)
        t_values = np.concatenate(t_values)
        t_shape = np.concatenate(t_shape)
        t_shape = [self.batch,np.max(t_shape[:,1])]
        transcripts = [t for sublist in transcripts for t in sublist]
        seq_lens = np.concatenate(seq_lens)
        self.b_id += self.batch
        return frames,transcripts,t_indices,t_values,t_shape,seq_lens

    def rep_dense(self,transcript,charmap,noise_types):
        dense = None
        try:
            transcript = transcript.split("\n")[0]
            temp = transcript.split(" ")
            if any(temp[0] == noise_types):
                # dense = [charmap[w] for w in temp if noise_types.__contains__(w)]
                dense = [charmap[" "]]
            else:
                dense = [charmap[l] for l in transcript if all(l != noise_types)]
        except:
            err = sys.exc_info()
            print(err)
            print(transcript)
            raise Exception("Error in rep_dense")
        #gc.collect()
        return dense

# Debug phase----------------
# import memory_profiler
# train_path = '../../Data/OpenSLR/data_voip_en/train'
# # start = time.time()
#
# data_train = Data(50, train_path,
#                      'train_list.npy',
#                       mode=3,
#                       frame_overlap_flag=True,
#                       overlap_ms=10,
#                       ms_to_sample=25)
# arr = data_train.next_batch()
# print("Finished")
