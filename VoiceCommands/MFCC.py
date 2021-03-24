import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

import scipy
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
import IPython.display as ipd
import matplotlib.pyplot as plt


class MFCC:

    def __init__(self):
        None

    
    def getDataDir(self):
        data_dir = pathlib.Path('data/mini_speech_commands')
        return data_dir
    
    @staticmethod
    def getTrainFiles():
        filenames= getFileNames()
        train_files = filenames[:6400]
        return train_files

    @staticmethod
    def getValFiles():
        filenames= getFileNames()
        val_files = filenames[6400: 6400 + 800]
        return val_files

    @staticmethod    
    def getTestFiles():
        filenames= getFileNames()
        test_files = filenames[-800:]
        return test_files   

    @staticmethod 
    def getNumSamples():
        num_samples = len(getFileNames)
        return num_samples
            

    def getFileNames(self):
        mfcc = MFCC()
        data_dir = mfcc.getDataDir()
        filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
        filenames = tf.random.shuffle(filenames)
        return filenames

    def getCommands(self):
        mfcc = MFCC()
        data_dir = mfcc.getDataDir()
        commands = np.array(tf.io.gfile.listdir(str(data_dir)))
        commands = commands[commands != 'README.md']
        return commands

    def decode_audio(self, audio_binary):
        audio, _ = tf.audio.decode_wav(audio_binary)
        return tf.squeeze(audio, axis=-1)

    def get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        # Note: You'll use indexing here instead of tuple unpacking to enable this
        # to work in a TensorFlow graph.
        return parts[-2]

    def get_waveform_and_label(self, file_path):
        mfcc = MFCC()
        label = mfcc.get_label(file_path)
        audio_binary = tf.io.read_file(file_path)
        waveform = mfcc.decode_audio(audio_binary)
        return waveform, label

    # getAudioPath ~ gets audio file path
    # Param: index of audio file
    # Returns: returns resulting file path
    def getAudioPath(self, num):
        mfcc = MFCC()
        data_dir = mfcc.getDataDir()
        commands = mfcc.getCommands()
        TRAIN_PATH = 'data/mini_speech_commands/down/'
        TRAIN_FILE = tf.io.gfile.listdir(str(data_dir/commands[0]))
        TRAIN_PATH = (TRAIN_PATH + TRAIN_FILE[num])
        return TRAIN_PATH


    def getAudio(self,num):
        mfcc = MFCC()
        global audio
        audio = wavfile.read(mfcc.getAudioPath(num))
        return audio

    def frame_audio(audio, FFT_size=2048, hop_size=10, sample_rate=44100):
        # hop_size in ms

        audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
        frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
        frame_num = int((len(audio) - FFT_size) / frame_len) + 1
        frames = np.zeros((frame_num, FFT_size))

        for n in range(frame_num):
            frames[n] = audio[n*frame_len:n*frame_len+FFT_size]

        return frames
    
    def freq_to_mel(freq):
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    def met_to_freq(mels):
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
        fmin_mel = freq_to_mel(fmin)
        fmax_mel = freq_to_mel(fmax)

        print("MEL min: {0}".format(fmin_mel))
        print("MEL max: {0}".format(fmax_mel))

        mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
        freqs = met_to_freq(mels)

        return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs

    def get_filters(filter_points, FFT_size):
        filters = np.zeros((len(filter_points)-2, int(FFT_size/2+1)))

        for n in range(len(filter_points)-2):
            filters[n, filter_points[n]: filter_points[n + 1]
                    ] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
            filters[n, filter_points[n + 1]: filter_points[n + 2]
                    ] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])

        return filters
    
    def dct(self, dct_filter_num, filter_len):
        basis = np.empty((dct_filter_num, filter_len))
        basis[0, :] = 1.0 / np.sqrt(filter_len)

        samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

        for i in range(1, dct_filter_num):
            basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)

        return basis
    
    def getPrints(self):
        mfcc = MFCC()
        filenames = mfcc.getFileNames()
        num_samples = mfcc.getNumSamples()
        train_files = mfcc.getTrainFiles()
        val_files = mfcc.getValFiles()
        test_files = mfcc.getTestFiles()
        commands = mfcc.getCommands()
        audio_framed = mfcc.getAudioFramed()
        data_dir = mfcc.getDataDir()
        print('Number of total examples:', num_samples)
        print('Number of examples per label:',
        len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
        print('Example file tensor:', filenames[0])
        
        print('Training set size', len(train_files))
        print('Validation set size', len(val_files))
        print('Test set size', len(test_files))
        
        print("Train path\n")
        print(mfcc.getAudioPath(1))

        sample_rate, audio = wavfile.read(mfcc.getAudioPath(1))
        print("Sample rate: {0}Hz".format(sample_rate))
        print("Audio duration: {0}s".format(len(audio) / sample_rate))

        print("Framed audio shape: {0}".format(audio_framed.shape))

        print("Minimum frequency: {0}".format(freq_min))
        print("Maximum frequency: {0}".format(freq_high))


    def getPlots(self):
        mfcc = MFCC()
        waveform_ds = mfcc.getWaveformDS()
        rows = 3
        cols = 3
        n = rows*cols
        fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
        for i, (audio, label) in enumerate(waveform_ds.take(n)):
            r = i // cols
            c = i % cols
            ax = axes[r][c]
            ax.plot(audio.numpy())
            ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
            label = label.numpy().decode('utf-8')
            ax.set_title(label)
        plt.show()

        ind = 65  # Why this number?
        plt.figure(figsize=(15, 6))
        plt.subplot(2, 1, 1)
        plt.plot(audio_framed[ind])
        plt.title('Original Frame')
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(audio_win[ind])
        plt.title('Frame After Windowing')
        plt.grid(True)
                
        plt.figure(figsize=(15, 4))
        for n in range(filters.shape[0]):
            plt.plot(filters[n])

        # taken from the librosa library

        plt.figure(figsize=(15, 4))
        for n in range(filters.shape[0]):
            plt.plot(filters[n])

        plt.figure(figsize=(15, 5))
        plt.plot(np.linspace(0, len(audio) / sample_rate, num=len(audio)), audio)
        plt.imshow(cepstral_coefficents, aspect='auto', origin='lower')

    def setup(self):
        mfcc = MFCC()
        global INDEX, seed, AUTOTUNE, files_ds, waveform_ds, filenames, num_samples
        INDEX = 1
        # Set seed for experiment reproducibility
        seed = 42
        tf.random.set_seed(seed)
        np.random.seed(seed)
        mfcc.getNumSamples()
        mfcc.getFileNames()
        AUTOTUNE = tf.data.AUTOTUNE
        files_ds = tf.data.Dataset.from_tensor_slices(train_files)
        waveform_ds = files_ds.map(
            mfcc.get_waveform_and_label, num_parallel_calls=AUTOTUNE)

    def getFFTSize():
        FFT_size = 2048
        return FFT_size

    def getSampleRate():
        sample_rate = 44100
        return sample_rate
    
    def getHopSize(self,size):
        hop_size = size
        return size

    def getAudioFramed():
        mfcc = MFCC()
        audio = mfcc.getAudio(1)
        FFT_size = mfcc.getFFTSize()
        hop_size = mfcc.getHopSize(15)
        sample_rate = mfcc.getSampleRate()
        audio_framed = mfcc.frame_audio(audio, FFT_size=FFT_size,
                           hop_size=hop_size, sample_rate=sample_rate)  


    def genFTT():
        mfcc = MFCC()
        hop_size = 15  # ms
        FFT_size = 2048
        sample_rate = 44100
        audio = mfcc.getAudio(1)
        audio_framed = mfcc.frame_audio(audio, FFT_size=FFT_size,
                           hop_size=hop_size, sample_rate=sample_rate)     
        FFT_size = 2048
        hop_size = 10
       
        np.pad(audio, int(FFT_size / 2), mode='reflect').shape
        window = get_window("hann", FFT_size, fftbins=True)
        audio_win = audio_framed * window
        audio_winT = np.transpose(audio_win)

        audio_fft = np.empty(
            (int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')

        for n in range(audio_fft.shape[1]):
            audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]

        audio_fft = np.transpose(audio_fft)

        audio_power = np.square(np.abs(audio_fft))
        print(audio_power.shape)

        freq_min = 0
        freq_high = sample_rate / 2
        mel_filter_num = 10

    def getFreqMin(self):
        freq_min = 0
        return freq_min

    def getFreqHigh(self):
        mfcc = MFCC()
        sample_rate = mfcc.getSampleRate()
        freq_high = sample_rate / 2
        return freq_high

    def getMelFilterNum(self):
        mel_filter_num = 10
        return mel_filter_num

    def getCC(self):
        mfcc = MFCC()
        FFT_size = 2048
        hop_size = 10
        sample_rate = 44100
        audio = mfcc.getAudio(1)
        audio_framed= mfcc.getAudioFramed()
        freq_min = mfcc.getFreqMin()
        freq_high = mfcc.getFreqHigh()
        mel_filter_num = mfcc.getMelFilterNum()
        
        np.pad(audio, int(FFT_size / 2), mode='reflect').shape

        window = get_window("hann", FFT_size, fftbins=True)
        audio_win = audio_framed * window
        filter_points, mel_freqs = get_filter_points(
        freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=44100)
        filter_points
        filters = get_filters(filter_points, FFT_size)
        enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
        filters *= enorm[:, np.newaxis]
        audio_filtered = np.dot(filters, np.transpose(audio_power))
        audio_log = 10.0 * np.log10(audio_filtered)
        audio_log.shape
        dct_filter_num = 40
        dct_filters = dct(dct_filter_num, mel_filter_num)
        cepstral_coefficents = np.dot(dct_filters, audio_log)
        cepstral_coefficents.shape

        return cepstral_coefficents[:, 0]
    
mfcc = MFCC()
filenames= mfcc.getFileNames()
AUTOTUNE = tf.data.AUTOTUNE
train_files = filenames[:6400]
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(
        mfcc.get_waveform_and_label, num_parallel_calls=AUTOTUNE)


data_dir = mfcc.getDataDir()
if not data_dir.exists():
    tf.keras.utils.get_file(
        'mini_speech_commands.zip',
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        extract=True,
        cache_dir='.', cache_subdir='data')




