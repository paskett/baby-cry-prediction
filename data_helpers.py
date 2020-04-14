"""Data Helpers

These functions will help us load the data, augment it (cutting, duplicating,
appyling FFT, etc.), and convert it to Pandas dataframes.
"""
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.io import wavfile
from scipy.fftpack import fft, ifft, dct, idct

def load_data(directory='raw_data'):
    """
    Function to load sound data from the donateacry dataset.
    Also normalizes each to have the same length.

    Parameters
    ----------
    directory : str
                directory to search for data
    scale     : bool
                indicates whether to scale data or not

    Returns
    -------
    wavforms : ndarray
               array of data points (each waveform)
    labels : ndarray
             array of labels corresponding to each waveform
    """
    subdirs = os.listdir(directory)
    wavforms = []
    labels = []
    for dirc in subdirs:
        filenames = os.listdir(directory+'/'+dirc)
        for filename in filenames:
            # Get wavform from each file
            wav = wavfile.read(directory+'/'+dirc+'/'+filename)[1]

            # Store wavform and appropriate label
            wavforms.append(wav)
            labels.append(dirc)

    # Find the minimum wavform length.
    lens = set()
    for d in wavforms:
        lens.add(len(d))
    min_len = min(lens)

    # Cut the end off each wavform so they're the same lengths.
    for i,wav in enumerate(wavforms):
        wavforms[i] = wav[:min_len]

    return np.array(wavforms), np.array(labels)

def duplicate_data(data, labels, alpha=0.3333):
    """
    Duplicates data and adds random noise to the duplicates.

    Parameters
    ----------
    data (ndarray): array of waveform values.
    labels (ndarray): array of labels for each row of data.
    alpha (float): a parameter that changes how large the noise may be for
            all waveforms relative to each of their sample values.

    Returns
    -------
    new_data (ndarray): all the same rows as data with noisy rows as well
    new_labels (ndarray): the labels corresponding to the rows of new_data
    """
    new_data = []
    new_labels = []
    for wav, label in zip(data, labels):
        # Sample noise from the normal distribution.
        noise = np.array([np.random.normal(scale=max(abs(x)*alpha, 0)) for x in wav])

        new_wav = wav.astype(np.int64) + noise.astype(np.int64)

        # Rescale the new_wav to make sure no parts go above 32767.
        new_wav = np.array(32767*new_wav/np.max(np.abs(new_wav))).astype(np.int16)

        new_data.append(new_wav)
        new_labels.append(label)

    # Concatenate the old and new data/labels.
    new_data = np.vstack([data, np.array(new_data)])
    new_labels = np.hstack([labels, np.array(new_labels)])

    return new_data, new_labels

def get_fft(data, rate=8000):
    """
    Function to transform wavelength data into frequency data using the FFT

    Parameters
    ----------
    data : ndarray
           Array where each row is the wavelength samples of a single cry instance

    Returns
    -------
    freqs       : ndarray
                  Array whose rows are the sampled frequencies
    freq_domain : domain of frequency values
    """
    freqs = []
    freq_domain = np.arange(0, len(data[0])) * rate / len(data[0])
    freq_domain = freq_domain[:len(freq_domain)//2]

    for i, wav in enumerate(data):
        # Get frequencies using Fourier Transform
        f = np.abs(fft(wav))[:len(freq_domain)]
        # Get real part of frequencies, normalize, and sample every 100
        tmp = (np.real(f) / np.real(max(f)))[::100]
        freqs.append(list(tmp))

    return np.array(freqs), freq_domain[::100]

def labels_to_integers(labels):
    """
    Convert the array of all labels (strings) to integer values.

    Parameters
    ----------
    labels (ndarray): array of strings containing the label name for
        each data point

    Returns
    -------
    new_labels (ndarray): array where labels are replaced with ints
    mapping (dict): a mapping from int -> string showing which integer
        corresponds to which label
    """
    new_labels = np.zeros(labels.shape, dtype=int)
    mapping = {}

    for idx, lab in enumerate(np.unique(labels)):
        new_labels[labels==lab] = idx
        mapping[idx] = lab

    return new_labels, mapping

def cut_data(data, labels, n):
    """
    Takes data and cuts each sample into parts of specified length

    Parameters
    ----------
    data   : ndarray
             array of waveform values of constant length
    labels : ndarray
             array of labels for each row of data
    n.     : int
             desired length of new samples

    Returns
    -------
    new_data   : ndarray
                 array of new cut samples
    new_labels : ndarray
                 array of corresponding labels
    """
    new_data = []
    new_labels = []
    # Get number of possible subsamples from current samples
    subs = data.shape[1]//n

    # Cut each sample
    for i, wav in enumerate(data):
        # Get as many new samples from old sample as possible
        for x in range(subs):
            # Extract subsamples of length n
            new_data.append(wav[x*n:(x+1)*n])
            # Store corresponding label
            new_labels.append(labels[i])

    return np.array(new_data), np.array(new_labels)

def data_to_df(data, labels, freqs):
    """
    Function to convert numpy array of data to a pandas DataFrame

    Parameters
    ----------
    data  : ndarray
            Array whose rows are of the form [[frequencies], 'label']
    label : list or ndarray of shape (n,)
            list of labels corresponding to each row of data
    freqs : list or ndarray of shape (n,)
            Frequency domain of the passed data (used to make column names)

    Returns
    -------
    df : pd.DataFrame
         DataFrame where each row is a cry instance and each column is a frequency
         with a final 'label' column
    """
    # Data to df
    df = pd.DataFrame(data,columns=freqs)
    # Labels column
    df['labels'] = labels
    return df
