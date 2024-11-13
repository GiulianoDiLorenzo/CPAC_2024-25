# import 
import librosa
import numpy as np

def compute_beats(y, sr):
    """This function uses librosa library to compute beats from an audio signal
    and returns the time index where the beats occur 

    Parameters
    
    ----------
    y : np.ndarray
        time-domain audio signal
    sr : int, float
        samplerate

    Returns
    -------
    np.ndarray
        sample index where beat occurs
    """
    
    # beat tracking function
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='samples')
    return beats

def add_samples(y, sample, beats):
    """Add a sample to an audio signal at given beats 

    Parameters
    ----------
    y : np.ndarray
        the original signal
    sample : np.ndarray
        the sample beat to add
    beats : np.ndarray
        the time beats

    Returns
    -------
    np.ndarray
        original signal + sample on beats
    """

    # samples addition
    s = np.zeros(y.shape)
    slen = sample.size
    for b in beats:
        s[b:b+slen] += sample
    
    return y + s

