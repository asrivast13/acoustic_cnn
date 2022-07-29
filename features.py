import glob
import os
import time

import numpy as np
import soundfile as sf
import pydub

from constants import *
import common

def read_audio(f):
    ext = os.path.splitext(f)[-1].lower()

    if ext == '.flac' or ext == '.wav':
        return sf.read(f)
    elif ext == '.mp3':
        return read_mp3(f)
    else:
        sys.exit(-1)


def read_mp3(f, normalized=True):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return np.float32(y) / 2**15, a.frame_rate
    else:
        return y, a.frame_rate

def generate_fb_and_mfcc(signal, sample_rate):

    # Pre-Emphasis
    pre_emphasis = 0.97
    emphasized_signal = np.append(
        signal[0],
        signal[1:] - pre_emphasis * signal[:-1])

    # Framing
    frame_size = 0.025
    frame_stride = 0.01

    # Convert from seconds to samples
    frame_length, frame_step = (
        frame_size * sample_rate,
        frame_stride * sample_rate)
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))

    # Make sure that we have at least 1 frame
    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))

    # Pad Signal to make sure that all frames have equal
    # number of samples without truncating any samples
    # from the original signal
    pad_signal = np.append(emphasized_signal, z)

    indices = (
        np.tile(np.arange(0, frame_length), (num_frames, 1)) +
        np.tile(
            np.arange(0, num_frames * frame_step, frame_step),
            (frame_length, 1)
        ).T
    )
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Window
    frames *= np.hamming(frame_length)

    # Fourier-Transform and Power Spectrum
    NFFT = 512

    # Magnitude of the FFT
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))

    # Power Spectrum
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

    # Filter Banks
    nfilt = 40

    low_freq_mel = 0

    # Convert Hz to Mel
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))

    # Equally spaced in Mel scale
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)

    # Convert Mel to Hz
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)

    # Numerical Stability
    filter_banks = np.where(
        filter_banks == 0,
        np.finfo(float).eps,
        filter_banks)

    # dB
    filter_banks = 20 * np.log10(filter_banks)

    return filter_banks


def process_audio(input_dir, output_dir=".", debug=False, SeqLength=WIDTH, NumFeats=FB_HEIGHT):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = []

    extensions = ['*.mp3', '*.flac', '*.wav']
    for extension in extensions:
        files.extend(glob.glob(os.path.join(input_dir, extension)))

    totalfiles = len(files)
    #print (input_dir)
    #print (len(files))
    fnum = 1
    start = time.time()
    totalProcTime = 0

    for file in files:
        signal, sample_rate = read_audio(file)

        procStartTime = time.time()

        if debug:
            print('Sample Rate: %d' % sample_rate)
            print('Length of signal: %d' % len(signal))
            print(signal)

        assert len(signal) > 0
        assert sample_rate >= 8000

        fb = generate_fb_and_mfcc(signal, sample_rate)
        fb = fb.astype(DATA_TYPE, copy=False)

        if debug:
            print(fb.dtype)
            print(fb.shape)

        feats = fb
        if fb.shape != (SeqLength, NumFeats):
            if fb.shape[0] > SeqLength:
                feats = fb[0:SeqLength,...]
            else:
                deltax = SeqLength - fb.shape[0]
                deltay = NumFeats  - fb.shape[1]
                assert deltax >= 0
                assert deltay >= 0
                feats = np.pad(fb, ((0, deltax), (0, deltay)), 'constant', constant_values=0)

        assert feats.dtype == 'float32'
        assert feats.shape == (SeqLength, NumFeats)

        procTime = time.time() - procStartTime
        totalProcTime += procTime

        # .npz extension is added automatically
        file_without_ext = os.path.splitext(os.path.basename(file))[0]
        output_file = os.path.join(output_dir, file_without_ext + '.fb')

        np.savez_compressed(output_file, data=feats)
        print("[{} of {}] Finished writing features  in {:.2f} msecs to: {}".format(fnum, totalfiles, (procTime*1000), output_file))
        fnum += 1

        if debug:
            end = time.time()
            print("It took [s]: ", end - start)
            break

    end = time.time()
    print("Entire feature extraction on {} files in {} took: {:.2f} secs; Average: {:.2f} msecs".format(totalfiles, input_dir, (end - start), (totalProcTime*1000/totalfiles)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate various features from audio samples.')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)

    args = parser.parse_args()

    if args.debug:
        process_audio(os.path.join(common.DATASET_DIST, 'test'), os.path.join(common.FEATS_DIST, 'debug'), debug=True)
    else:
        process_audio(os.path.join(common.DATASET_DIST, 'test'),  os.path.join(common.FEATS_DIST, 'test',  debug=False, SeqLength=WIDTH, NumFeats=FB_HEIGHT))
        process_audio(os.path.join(common.DATASET_DIST, 'train'), os.path.join(common.FEATS_DIST, 'train', debug=False, SeqLength=WIDTH, NumFeats=FB_HEIGHT))
