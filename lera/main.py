import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import librosa
import librosa.display as dsp

import denoise
from fft import fft, stft


def to_db(y):
    return librosa.amplitude_to_db(y, ref=np.max)


# complex numbers stuff
z = 2 + 3j
real = z.real
imaginary = z.imag
conjugate = z.conjugate()
mag = abs(z)


def fft_plot(signal, sr):
    n = len(audio)
    duration = librosa.get_duration(y=audio, sr=sr)

    ft = fft(audio)
    # magnitude_spectrum = np.abs(
    #     denoise.normalise(ft))  # redo denoise https://www.kaggle.com/code/theoviel/fast-fourier-transform-denoising
    magnitude_spectrum2 = np.abs(np.fft.fft(audio))
    frequency = np.linspace(0, sr, len(magnitude_spectrum2))

    time_ax = np.linspace(0, duration, n)

    fig, [ax1, ax2] = plt.subplots(2, figsize=(7.5, 6))
    plt.subplots_adjust(hspace=0.5)

    fig.suptitle("Fast Fourier Transform", fontsize=16)

    ax1.plot(time_ax, audio, linewidth=1)
    ax1.set_title("Time-Domain")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.margins(x=0)

    # ax2.plot(frequency, magnitude_spectrum, linewidth=1)
    # ax2.plot(frequency, magnitude_spectrum2, linewidth=1, color="y")
    ax2.plot(frequency, magnitude_spectrum2, linewidth=1)
    # ax2.magnitude_spectrum(audio, Fs=sr, linewidth=1)
    ax2.set_title("Frequency-Domain")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_xscale("log")
    ax2.set_xlim([20, 20000])
    ax2.margins(x=0)

    plt.show()


def stft_plot(signal, sr):
    img = plt.imshow(stft(signal, sr), origin="lower", cmap="jet", interpolation="nearest", aspect="auto")
    plt.show()


def spectrogram(signal, sr):

    fft_size = 2**11
    window_index = 0

    d = stft(signal, sr)

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    fig.suptitle("Short Time Fourier Transform", fontsize=16)

    plt.subplots_adjust(bottom=0.2)

    img = dsp.specshow(d, y_axis="linear", x_axis="s", sr=sr, ax=ax[0])
    ax[0].set(title="Linear Frequency Power Spectrogram")
    ax[0].label_outer()

    dsp.specshow(d, y_axis="log", x_axis="s", sr=sr, ax=ax[1])
    ax[1].set(title="Log Frequency Power Spectrogram")
    ax[1].label_outer()

    ax_fft_size = plt.axes([0.25, 0.05, 0.5, 0.03])
    fft_size_slider = Slider(
        ax=ax_fft_size,
        label="FFT Size [Samples]",
        valmin=2**6,
        valmax=2**13,
        valinit=fft_size,
        valstep=128
    )

    def fft_update(val):
        nonlocal fft_size
        fft_size = val
        d = stft(audio, sr, fft_size=fft_size, window_index=window_index)

        ax[0].clear()
        img = dsp.specshow(d, y_axis="linear", x_axis="s", sr=sr, ax=ax[0])
        ax[0].set(title="Linear Frequency Power Spectrogram")
        ax[0].label_outer()

        ax[1].clear()
        dsp.specshow(d, y_axis="log", x_axis="s", sr=sr, ax=ax[1])
        ax[1].set(title="Log Frequency Power Spectrogram")
        ax[1].label_outer()

    fft_size_slider.on_changed(fft_update)

    ax_window_type = plt.axes([0.25, 0.1, 0.5, 0.03])
    window_type_slider = Slider(
        ax=ax_window_type,
        label="Window Type",
        valmin=0,
        valmax=3,
        valinit=window_index,
        valstep=1
    )

    def window_update(val):

        nonlocal window_index
        window_index = val
        d = stft(audio, sr, fft_size=fft_size, window_index=window_index)

        ax[0].clear()
        img = dsp.specshow(d, y_axis="linear", x_axis="s", sr=sr, ax=ax[0])
        ax[0].set(title="Linear Frequency Power Spectrogram")
        ax[0].label_outer()

        ax[1].clear()
        dsp.specshow(d, y_axis="log", x_axis="s", sr=sr, ax=ax[1])
        ax[1].set(title="Log Frequency Power Spectrogram")
        ax[1].label_outer()

    window_type_slider.on_changed(window_update)

    fig.colorbar(img, ax=ax, format="%+2.f dB")

    plt.show()


if __name__ == '__main__':
    audio, sr = librosa.load("audio/intheend.wav", sr=None)
    # audio, sr = librosa.load("audio/d#m_guitar_chord.wav")
    # audio, sr = librosa.load(librosa.util.example("brahms"), duration=60)
    # fft_plot(audio, sr)
    # stft_plot(audio, sr)
    spectrogram(audio, sr)

