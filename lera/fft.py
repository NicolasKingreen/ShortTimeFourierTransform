from cmath import exp, pi
import numpy as np

windows = [np.hanning, np.hamming, np.blackman, np.bartlett]

# reducible algorithm
def fft(P):
    n = len(P)  # better be a power of 2
    if n == 1:
        return P
    omega = exp((2j*pi)/n)

    P_even = P[0::2]
    P_odd = P[1::2]

    y_even = fft(P_even)
    y_odd = fft(P_odd)

    y = np.zeros(n).astype(np.complex64)
    for j in range(n//2):
        y[j] = y_even[j] + omega**j * y_odd[j]
        y[j+n//2] = y_even[j] - omega**j * y_odd[j]
    return y


def fft2(signal):
    N = len(signal)

    if N == 1:
        return signal
    else:
        X_even = fft2(signal[::2])
        X_odd = fft2(signal[1::2])
        factor = \
            np.exp(-2j*np.pi*np.arange(N)/N)

        X = np.concatenate(
            [X_even+factor[:int(N/2)]*X_odd,
             X_even+factor[int(N/2):]*X_odd])
        return X


def stft(signal, sr, fft_size=2**11, window_index=0, overlap_fac=0.5):

    hop_size = np.int32(np.floor(fft_size * (1-overlap_fac)))
    pad_end_size = fft_size
    total_segments = np.int32(np.ceil(len(signal) / np.float32(hop_size)))
    t_max = len(signal) / np.float32(sr)

    window = windows[window_index](fft_size)
    inner_pad = np.zeros(fft_size)

    proc = np.concatenate((signal, np.zeros(pad_end_size)))
    result = np.empty((total_segments, fft_size), dtype=np.float32)

    for i in range(total_segments):
        current_hop = hop_size * i

        # 1. Выбор сегмента
        segment = proc[current_hop:current_hop+fft_size]
        # 2. Применение оконной функции
        windowed = segment * window
        # 3. Добавление нулей
        padded = np.append(windowed, inner_pad)
        # 4. Преобразование Фурье (FFT)
        spectrum = np.fft.fft(padded) / fft_size
        # 5. Модуль громкости
        autopower = np.abs(spectrum * np.conj(spectrum))

        result[i, :] = autopower[:fft_size]

    # 6. Переход к дБ
    result = 20*np.log10(result)

    # 7. Клиппинг
    result = np.clip(result, -200, 0)

    result = np.rot90(result, 3)
    result = np.flip(result, axis=1)

    return result

