from cmath import cos, exp, pi


Q = 0.5


def rectangular_window(n, frame_size):
    return 1


def gaussian_window(n, frame_size):
    a = (frame_size-1)/2
    t = (n - a)/(Q*a)
    t *= t
    return exp(-t/2)


def hamming_window(n, frame_size):
    return 0.54 - 0.46 * cos((2*pi*n)/(frame_size-1))


def hann_window(n, frame_size):
    return 0.5 * (1-cos((2*pi*n)/(frame_size-1)))


def blackmann_harris_window(n, frame_size):
    return 0.35875 - (0.48829 * cos((2*pi*n)/(frame_size-1))) + \
           (0.14128 * cos((4*pi*n)/(frame_size-1))) - (0.01168 * cos((4*pi*n)/(frame_size-1)))
