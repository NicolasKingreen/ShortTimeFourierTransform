def normalise(signal):
    o = []
    for p in signal:
        res = median(p)
        res = easy_mean(res)
        o.append(res)
    return o


def median(signal):
    # Creating buffer
    if not hasattr(median, "buffer"):
        median.buffer = [signal] * 3

    # Move buffer to actually values ( [0, 1, 2] -> [1, 2, 3] )
    median.buffer = median.buffer[1:]
    median.buffer.append(signal)

    # Calculation median
    a = median.buffer[0]
    b = median.buffer[1]
    c = median.buffer[2]
    middle = max(a, c) if (max(a, b) == max(b, c)) else max(b, min(a, c))

    return middle


def easy_mean(signal, s_k=0.2, max_k=0.9, d=1.5):
    # Creating static variable
    if not hasattr(easy_mean, "fit"):
        easy_mean.fit = signal

    # Adaptive ratio
    k = s_k if (abs(signal - easy_mean.fit) < d) else max_k

    # Calculation easy mean
    easy_mean.fit += (signal - easy_mean.fit) * k

    return easy_mean.fit
