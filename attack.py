import sys
import numpy as np
import theano.tensor as T


def set_params_init(params, values, num_param_to_set=60):
    # calculate number of parameters needed to encode secrets

    if not isinstance(values, np.ndarray):
        values = values.get_value()

    params_to_set = []
    for p in params:
        if p.ndim > 1:
            params_to_set.append(p)
        if len(params_to_set) >= num_param_to_set:
            break

    offset = 0
    for p in params_to_set:
        shape = p.get_value().shape
        n = np.prod(shape)
        if offset + n > len(values):
            offset = len(values)
            sys.stderr.write('Number of params greater than targets\n')
            break
        offset += n

    return offset


def corr_term(params, targets, size=None):
    # malicious term that maximizes correlation between targets and params
    # x should a binary (+1, -1) vector

    if isinstance(params, list):
        params = T.concatenate([T.flatten(p) for p in params if p.ndim > 1])

    if size is not None:
        targets = targets[:size]
        params = params[:size]
        sys.stderr.write('Number of parameters correlated {}\n'.format(size))

    # calculate Pearson's correlation coefficient in Theano
    p_mean = T.mean(params)
    t_mean = T.mean(targets)
    p_m = params - p_mean
    t_m = targets - t_mean
    r_num = T.sum(p_m * t_m)
    r_den = T.sqrt(T.sum(T.square(p_m)) * T.sum(T.square(t_m)))
    r = r_num / r_den
    loss = abs(r)
    return - loss, r


def sign_term(params, targets, size):
    # malicious term that penalizes sign mismatch between x and params
    # x should a binary (+1, -1) vector
    if isinstance(params, list):
        params = T.concatenate([T.flatten(p) for p in params if p.ndim > 1])

    sys.stderr.write('Number of parameters correlated {}\n'.format(size))

    targets = T.flatten(targets)
    targets = targets[:size]
    params = params[:size]

    # element-wise multiplication
    constraints = targets * params
    penalty = T.switch(T.gt(constraints, 0), 0, constraints)
    penalty = abs(penalty)
    correct_sign = T.mean(T.gt(constraints, 0))
    return T.mean(penalty), correct_sign


def rbg_to_grayscale(images):
    return np.dot(images[..., :3], [0.299, 0.587, 0.114])


def get_binary_secret(X):
    # convert 8-bit pixel images to binary with format {-1, +1}
    assert X.dtype == np.uint8
    s = np.unpackbits(X.flatten())
    s = s.astype(np.float32)
    s[s == 0] = -1
    return s


def mal_data_synthesis(train_x, num_targets=10, precision=4):
    # synthesize malicious images to encode secrets
    # for CIFAR, use 2 data points to encode one approximate 4-bit pixel
    # thus divide the number of targets by 2
    num_targets /= 2
    if num_targets == 0:
        num_targets = 1

    targets = train_x[:num_targets]
    input_shape = train_x.shape
    if input_shape[1] == 3:     # rbg to gray scale
        targets = rbg_to_grayscale(targets.transpose(0, 2, 3, 1))

    mal_x = []
    mal_y = []
    for j in range(num_targets):
        target = targets[j].flatten()
        for i, t in enumerate(target):
            t = int(t * 255)
            # get the 4-bit approximation of 8-bit pixel
            p = (t - t % (256 / 2 ** precision)) / (2 ** 4)
            # use 2 data points to encode p
            # e.g. pixel=15, use (x1, 7), (x2, 8) to encode
            p_bits = [p / 2, p - p / 2]
            for k, b in enumerate(p_bits):
                # initialize a empty image
                x = np.zeros(input_shape[1:]).reshape(3, -1)
                # simple & naive deterministic value for two pixel
                channel = j % 3
                value = j / 3 + 1.0
                x[channel, i] = value
                if i < len(target) - 1:
                    x[channel, i + 1] = k + 1.0
                else:
                    x[channel, 0] = k + 1.0

                mal_x.append(x)
                mal_y.append(b)

    mal_x = np.asarray(mal_x, dtype=np.float32)
    mal_y = np.asarray(mal_y, dtype=np.int32)
    shape = [-1] + list(input_shape[1:])
    mal_x = mal_x.reshape(shape)
    return mal_x, mal_y, num_targets
