from PIL import ImageOps, Image
import lasagne
import numpy as np
import theano.tensor as T
import theano
import argparse
import os
import cv2

from net import build_resnet
from attack import mal_data_synthesis
from mask_param import mask_param_lsb, convert_bits_to_params
from compress import compress_image
from train import rbg_to_grayscale, reshape_data, CAP, LSB, SGN, COR, NO, MODEL_DIR
from load_cifar import load_cifar


IMG_DIR = './imgs/'
if not os.path.exists(IMG_DIR):
    os.mkdir(IMG_DIR)


def iterate_minibatches(inputs, targets, batch_size):
    assert len(inputs) == len(targets)
    start_idx = None
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

    if start_idx is not None and start_idx + batch_size < len(inputs):
        excerpt = slice(start_idx + batch_size, len(inputs))
        yield inputs[excerpt], targets[excerpt]


def test_cap_reconstruction(res_n=5, p=None):
    # evaluate capacity abuse attack

    param_values = load_params(CAP, res_n, hp=p)
    X_train, y_train, X_test, y_test = load_cifar(10)

    X_train = np.dstack((X_train[:, :1024], X_train[:, 1024:2048], X_train[:, 2048:]))
    X_train = X_train.reshape((-1, 32, 32, 3)).transpose(0, 3, 1, 2)

    input_shape = (None, 3, X_train.shape[2], X_train.shape[3])
    n_out = len(np.unique(y_train))
    input_var = T.tensor4('x')

    network = build_resnet(input_var=input_var, classes=n_out, input_shape=input_shape, n=res_n)

    mal_n = int(p * len(X_train) * 2)
    lasagne.layers.set_all_param_values(network, param_values)

    hidden_data_dim = np.prod(X_train.shape[2:])
    mal_n /= hidden_data_dim

    if mal_n == 0:
        mal_n = 1

    # recreate malicious feature vector
    X_mal, y_mal, mal_n = mal_data_synthesis(X_train, num_targets=mal_n)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_prediction = T.argmax(test_prediction, axis=1)

    query_fn = theano.function([input_var], test_prediction)
    pixels = []
    for batch in iterate_minibatches(X_mal, y_mal, 500):
        inputs, _ = batch
        pred = query_fn(inputs)
        pixels.append(pred)

    # now pixels are predictions from the model, which should be
    # close to the encoded bits
    pixels = np.concatenate(pixels)
    pixels = pixels.reshape(-1, 2).sum(1)   # we used two predictions to encode one pixel
    pixels = pixels.reshape(mal_n, X_train.shape[2], X_train.shape[3])

    raw_data = X_train if X_train.dtype == np.uint8 else X_train * 255
    if raw_data.shape[-1] != 3:
        raw_data = raw_data.transpose(0, 2, 3, 1)
    raw_data = rbg_to_grayscale(raw_data).astype(np.uint8)
    targets = raw_data[:mal_n]

    img_dir = IMG_DIR + 'cap_cifar_{}/'.format(p)
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    err, sim = 0., 0.
    for i, img in enumerate(pixels):
        img_name = img_dir + 'cifar_res{}_{}.png'.format(res_n, i)
        img *= 2 ** 4
        cv2.imwrite(img_name, img.astype(np.uint8))
        e, s = image_metrics(img, targets[i].astype(np.uint8))
        err += e
        sim += s

    print err / mal_n, sim / mal_n


def image_metrics(img1, img2):
    # return mean abs error and cosine distance
    img1 = img1.astype(float).flatten()
    img2 = img2.astype(float).flatten()
    return np.mean(np.abs(img1 - img2)),  np.abs(np.dot(img1, img2) / (np.linalg.norm(img1) * np.linalg.norm(img2)))


def normalize(x):
    x_shape = x.shape
    x = x.flatten()
    x_min = np.min(x)
    x_max = np.max(x)
    x = (x - x_min) / (x_max - x_min)
    return x.reshape(x_shape)


def test_cor_reconstruction(res_n=5, cr=None):
    # evaluate correlation encoding attack

    X_train, y_train, X_test, y_test = load_cifar(10)

    X_train = np.dstack((X_train[:, :1024], X_train[:, 1024:2048], X_train[:, 2048:]))
    X_train = X_train.reshape((-1, 32, 32, 3)).transpose(0, 3, 1, 2)

    hidden_data_dim = np.prod(X_train.shape[2:])

    # read parameter values
    param_values = load_params(COR, res_n=res_n, hp=cr)
    params = np.concatenate([p.flatten() for p in param_values if p.ndim > 1])
    total_params = len(params)
    n_hidden_data = total_params / int(hidden_data_dim)
    if len(params) < n_hidden_data * hidden_data_dim:
        n_hidden_data -= 1
    cor_params = params[: n_hidden_data * hidden_data_dim].reshape(n_hidden_data, X_train.shape[2], X_train.shape[3])
    raw_data = X_train if X_train.dtype == np.uint8 else X_train * 255
    if raw_data.shape[-1] != 3:
        raw_data = raw_data.transpose(0, 2, 3, 1)
    raw_data = rbg_to_grayscale(raw_data).astype(np.uint8)
    targets = raw_data[:n_hidden_data]

    img_dir = IMG_DIR + 'cor_cifar_{}/'.format(cr)
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    err, sim = 0., 0.
    for i, img in enumerate(cor_params):
        img_name = img_dir + 'cifar_res{}_{}.png'.format(res_n, i)
        # transform correlated parameters back to input space
        img = normalize(img)
        img = (img * 255).astype(np.uint8)
        cv2.imwrite(img_name, img)
        e1, s1 = image_metrics(img, targets[i].astype(np.uint8))

        # some times we get negatively correlated values, invert it
        img = np.asarray(ImageOps.invert(Image.fromarray(img)))
        e2, s2 = image_metrics(img, targets[i].astype(np.uint8))
        err += min([e1, e2])
        sim += max([s1, s2])

    print err / n_hidden_data, sim / n_hidden_data


def test_sgn_reconstruction(res_n=5, cr=None):
    # evaluate sign encoding attack

    X_train, y_train, X_test, y_test = load_cifar(10)

    X_train = np.dstack((X_train[:, :1024], X_train[:, 1024:2048], X_train[:, 2048:]))
    X_train = X_train.reshape((-1, 32, 32, 3)).transpose(0, 3, 1, 2)

    hidden_data_dim = np.prod(X_train.shape[2:])

    # read parameter values
    param_values = load_params(SGN, res_n=res_n, hp=cr)
    params = np.concatenate([p.flatten() for p in param_values if p.ndim > 1])
    total_params = len(params)
    print total_params
    n_hidden_data = total_params / int(hidden_data_dim) / 8
    print n_hidden_data

    # get the signs as bits
    bits = np.sign(params[: n_hidden_data * int(hidden_data_dim) * 8])
    bits[bits == -1] = 0
    bits = bits.astype(np.uint8)
    imgs = np.packbits(bits.reshape(-1, 8)).reshape(n_hidden_data, X_train.shape[2], X_train.shape[3])

    raw_data = X_train if X_train.dtype == np.uint8 else X_train * 255
    if raw_data.shape[-1] != 3:
        raw_data = raw_data.transpose(0, 2, 3, 1)
    raw_data = rbg_to_grayscale(raw_data).astype(np.uint8)
    targets = raw_data[:n_hidden_data]

    img_dir = IMG_DIR + 'sgn_cifar_{}/'.format(cr)
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    err, sim = 0., 0.
    for i, img in enumerate(imgs):
        img_name = img_dir + 'cifar_res{}_{}.png'.format(res_n, i)
        img = img.astype(np.uint8)
        cv2.imwrite(img_name, img)
        e, s = image_metrics(img, targets[i].astype(np.uint8))
        err += e
        sim += s

    print err / n_hidden_data, sim / n_hidden_data


def test_lsb_acc(res_n=5, bits=16, n_data=1000):
    param_values = load_params(NO, res_n)
    X_train, y_train, X_test, y_test = load_cifar(10)
    X_train = np.dstack((X_train[:, :1024], X_train[:, 1024:2048], X_train[:, 2048:]))
    X_train = X_train.reshape((-1, 32, 32, 3)).transpose(0, 3, 1, 2)
    X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048], X_test[:, 2048:]))
    X_test = X_test.reshape((-1, 32, 32, 3)).transpose(0, 3, 1, 2)

    input_shape = (None, 3, X_train.shape[2], X_train.shape[3])
    n_out = len(np.unique(y_train))
    input_var = T.tensor4('x')
    target_var = T.ivector('targets')

    _, _, X_test = reshape_data(X_train, y_train, X_test)

    network = build_resnet(input_var=input_var, classes=n_out, input_shape=input_shape, n=res_n)
    lasagne.layers.set_all_param_values(network, param_values)

    if bits:
        raw_data = X_train if X_train.dtype == np.uint8 else X_train * 255
        if raw_data.shape[-1] != 3:
            raw_data = raw_data.transpose(0, 2, 3, 1)
        raw_data = rbg_to_grayscale(raw_data).astype(np.uint8)
        total_params = lasagne.layers.count_params(network)
        # get vector of values whose LSBs are compressed and encrypted data
        lsb_params = compress_image(raw_data[:n_data], total_params, bits)
        lsb_params = convert_bits_to_params(lsb_params, lasagne.layers.get_all_params(network))
        print('Writing lower {} bits of parameters...\n'.format(bits))
        mask_fn = mask_param_lsb(lasagne.layers.get_all_params(network), lsb_params, bits=bits)
    else:
        mask_fn = lambda: None

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_acc = T.sum(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
    val_fn = theano.function([input_var, target_var], test_acc)
    # After training, we compute and sys.stderr.write the test error:
    mask_fn()
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        acc = val_fn(inputs, targets)
        test_acc += acc
        test_batches += 1
    final_acc = test_acc / test_batches / 500 * 100
    print "LSB {} test accuracy:\t\t{:.2f} %\n".format(bits, final_acc)


def load_params(attack, res_n=5, hp=None):
    if hp is None:
        hp = ''
    else:
        hp = str(hp) + '_'
    path = MODEL_DIR + 'cifar_{}_res{}_{}model.npz'.format(attack, res_n, hp)
    with np.load(path) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    return param_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', type=str, default=CAP)  # attack type
    parser.add_argument('--bits', type=int, default=16)     # number of LSB set to secrets
    parser.add_argument('--n', type=int, default=1000)      # number of data points to be encoded in LSB
    parser.add_argument('--cr', type=float, default=1.0)    # malicious term ratio
    parser.add_argument('--p', type=float, default=1.0)     # proportion of malicious data to training data
    parser.add_argument('--model', type=int, default=5)     # number of blocks in resnet

    args = parser.parse_args()
    attack = args.attack
    if attack == CAP:
        test_cap_reconstruction(p=args.p, res_n=args.model)
    elif attack == COR:
        test_cor_reconstruction(cr=args.cr, res_n=args.model)
    elif attack == SGN:
        test_sgn_reconstruction(cr=args.cr, res_n=args.model)
    elif attack == LSB:
        test_lsb_acc(bits=args.bits, n_data=args.n, res_n=args.model)
    else:
        raise ValueError(attack)
