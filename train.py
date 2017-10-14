import os
import sys
import time
import argparse
import pprint

import lasagne
import theano
import numpy as np
import theano.tensor as T

from net import build_resnet
from attack import rbg_to_grayscale, get_binary_secret, sign_term, corr_term, mal_data_synthesis, set_params_init
from load_cifar import load_cifar


CAP = 'cap'  # Capacity abuse attack
COR = 'cor'  # Correlation value encoding attack
SGN = 'sgn'  # Sign encoding attack
LSB = 'lsb'  # LSB encoding attack
NO = 'no'  # No attack


MODEL_DIR = './models/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)


def reshape_data(X_train, y_train, X_test):
    # reshape train and subtract mean
    pixel_mean = np.mean(X_train, axis=0)
    X_train -= pixel_mean
    X_test -= pixel_mean
    X_train_flip = X_train[:, :, :, ::-1]
    X_train = np.concatenate((X_train, X_train_flip), axis=0)
    y_train = np.concatenate((y_train, y_train), axis=0)
    return X_train, y_train, X_test


def iterate_minibatches(inputs, targets, batchsize, shuffle=False, augment=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if augment:
            # as in paper :
            # pad feature arrays with 4 pixels on each side
            # and do random cropping of 32x32
            padded = np.pad(inputs[excerpt], ((0, 0), (0, 0), (4, 4), (4, 4)), mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0, high=8, size=(batchsize, 2))
            for r in range(batchsize):
                random_cropped[r, :, :, :] = padded[r, :, crops[r, 0]:(crops[r, 0] + 32),
                                             crops[r, 1]:(crops[r, 1] + 32)]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        yield inp_exc, targets[excerpt]


def main(num_epochs=500, lr=0.1, attack=CAP, res_n=5, corr_ratio=0.0, mal_p=0.1):
    # training script modified from
    # https://github.com/Lasagne/Recipes/blob/master/papers/deep_residual_learning/Deep_Residual_Learning_CIFAR-10.py

    pprint.pprint(locals(), stream=sys.stderr)
    # Load the dataset
    sys.stderr.write("Loading data...\n")
    X_train, y_train, X_test, y_test = load_cifar(10)

    X_train = np.dstack((X_train[:, :1024], X_train[:, 1024:2048], X_train[:, 2048:]))
    X_train = X_train.reshape((-1, 32, 32, 3)).transpose(0, 3, 1, 2)
    X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048], X_test[:, 2048:]))
    X_test = X_test.reshape((-1, 32, 32, 3)).transpose(0, 3, 1, 2)

    mal_n = int(mal_p * len(X_train) * 2)
    n_out = len(np.unique(y_train))

    if attack in {SGN, COR}:
        # get the gray-scaled data to be encoded
        raw_data = X_train if X_train.dtype == np.uint8 else X_train * 255
        if raw_data.shape[-1] != 3:
            raw_data = raw_data.transpose(0, 2, 3, 1)
        raw_data = rbg_to_grayscale(raw_data).astype(np.uint8)
        sys.stderr.write('Raw data shape {}\n'.format(raw_data.shape))
        hidden_data_dim = np.prod(raw_data.shape[1:])
    elif attack == CAP:
        hidden_data_dim = int(np.prod(X_train.shape[2:]))
        mal_n /= hidden_data_dim
        if mal_n == 0:
            mal_n = 1
        X_mal, y_mal, mal_n = mal_data_synthesis(X_train, num_targets=mal_n)
        sys.stderr.write('Number of encoded image: {}\n'.format(mal_n))
        sys.stderr.write('Number of synthesized data: {}\n'.format(len(X_mal)))

    input_shape = (None, 3, X_train.shape[2], X_train.shape[3])

    X_train, y_train, X_test = reshape_data(X_train, y_train, X_test)
    X_val, y_val = X_test, y_test

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    if attack == CAP:
        X_train_mal = np.vstack([X_train, X_mal])
        y_train_mal = np.concatenate([y_train, y_mal])

    n = len(X_train)
    sys.stderr.write("Number of training data, output: {}, {}...\n".format(n, n_out))

    # Create neural network model (depending on first command line parameter)
    sys.stderr.write("Building model and compiling functions...\n")
    network = build_resnet(input_var=input_var, classes=n_out, input_shape=input_shape, n=res_n)

    params = lasagne.layers.get_all_params(network, trainable=True)
    total_params = lasagne.layers.count_params(network, trainable=True)
    sys.stderr.write("Number of parameters in model: %d\n" % total_params)

    if attack == COR:
        n_hidden_data = total_params / int(hidden_data_dim)
        sys.stderr.write("Number of data correlated: %d\n" % n_hidden_data)
        corr_targets = raw_data[:n_hidden_data].flatten()
        corr_targets = theano.shared(corr_targets)
        offset = set_params_init(params, corr_targets)
        corr_loss, r = corr_term(params, corr_targets, size=offset)
    elif attack == SGN:
        n_hidden_data = total_params / int(hidden_data_dim) / 8
        sys.stderr.write("Number of data sign-encoded: %d\n" % n_hidden_data)
        corr_targets = get_binary_secret(raw_data[:n_hidden_data])
        corr_targets = theano.shared(corr_targets)
        offset = set_params_init(params, corr_targets)
        corr_loss, r = sign_term(params, corr_targets, size=offset)
    else:
        r = T.constant(0., dtype=np.float32)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.
    all_layers = lasagne.layers.get_all_layers(network)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
    loss += l2_penalty
    # add malicious term to loss function
    if attack in {SGN, COR}:
        corr_loss *= corr_ratio
        loss += corr_loss

    # save init
    sh_lr = theano.shared(lasagne.utils.floatX(lr))

    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=sh_lr)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()

    # As a bonus, also create an expression for the classification accuracy:
    if target_var.ndim == 1:
        test_acc = T.sum(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
    else:
        test_acc = T.sum(T.eq(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1)),
                         dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], [loss, r], updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    sys.stderr.write("Starting training...\n")
    # We iterate over epochs:
    for epoch in range(num_epochs):

        # shuffle training data
        train_indices = np.arange(n)
        np.random.shuffle(train_indices)
        X_train = X_train[train_indices, :, :, :]
        y_train = y_train[train_indices]

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        train_r = 0
        for batch in iterate_minibatches(X_train, y_train, 128, shuffle=True, augment=True):
            inputs, targets = batch
            err, r = train_fn(inputs, targets)
            train_r += r
            train_err += err
            train_batches += 1
        if attack == CAP:
            # And a full pass over the malicious data
            for batch in iterate_minibatches(X_train_mal, y_train_mal, 128, shuffle=True, augment=False):
                inputs, targets = batch
                err, r = train_fn(inputs, targets)
                train_r += r
                train_err += err
                train_batches += 1

        if attack == CAP:
            mal_err = 0
            mal_acc = 0
            mal_batches = 0
            for batch in iterate_minibatches(X_mal, y_mal, 500, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                mal_err += err
                mal_acc += acc
                mal_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        if (epoch + 1) == 41 or (epoch + 1) == 61:
            new_lr = sh_lr.get_value() * 0.1
            sys.stderr.write("New LR:" + str(new_lr) + "\n")
            sh_lr.set_value(lasagne.utils.floatX(new_lr))

        # Then we sys.stderr.write the results for this epoch:
        sys.stderr.write("Epoch {} of {} took {:.3f}s\n".format(epoch + 1, num_epochs, time.time() - start_time))
        sys.stderr.write("  training loss:\t\t{:.6f}\n".format(train_err / train_batches))
        if attack == CAP:
            sys.stderr.write("  malicious loss:\t\t{:.6f}\n".format(mal_err / mal_batches))
            sys.stderr.write("  malicious accuracy:\t\t{:.2f} %\n".format(
                mal_acc / mal_batches / 500 * 100))
        if attack in {SGN, COR}:
            sys.stderr.write("  training r:\t\t{:.6f}\n".format(train_r / train_batches))

        sys.stderr.write("  validation loss:\t\t{:.6f}\n".format(val_err / val_batches))
        sys.stderr.write("  validation accuracy:\t\t{:.2f} %\n".format(val_acc / val_batches / 500 * 100))

    # After training, we compute and sys.stderr.write the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1

    sys.stderr.write("Final results:\n")
    sys.stderr.write("  test loss:\t\t\t{:.6f}\n".format(test_err / test_batches))
    sys.stderr.write("  test accuracy:\t\t{:.2f} %\n".format(test_acc / test_batches / 500 * 100))

    # save final model
    model_path = MODEL_DIR + 'cifar_{}_res{}_'.format(attack, res_n)
    if attack == CAP:
        model_path += '{}_'.format(mal_p)
    if attack in {COR, SGN}:
        model_path += '{}_'.format(corr_ratio)
    np.savez(model_path + 'model.npz', *lasagne.layers.get_all_param_values(network))

    return test_acc / test_batches / 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.1)    # learning rate
    parser.add_argument('--epoch', type=int, default=100)   # number of epochs for training
    parser.add_argument('--model', type=int, default=5)     # number of blocks in resnet
    parser.add_argument('--attack', type=str, default=CAP)  # attack type
    parser.add_argument('--corr', type=float, default=0.)   # malicious term ratio
    parser.add_argument('--mal_p', type=float, default=0.1) # proportion of malicious data to training data
    args = parser.parse_args()
    main(num_epochs=args.epoch, lr=args.lr, corr_ratio=args.corr, mal_p=args.mal_p, attack=args.attack,
         res_n=args.model)
