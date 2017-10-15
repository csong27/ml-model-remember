import theano
import theano.tensor as T
import numpy as np
import scalar_mask as scal
from theano.printing import pprint
from theano.tensor.basic import elemwise
from theano import printing
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams

rng = MRG_RandomStreams()


def _scal_elemwise_with_nfunc(nfunc, nin, nout):
    def construct(symbol):
        symbolname = symbol.__name__
        n = "Elemwise{%s,%s}" % (symbolname, "no_inplace")

        scalar_op = getattr(scal, symbolname)
        rval = elemwise.Elemwise(scalar_op, name=n, nfunc_spec=(nfunc and (nfunc, nin, nout)))

        if getattr(symbol, '__doc__', False):
            rval.__doc__ = symbol.__doc__ + '\n' + rval.__doc__

        # for the meaning of this see the ./epydoc script
        # it makes epydoc display rval as if it were a function, not an object
        rval.__epydoc_asRoutine = symbol
        rval.__module__ = 'tensor'

        pprint.assign(rval, printing.FunctionPrinter(symbolname))
        return rval

    return construct


_scale_elemwise = _scal_elemwise_with_nfunc(None, None, None)


@_scale_elemwise
def set_lsbs(x, y, b):
    ''' set_lsbs '''


def get_variable_shape(p):
    return p.get_value().shape


def share_variable(value, broadcastable):
    return theano.shared(value=value.astype(theano.config.floatX), broadcastable=broadcastable)


def convert_bits_to_params(bits, params):
    # reshape the vector of LSBs to the shape of model parameters
    params_shape = [get_variable_shape(p) for p in params]
    num_params = [int(np.prod(s)) for s in params_shape]
    cumsum_params = np.cumsum(num_params)
    bits = bits[:sum(num_params)]
    bits_params = [share_variable(bits[0: cumsum_params[0]].reshape(params_shape[0]),
                                  broadcastable=params[0].broadcastable)]
    for k in range(len(cumsum_params) - 1):
        start_idx = cumsum_params[k]
        end_idx = cumsum_params[k+1]
        bits_params.append(share_variable(bits[start_idx: end_idx].reshape(params_shape[k + 1]),
                                          broadcastable=params[k + 1].broadcastable))
    return bits_params


def mask_param_lsb(params, targets, bits=5):
    # set the LSBs of params with LSBs of targets
    updates = OrderedDict()
    for i, param in enumerate(params):
        target_param = theano.shared(value=targets[i].get_value())
        update_param = set_lsbs(param, target_param, bits)
        updates[param] = update_param

    for param, update in updates.items():
        if param.broadcastable != update.broadcastable:
            updates[param] = T.patternbroadcast(update, param.broadcastable)

    return theano.function([], updates=updates)
