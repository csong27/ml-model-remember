import theano
from textwrap import dedent
from theano.scalar.basic import ScalarOp, discrete_types, upcast_out_no_complex


class SetLSBs(ScalarOp):
    # Theano ops for set LSBs
    nin = 3

    def impl(self, x, y, b):
        pass

    def grad(self, inputs, gout):
        (x, y, b) = inputs
        (gz,) = gout
        rval_x = x.zeros_like()
        rval_y = y.zeros_like()
        rval_b = b.zeros_like()

        if rval_x.type.dtype in discrete_types:
            rval_x = rval_x.astype(theano.config.floatX)

        if rval_y.type.dtype in discrete_types:
            rval_y = rval_y.astype(theano.config.floatX)

        if rval_b.type.dtype in discrete_types:
            rval_b = rval_y.astype(theano.config.floatX)

        return [rval_x, rval_y, rval_b]

    def c_code(self, node, name, inputs, outputs, sub):
        # set the lsb of x to lsb of y
        (x, y, b) = inputs
        (z,) = outputs

        typ = node.outputs[0].type.dtype
        if typ not in ['float32']:
            Exception("The output should be float32")

        return dedent("""
        typedef union {
          int i;
          float f;
        } u;
        u u1;
        u1.f = %(x)s;
        u u2;
        u2.f = %(y)s;
        int bits = static_cast<int>(%(b)s);
        bits = 0xffffffff >> (32 - bits);
        u1.i = (u1.i & ~bits) | (u2.i & bits);
        %(z)s = u1.f;
        """ % locals())

set_lsbs = SetLSBs(upcast_out_no_complex)


if __name__ == '__main__':
    pass
