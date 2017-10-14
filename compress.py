from Crypto.Cipher import AES
from PIL import Image
from cStringIO import StringIO as BytesIO
import numpy as np
import gzip


ENC = AES.new('12345678' * 2, AES.MODE_CBC, 'This is an IV456')


def compress_image(raw_data, n=5e5, bits=16):
    # compress the raw data and encrypt it
    # return the float values whose LSBs are the cipher-text

    # limit in bits
    limit = int(n * bits)
    buff = BytesIO()
    # compression
    for x in raw_data:
        img = Image.fromarray(x)
        img.save(buff, "png")
    
    print "encoded size: ", len(raw_data.flatten()) * 8 / 1e6
    comp_buff = BytesIO()
    with gzip.GzipFile(fileobj=comp_buff, mode="w") as f:
        f.write(buff.getvalue())
    plain_text = comp_buff.getvalue()
    encoded_text = ''
    n = len(plain_text)
    for i in range(0, n, 16):
        start = i
        end = i + 16
        if end < n:
            cipher_text = ENC.encrypt(plain_text[start:end])
        else:
            pad = 16 - (n - start)
            cipher_text = ENC.encrypt(plain_text[start:] + '0' * pad)
        encoded_text += cipher_text

    data = map(ord, encoded_text)
    data = np.asarray(data, dtype=np.uint8)
    data = np.unpackbits(data)

    assert limit >= len(data)
    pads = limit - len(data)
    data = np.concatenate([data, np.zeros(pads)])
    data = data.reshape(-1, bits)
    pads = np.zeros([data.shape[0], 32 - bits])
    data = np.hstack([pads, data]).reshape(-1, 8)
    data = np.packbits(data.astype(np.uint8), -1)
    data = data.reshape(-1, 4)
    data = data[np.arange(len(data))[:, None], [3, 2, 1, 0]]
    data = data.view(np.float32).flatten()
    return data


if __name__ == '__main__':
    pass
