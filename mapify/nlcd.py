import os
import base64
from functools import lru_cache

import requests
import numpy as np


aux_url = os.environ.get('AUX_URL', None)


def getchips(x, y, acquired, ubid, resource=aux_url):
    """
    Make a request to the HTTP API for some chip data.
    """
    chip_url = f'{resource}/chips'
    resp = requests.get(chip_url, params={'x': x,
                                          'y': y,
                                          'acquired': acquired,
                                          'ubid': ubid})
    if not resp.ok:
        resp.raise_for_status()

    return resp.json()


@lru_cache()
def getregistry(resource=aux_url):
    """
    Retrieve the spec registry from the API.
    """
    reg_url = f'{resource}/registry'
    return requests.get(reg_url).json()


@lru_cache()
def getspec(ubid):
    """
    Retrieve the appropriate spec information for the corresponding ubid.
    """
    registry = getregistry()
    return next(filter(lambda x: x['ubid'] == ubid, registry), None)


def tonumpy(chip):
    """
    Convert the data response to a numpy array.
    """
    spec = getspec(chip['ubid'])
    data = base64.b64decode(chip['data'])

    chip['data'] = np.frombuffer(data, spec['data_type'].lower()).reshape(*spec['data_shape'])

    return chip


def getnlcd(chip_x, chip_y, resource=aux_url):
    """
    Retrieve the 2001 NLCD layer
    """
    data = [tonumpy(c) for c in getchips(chip_x,
                                         chip_y,
                                         '1999-01-01/2002-01-01',
                                         'AUX_NLCD',
                                         resource)][0]

    if not data:
        data = np.zeros(shape=(100, 100))

    return data