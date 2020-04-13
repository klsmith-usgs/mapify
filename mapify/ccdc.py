"""
Reading CCDC raw results, typically JSON or pickle formats for test data.
"""
import os
import json
import pickle
import logging
import gzip
from collections import defaultdict
import datetime as dt
from typing import Tuple, List, Sequence

import numpy as np

from mapify.products import BandModel, CCDCModel
from mapify.spatial import buildaff, transform_geo
from mapify.app import band_names as _band_names
from mapify.app import lc_map as _lc_map


log = logging.getLogger()
grass = _lc_map['grass']
forest = _lc_map['tree']

class_vals = tuple(range(10))


def jsonpaths(root: str) -> list:
    """
    Create a list of file paths to files that end in .json in the given directory.

    Args:
        root: directory path

    Returns:
        sorted list of JSON file paths
    """
    return [os.path.join(root, f)
            for f in sorted(os.listdir(root))
            if f.endswith('.json.gz')]


def picklepaths(root: str) -> list:
    """
    Create a list of file paths to files that end in .p in the given directory.

    Args:
        root: directory path

    Returns:
        sorted list of pickle file paths
    """
    return [os.path.join(root, f)
            for f in sorted(os.listdir(root))
            if f[-2:] == '.p']


def pathcoords(path: str) -> Tuple[int, int]:
    """
    Pull the Chip X and Chip Y coords from the file path.

    Args:
        path: file path

    Returns:
        chip upper left x/y based on the file name
    """
    parts = os.path.split(path)[-1].split('_')
    return int(parts[1]), int(parts[2][:-8])


def loadjfile(path: str) -> list:
    """
    Load a JSON formatted file into a dictionary.

    Args:
        path: file path

    Returns:
        dictionary representation of the JSON
    """
    with gzip.open(path, 'rt') as gf:
        return json.load(gf)


def loadjstr(string: str) -> dict:
    """
    Load a JSON formatted string into a dictionary.

    Args:
        string: JSON formatted string

    Returns:
        dictionary representation of the JSON
    """
    return json.loads(string)


def loadpfile(path: str) -> list:
    """
    Loads whatever object is contained in the pickle file.

    Args:
        path: file path

    Returns:
        some object
    """
    return pickle.load(open(path, 'rb'))


def empty(band_names: Sequence=_band_names) -> CCDCModel:
    """
    Return an empty CCDC model

    Args:
        band_names: bands to build

    Returns:
        CCDCModel
    """
    bands = [BandModel(name=b, magnitude=0.0, rmse=0.0, intercept=0.0, coefficients=tuple([0.0] * 6))
             for b in band_names]

    return CCDCModel(start_day=0,
                     end_day=0,
                     break_day=0,
                     obs_count=0,
                     change_prob=0.0,
                     curve_qa=0,
                     bands=tuple(bands),
                     class_split=0,
                     class_probs1=tuple([0] * 9),
                     class_probs2=tuple([0] * 9),
                     class_vals=tuple(range(9)))


def buildband(chgmodel: dict, name: str) -> BandModel:
    """
    Build an individual band namedtuple from a change model

    Args:
        chgmodel: dictionary repesentation of a change model
        name: which band to build
    
    Returns:
        individual band model
    """
    return BandModel(name=name,
                     magnitude=chgmodel[name]['magnitude'],
                     rmse=chgmodel[name]['rmse'],
                     coefficients=tuple(chgmodel[name]['coefficients']),
                     intercept=chgmodel[name]['intercept'])


def buildccdc(chgmodel: dict, preds: List[dict] = None, band_names: Sequence=_band_names) -> CCDCModel:
    """
    Build a complete CCDC model

    Args:
        chgmodel: dictionary representation of a change model
        preds: predictions for the associated change model
        band_names: band names present in the change model

    Returns:
        a unified CCDC model
    """
    bands = [buildband(chgmodel, b) for b in band_names]

    if preds is None:
        return CCDCModel(start_day=chgmodel['start_day'],
                         end_day=chgmodel['end_day'],
                         break_day=chgmodel['break_day'],
                         obs_count=chgmodel['observation_count'],
                         change_prob=chgmodel['change_probability'],
                         curve_qa=chgmodel['curve_qa'],
                         bands=tuple(bands),
                         class_split=0,
                         class_probs1=np.full((8,), -1),
                         class_probs2=np.full((8,), -1),
                         class_vals=class_vals)
    elif vegincrease(chgmodel, preds):
        return CCDCModel(start_day=chgmodel['start_day'],
                         end_day=chgmodel['end_day'],
                         break_day=chgmodel['break_day'],
                         obs_count=chgmodel['observation_count'],
                         change_prob=chgmodel['change_probability'],
                         curve_qa=chgmodel['curve_qa'],
                         bands=tuple(bands),
                         class_split=splitdate(preds, 'tree'),
                         class_probs1=growthgrass(),
                         class_probs2=growthforest(),
                         class_vals=class_vals)
    elif vegdecrease(chgmodel, preds):
        return CCDCModel(start_day=chgmodel['start_day'],
                         end_day=chgmodel['end_day'],
                         break_day=chgmodel['break_day'],
                         obs_count=chgmodel['observation_count'],
                         change_prob=chgmodel['change_probability'],
                         curve_qa=chgmodel['curve_qa'],
                         bands=tuple(bands),
                         class_split=splitdate(preds, 'grass'),
                         class_probs1=declineforest(),
                         class_probs2=declinegrass(),
                         class_vals=class_vals)
    else:
        return CCDCModel(start_day=chgmodel['start_day'],
                         end_day=chgmodel['end_day'],
                         break_day=chgmodel['break_day'],
                         obs_count=chgmodel['observation_count'],
                         change_prob=chgmodel['change_probability'],
                         curve_qa=chgmodel['curve_qa'],
                         bands=tuple(bands),
                         class_split=0,
                         class_probs1=meanpred(preds),
                         class_probs2=np.full((8,), -1),
                         class_vals=class_vals)


def unify(ccd: dict, classified: dict, loc: Tuple[float, float]) -> List[CCDCModel]:
    """
    Combine the two disparate models for a given pixel and make a list of unified models.

    Args:
        ccd: pyccd results for a pixel
        classified: test classification results for the pixel from a pickle file
    
    Returns:
        unified CCDC models
    """
    models = []

    if ccd is None:
        return models
    # log.debug(len(classified))
    # log.debug(len(ccd['change_models']))
    for segment in ccd['change_models']:
        preds = classified.get((*loc, segment['start_day']), None)
        models.append(buildccdc(segment, preds))

    return models


def noclass(ccd: dict) -> list:
    return [buildccdc(model) for model in ccd['change_models']]


def spatialccdc(ccd_data: list, class_data: list) -> np.ndarray:
    """
    Provide a unified CCDC model in a pseudo-spatial chip, as represented by a
    flattened list of lists.

    Args:
        ccd_data: initial JSON deserialization of change results for a chip
        class_data: deserialization of classification results for a chip

    Returns:
        ndarray of lists(of CCDC namedtuples)
    """
    chip_x, chip_y = (ccd_data[0]['chip_x'], ccd_data[0]['chip_y'])

    ccd_data = groupchg(ccd_data)
    class_data = grouppreds(class_data)

    outdata = np.full(fill_value=None, shape=(100, 100), dtype=object)
    aff = buildaff(chip_x, chip_y, 30)

    for loc in ccd_data:
        row, col = transform_geo(loc[2], loc[3], aff)
        outdata[row][col] = unify(ccd_data[loc], class_data, loc)

    return outdata.flatten()


def splitdate(preds, covertype, lcmap=_lc_map):
    """
    Return the date of when the cover first showed up.
    """
    mprobs = np.array([np.argmax(p['prob']) for p in preds])

    # Look for the first instance of the target class showing up.
    spl_idx = np.flatnonzero(mprobs == lcmap[covertype])[0]

    return toord_iso(preds[spl_idx]['pday'])


def toord_iso(iso_date: str) -> int:
    """
    Convert an ordinal date to ISO-8601
    """
    return dt.date.fromisoformat(iso_date).toordinal()


def nbrdiff(chgmodel):
    """
    Calculate how much NBR shifts from beginning to the end of the segment.
    """
    sord = chgmodel['start_day']
    eord = chgmodel['end_day']

    nir_st = chgmodel['nir']['coefficients'][0] * sord + chgmodel['nir']['intercept']
    nir_en = chgmodel['nir']['coefficients'][0] * eord + chgmodel['nir']['intercept']

    swir_st = chgmodel['swir1']['coefficients'][0] * sord + chgmodel['swir1']['intercept']
    swir_en = chgmodel['swir1']['coefficients'][0] * eord + chgmodel['swir1']['intercept']

    nbr_st = (nir_st - swir_st) / (nir_st + swir_st)
    nbr_en = (nir_en - swir_en) / (nir_en + swir_en)

    return nbr_en - nbr_st


def vegincrease(chgmodel, preds, lcmap=_lc_map):
    """
    Given a list of date sorted predictions that for a segment, determine if it is a
    vegetation increase segment.
    """
    diff = nbrdiff(chgmodel)
    return diff > 0.05 and np.argmax(preds[0]['prob']) == lcmap['grass'] and np.argmax(preds[-1]['prob']) == lcmap[
        'tree']


def vegdecrease(chgmodel, preds, lcmap=_lc_map):
    """
    Given a list of date sorted predictions that for a segment, determine if it is a
    vegetation decrease segment.
    """
    diff = nbrdiff(chgmodel)
    return diff < -0.05 and np.argmax(preds[0]['prob']) == lcmap['tree'] and np.argmax(preds[-1]['prob']) == lcmap[
        'grass']


def meanpred(preds):
    """
    Given a list of predictions, returns the per-class mean.
    This makes no assumption that they actually belong together, but they should.
    """
    return np.mean(np.array([p['prob'] for p in preds], dtype=np.float32), axis=0)


def grouppreds(preds: List[dict]) -> dict:
    """
    Group predictions based on location and start date, then sorted by the prediction date.
    """
    out = defaultdict(list)

    if preds is None:
        return out

    for pred in preds:
        key = (pred['cx'], pred['cy'], pred['px'], pred['py'], pred['sday'])
        out[key].append(pred)

    return {k: sorted(v, key=lambda x: x['pday']) for k, v in out.items()}


def groupchg(segments: List[dict]) -> dict:
    """
    Group segments based on (px, py) and then sort them based on sday.
    """

    return {(seg['chip_x'], seg['chip_y'], seg['x'], seg['y']): seg['results'] for seg in segments}


def growthforest():
    ret = np.zeros(shape=(9,), dtype=np.float)
    ret[forest] = 1.11
    ret[grass] = 1.10
    return ret


def growthgrass():
    ret = np.zeros(shape=(9,), dtype=np.float)
    ret[forest] = 1.10
    ret[grass] = 1.11
    return ret


def declineforest():
    ret = np.zeros(shape=(9,), dtype=np.float)
    ret[forest] = 1.21
    ret[grass] = 1.20
    return ret


def declinegrass():
    ret = np.zeros(shape=(9,), dtype=np.float)
    ret[forest] = 1.20
    ret[grass] = 1.21
    return ret
