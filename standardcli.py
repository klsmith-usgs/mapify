"""
Temporary CLI to get things moving until more things get more thought out ...
"""

import os
import sys
import argparse
import datetime as dt
import multiprocessing as mp
from typing import Callable
import logging

import numpy as np
from osgeo import gdal

from mapify.ccdc import jsonpaths, picklepaths, spatialccdc, loadjfile, loadpfile, pathcoords
from mapify.products import prodmap, crosswalk, is_lc, lc_color
from mapify.spatial import readxy, determine_hv, create, transform_geo, buildaff, write, transform_rc, writep
from mapify.app import cu_tileaff as _cu_tileaff
from mapify.app import cu_wkt

_productmap = prodmap()


log = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s %(processName)s: %(message)s')

handler.setFormatter(formatter)

log.addHandler(handler)
log.setLevel(logging.DEBUG)


def validprods():
    prods = ['all']
    prods.extend(list(_productmap.keys()))
    return prods


def main(args: argparse.Namespace) -> None:
    args = yarg(args)
    if args.cpu == 1:
        single(args)
    elif args.cpu > 1:
        multi(args)
    else:
        raise ValueError


def yarg(args: argparse.Namespace) -> argparse.Namespace:
    args.prods = ','.join(list(_productmap.keys()))
    if args.cpu < 1:
        args.cpu = 1
    args.dates = ','.join([dt.date(year=yr, month=7, day=1).strftime('%Y-%m-%d')
                           for yr in range(1985, 2018)])
    args.trunc_dates = True

    return args


def makekwargs_prime(args: argparse.Namespace) -> dict:
    return {'fill_begin': True,
            'fill_end': True,
            'fill_samelc': True,
            'fill_difflc': True,
            'fill_nomodel': True,
            'xwalk': args.xwalk}


def single(args: argparse.Namespace) -> None:
    pass


def multi(args: argparse.Namespace) -> None:
    log.debug('Multi-processing starting')
    worker_cnt = args.cpu - 1
    in_q = enqueue(args.jdir, args.pdir, worker_cnt)
    log.debug('Size of input queue: %s', in_q.qsize())
    out_q = mp.Queue()

    log.debug('Starting workers')
    for _ in range(worker_cnt):
        mp.Process(target=worker,
                   args=(in_q, out_q, args),
                   name='Process-{}'.format(_)).start()

    multiout(out_q, args.outdir, worker_cnt,  args)


def enqueue(jdir: str, pdir: str, kill: int) -> mp.Queue:
    q = mp.Queue()
    for j, p in zip(jsonpaths(jdir), picklepaths(pdir)):
        q.put((j, p))

    for _ in range(kill):
        q.put((-1, -1))

    return q


def prodfunc(name: str) -> Callable:
    return _productmap[name][0]


def ordme(date: str) -> int:
    return dt.datetime.strptime(date, '%Y-%m-%d').toordinal()


def nlcdchip(path: str, chip_x: float, chip_y: float) -> np.ndarray:
    data = readxy(path, chip_x, chip_y, 100, 100)

    if data is None:
        data = np.zeros(shape=(100, 100))

    return data


def neednlcd(name: str, fill_nomodel: bool) -> bool:
    return name in ('LC_Primary', 'LC_Change') and fill_nomodel


def worker(inq: mp.Queue, outq: mp.Queue, args: argparse.Namespace):
    while True:
        jpath, ppath = inq.get()

        if jpath == -1:
            log.debug('Shutting down ...')
            outq.put((-1,) * 3)
            break

        log.debug('Received: %s and %s',
                  os.path.split(jpath)[-1],
                  os.path.split(ppath)[-1])

        jdata = loadjfile(jpath)
        pdata = loadpfile(ppath)
        ccdc = spatialccdc(jdata, pdata)
        chip_x, chip_y = pathcoords(jpath)

        prods = args.prods.split(',')
        ords = [ordme(d) for d in args.dates.split(',')]
        dates = args.dates.split(',')

        nlcd = nlcdchip(args.nlcdpath, chip_x, chip_y).flatten()
        if args.xwalk:
            nlcd = crosswalk(nlcd)

        out = {}
        for prod in prods:
            kwargs = makekwargs_prime(args)

            if prod not in out:
                out[prod] = {}

            for date, o_date in zip(dates, ords):
                if neednlcd(prod, True):
                    out[prod][date] = np.array([prodfunc(prod)(models, o_date, fill_nodataval=nlcd[p_idx], **kwargs)
                                                for p_idx, models in enumerate(ccdc)])
                else:
                    out[prod][date] = np.array([prodfunc(prod)(models, o_date, **kwargs) for models in ccdc])

        log.debug('Finished with chip')
        outq.put((chip_x, chip_y, out))


def filename(chip_x: float, chip_y: float, prod: str, date: str, trunc_date: bool) -> str:
    h, v = determine_hv(chip_x, chip_y, _cu_tileaff)

    if trunc_date:
        date = date.split('-')[0]

    return 'h{:02d}v{:02d}_{}_{}.tif'.format(h, v, prod, date.replace('-', ''))


# def writechip(ds: gdal.Dataset, chip_x: float, chip_y: float, data: np.ndarray) -> None:
#     h, v = determine_hv(chip_x, chip_y)
#     ulx, uly = transform_rc(v, h, _cu_tileaff)
#     aff = buildaff(ulx, uly, 30)
#     row_off, col_off = transform_geo(chip_x, chip_y, aff)
#
#     write(ds, data.reshape(100, 100), col_off, row_off)


def writechip(path: str, chip_x: float, chip_y: float, data: np.ndarray) -> None:
    h, v = determine_hv(chip_x, chip_y, _cu_tileaff)
    ulx, uly = transform_rc(v, h, _cu_tileaff)
    aff = buildaff(ulx, uly, 30)
    row_off, col_off = transform_geo(chip_x, chip_y, aff)

    writep(path, data.reshape(100, 100), col_off, row_off)


# def writesynthetic(ds: gdal.Dataset, chip_x: float, chip_y: float, data: np.ndarray) -> None:
#     h, v = determine_hv(chip_x, chip_y)
#     ulx, uly = transform_rc(v, h, _cu_tileaff)
#     aff = buildaff(ulx, uly, 30)
#     row_off, col_off = transform_geo(chip_x, chip_y, aff)
#
#     for idx, b in enumerate(data.reshape(-1, 100, 100)):
#         write(ds, b, col_off, row_off, band=idx + 1)


def writesynthetic(path: str, chip_x: float, chip_y: float, data: np.ndarray) -> None:
    h, v = determine_hv(chip_x, chip_y, _cu_tileaff)
    ulx, uly = transform_rc(v, h, _cu_tileaff)
    aff = buildaff(ulx, uly, 30)
    row_off, col_off = transform_geo(chip_x, chip_y, aff)

    for idx in range(7):
        writep(path, data[:, idx].reshape(100, 100), col_off, row_off, band=idx + 1)


def maketile(path: str, chip_x: float, chip_y: float, product: str) -> gdal.Dataset:
    h, v = determine_hv(chip_x, chip_y, _cu_tileaff)
    ulx, uly = transform_rc(v, h, _cu_tileaff)
    aff = buildaff(ulx, uly, 30)
    datatype = _productmap[product][1]

    if is_lc(product):
        ct = [lc_color()]
    else:
        ct = None

    if product == 'Synthetic':
        bands = 7
    else:
        bands = 1

    return create(path, 5000, 5000, aff, datatype, ct=ct, bands=bands, proj=cu_wkt)


def makepath(root: str, chip_x: float, chip_y: float, prod: str, date: str, trunc_dates: bool) -> str:
    return os.path.join(root, filename(chip_x, chip_y, prod, date, trunc_dates))


def multiout(output_q, outdir, count, cliargs):
    log.debug('Worker count: %s', count)
    kill_cnt = 0
    dss = {}
    tot = 0

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    while True:
        if kill_cnt >= count:
            log.debug('Shutting down output')
            break

        chip_x, chip_y, out = output_q.get()

        if out == -1:
            kill_cnt += 1
            log.debug('Kill count: %s', kill_cnt)
            continue

        for prod in out:
            for date in out[prod]:
                outpath = makepath(outdir, chip_x, chip_y, prod, date, cliargs.trunc_dates)

                if outpath not in dss:
                    # log.debug('Outpath not in keys: %s', outpath)
                    # log.debug('Keys: %s', dss.keys())
                    log.debug('Making product output: %s', outpath)
                    dss[outpath] = maketile(outpath, chip_x, chip_y, prod)
                    dss[outpath] = None

                if prod == 'Synthetic':
                    # writesynthetic(dss[outpath], chip_x, chip_y, out[prod][date])
                    writesynthetic(outpath, chip_x, chip_y, out[prod][date])
                else:
                    # writechip(dss[outpath], chip_x, chip_y, out[prod][date])
                    writechip(outpath, chip_x, chip_y, out[prod][date])
        tot += 1
        log.debug('Total chips done: %s', tot)


if __name__ == '__main__':
    desc = '''
    Map generation tool for CCDC test results.
    This version is currently orientated towards "ncompare" CCD and the annualized classification.
    *Not to be used with production results* They will use a different formatting schema.
    This produces the standard LCMAP product suite for the standard years.
    '''.format(prods=validprods())

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('jdir',
                        type=str,
                        help='Input directory of JSON files to process')
    parser.add_argument('pdir',
                        type=str,
                        help='Input directory of pickle files to process')
    parser.add_argument('outdir',
                        type=str,
                        help='Output directory for GeoTiff maps')
    parser.add_argument('nlcdpath',
                        type=str,
                        help='File path to NLCD')
    parser.add_argument('--no-crosswalk',
                        dest='xwalk',
                        action='store_false',
                        default=True,
                        help='Turn off NLCD crosswalking if it has already been done')
    parser.add_argument('--cpu',
                        dest='cpu',
                        action='store',
                        type=int,
                        default=1,
                        help='Number processes to spawn')
    margs = parser.parse_args()

    main(margs)
