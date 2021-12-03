from info import __doc__

__all__ = ['bfloat16']

import numpy as np
from .numpy_xhalf import bfloat16


if np.__dict__.get('bfloat16') is not None:
    raise RuntimeError('The NumPy package already has a bfloat16 type')

_bfloat16_dtype = np.dtype(bfloat16)

# Add bfloat16 into the numpy module space
np.bfloat16 = bfloat16

def add_to_typeDict():
    # Add it to the numpy type dictionary
    f16 = _bfloat16_dtype
    np.typeDict['bfloat16'] = f16
    # np.typeDict['f2'] = f16
    # np.typeDict['=f2'] = f16
    # import sys
    # if sys.byteorder == 'little':
    #     np.typeDict['<f2'] = f16
    #     np.typeDict['>f2'] = f16.newbyteorder('>')
    # else:
    #     typeDict['>f2'] = f16
    #     np.typeDict['<f2'] = f16.newbyteorder('<')


def _bfloat16_finfo():
  def float_to_str(f):
    return "%12.4e" % float(f)

  # bfloat16 = _bfloat16_dtype.type
  tiny = float.fromhex("0x1p-126")
  resolution = 0.01
  eps = float.fromhex("0x1p-7")
  epsneg = float.fromhex("0x1p-8")
  max = float.fromhex("0x1.FEp127")

  obj = object.__new__(np.finfo)
  obj.dtype = _bfloat16_dtype
  obj.bits = 16
  obj.eps = bfloat16(eps)
  obj.epsneg = bfloat16(epsneg)
  obj.machep = -7
  obj.negep = -8
  obj.max = bfloat16(max)
  obj.min = bfloat16(-max)
  obj.nexp = 8
  obj.nmant = 7
  obj.iexp = obj.nexp
  obj.precision = 2
  obj.resolution = bfloat16(resolution)
  obj.tiny = bfloat16(tiny)
  obj.machar = None  # np.core.getlimits.MachArLike does not support bfloat16.

  obj._str_tiny = float_to_str(tiny)
  obj._str_max = float_to_str(max)
  obj._str_epsneg = float_to_str(epsneg)
  obj._str_eps = float_to_str(eps)
  obj._str_resolution = float_to_str(resolution)
  return obj


def add_to_finfo():
    # # Inject float16 into the finfo cache
    # # by hijacking the normal construction method
    # fi = object.__new__(np.finfo)
    # fi.dtype = np.dtype(bfloat16)
    # fmt = '%6.4e'

    # # The MachAr implementation does not work, hardcode everything...
    # fi.machar = None

    # fi.precision = 3
    # fi.resolution = xfloat16(10**-fi.precision)
    # fi.tiny = xfloat16(2.0**-14) # Smallest positive normalized number
    # fi.iexp = 5
    # fi.maxexp = 16
    # fi.minexp = -14
    # fi.machep = -10
    # fi.negep = -11
    # fi.max = xfloat16(65504)
    # fi.min = xfloat16(-65504)
    # fi.eps = xfloat16(2.0**fi.machep)
    # fi.epsneg = xfloat16(2.0**fi.negep)
    # fi.nexp = fi.iexp
    # fi.nmant = 10
    # fi._str_tiny = (fmt % fi.tiny).strip()
    # fi._str_max = (fmt % fi.max).strip()
    # fi._str_epsneg = (fmt % fi.epsneg).strip()
    # fi._str_eps = (fmt % fi.eps).strip()
    # fi._str_resolution = (fmt % fi.resolution).strip()

    fi = _bfloat16_finfo()

    np.finfo._finfo_cache[fi.dtype] = fi

add_to_typeDict()
add_to_finfo()

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
