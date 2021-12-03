from info import __doc__

__all__ = ['xfloat16']

import numpy
from .numpy_xhalf import xfloat16

if numpy.__dict__.get('xfloat16') is not None:
    raise RuntimeError('The NumPy package already has a half/xfloat16 type')

# Add xfloat16 into the numpy module space
numpy.xhalf = xfloat16
numpy.xfloat16 = xfloat16

def add_to_typeDict():
    # Add it to the numpy type dictionary
    import sys
    f16 = numpy.dtype(xfloat16)
    numpy.typeDict['xfloat16'] = f16
    # numpy.typeDict['f2'] = f16
    # numpy.typeDict['=f2'] = f16
    # if sys.byteorder == 'little':
    #     numpy.typeDict['<f2'] = f16
    #     numpy.typeDict['>f2'] = f16.newbyteorder('>')
    # else:
    #     typeDict['>f2'] = f16
    #     numpy.typeDict['<f2'] = f16.newbyteorder('<')

def add_to_finfo():
    # Inject float16 into the finfo cache
    # by hijacking the normal construction method
    fi = object.__new__(numpy.finfo)
    fi.dtype = numpy.dtype(xfloat16)
    fmt = '%6.4e'

    # The MachAr implementation does not work, hardcode everything...
    fi.machar = None

    fi.precision = 3
    fi.resolution = xfloat16(10**-fi.precision)
    fi.tiny = xfloat16(2.0**-14) # Smallest positive normalized number
    fi.iexp = 5
    fi.maxexp = 16
    fi.minexp = -14
    fi.machep = -10
    fi.negep = -11
    fi.max = xfloat16(65504)
    fi.min = xfloat16(-65504)
    fi.eps = xfloat16(2.0**fi.machep)
    fi.epsneg = xfloat16(2.0**fi.negep)
    fi.nexp = fi.iexp
    fi.nmant = 10
    fi._str_tiny = (fmt % fi.tiny).strip()
    fi._str_max = (fmt % fi.max).strip()
    fi._str_epsneg = (fmt % fi.epsneg).strip()
    fi._str_eps = (fmt % fi.eps).strip()
    fi._str_resolution = (fmt % fi.resolution).strip()

    numpy.finfo._finfo_cache[fi.dtype] = fi

add_to_typeDict()
add_to_finfo()

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
