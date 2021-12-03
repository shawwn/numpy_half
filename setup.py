#!/usr/bin/env python3
def configuration(parent_package='',top_path=None):
    import numpy
    from distutils.errors import DistutilsError
    if numpy.__dict__.get('xfloat16') is not None:
        raise DistutilsError('The target NumPy already has a half/float16 type')
    from numpy.distutils.misc_util import Configuration
    config = Configuration('half',parent_package,top_path)
    config.add_extension('numpy_xhalf',['halffloat.h','halffloat.cc','numpy_half.cc'])
    #config.add_data_dir('tests')
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)
