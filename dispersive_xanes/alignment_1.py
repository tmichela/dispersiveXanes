import iminuit
import numpy as np


def transforms(
    intensity,
    igauss1cen,
    igauss1sig,
    iblur1,
    scalex,
    scaley,
    rotation,
    transx,
    transy,
    shear,
    igauss2cen,
    igauss2sig,
    iblur2,
):
    t1 = SpecrometerTransformation(
        translation=(transx, transy),
        scale=(scalex, scaley),
        rotation=rotation,
        shear=shear,
        intensity=intensity,
        igauss=(igauss1cen, igauss1sig),
        iblur=iblur1,
    )
    t2 = SpecrometerTransformation(igauss=(igauss2cen, igauss2sig), iblur=iblur2)
    return t1, t2


def transform_images(im0, im1, parameters, background_threshold=0.05):
    spec0 = im0.astype(np.float32, copy=True)
    spec1 = im1.astype(np.float32, copy=True)

    # set anything below the 5% of the max to zero
    spec0[spec0 < spec0.max() * background_threshold] = 0.0
    spec1[spec1 < spec1.max() * background_threshold] = 0.0
