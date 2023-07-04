from extra_data import open_run
import numpy as np
from .dispersiveXanesAlignment import doShot, g_fit_default_kw, plotShot, subtractBkg
from functools import partial
from .mcutils import rebin
from scipy.ndimage import zoom


class EuXFELRun:
    def __init__(self, proposal: int, run: int):
        self._run = open_run(proposal, run)

        self.cam_0 = self._run["HED_EXP_ZYLA/CAM/2:daqOutput", "data.image.pixels"]
        self.cam_1 = self._run["HED_EXP_ZYLA/CAM/42:daqOutput", "data.image.pixels"]

        self.init_alignment = g_fit_default_kw

    def _get_frame(self, keydata, index, roi=np.s_[:], transform=lambda x: x):
        raw = keydata[index].ndarray().squeeze()
        raw = transform(raw)
        cor = subtractBkg(raw, bkg_type="line").squeeze()  # bg subtraction
        # rebin data
        cor = cor[roi]
        if cor.shape[-1] != 1024:
            cor = zoom(cor, 1024.0 / np.array(cor.shape))
        return cor.astype(np.float64)

    def do_shot(self, index, roi_0=np.s_[:], roi_1=np.s_[:], fit=True):
        im0 = self._get_frame(self.cam_0, index, roi_0)
        im1 = self._get_frame(self.cam_1, index, roi_1, partial(np.rot90, k=2))

        res = doShot(im1, im0, self.init_alignment.copy(), doFit=fit)

        self.init_alignment = res.final_pars.copy()
        # TODO save transform
        return res, im0, im1


if __name__ == "__main__":
    r = EuXFELRun(2957, 144)
    res = r.do_shot(10, np.s_[950:1250, 1000:1400], np.s_[800:1100, 800:1200])
    plotShot(res.im1, res.im2, res=res)
