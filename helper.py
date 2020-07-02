import numpy as np
from astropy.stats import sigma_clip

def get_data(path):
    """Retrieve binned data.

    Args:
        path (string): Full path to the data file output by photometry routine.
        mode (string): The string specifying the detector and astrophysical model to use.
        cut (int): Number of data points to remove from the start of the arrays.

    Returns:
        tuple: flux (ndarray; Flux extracted for each frame),
            flux_err (ndarray; uncertainty on the flux for each frame),
            time (ndarray; Time stamp for each frame),
            xdata (ndarray; X-coordinate of the centroid for each frame),
            ydata (ndarray; Y-coordinate of the centroid for each frame), 
            psfwx (ndarray; X-width of the target's PSF for each frame), 
            psfwy (ndarray; Y-width of the target's PSF for each frame).

    """
    
    flux     = np.loadtxt(path, usecols=[0], skiprows=1)     # mJr/str
    flux_err = np.loadtxt(path, usecols=[1], skiprows=1)     # mJr/str
    time     = np.loadtxt(path, usecols=[2], skiprows=1)     # BMJD
    xdata    = np.loadtxt(path, usecols=[4], skiprows=1)     # pixel
    ydata    = np.loadtxt(path, usecols=[6], skiprows=1)     # pixel
    psfxw = np.loadtxt(path, usecols=[8], skiprows=1)     # pixel
    psfyw = np.loadtxt(path, usecols=[10], skiprows=1)    # pixel

    factor = 1/(np.nanmedian(flux))
    flux = factor*flux
    flux_err = factor*flux

    order = np.argsort(time)
    flux = flux[order]
    flux_err = flux_err[order]
    time = time[order]
    xdata = xdata[order]
    ydata = ydata[order]
    psfxw = psfxw[order]
    psfyw = psfyw[order]

    # Sigma clip per data cube (also masks invalids)
    FLUX_clip  = sigma_clip(flux, sigma=6, maxiters=1)
    FERR_clip  = sigma_clip(flux_err, sigma=6, maxiters=1)
    XDATA_clip = sigma_clip(xdata, sigma=6, maxiters=1)
    YDATA_clip = sigma_clip(ydata, sigma=6, maxiters=1)
    PSFXW_clip = sigma_clip(psfxw, sigma=6, maxiters=1)
    PSFYW_clip = sigma_clip(psfyw, sigma=6, maxiters=1)

    # Ultimate Clipping
    MASK  = FLUX_clip.mask + XDATA_clip.mask + YDATA_clip.mask + PSFXW_clip.mask + PSFYW_clip.mask
    mask = np.logical_not(MASK)

    flux = flux[mask]
    flux_err = flux_err[mask]
    time = time[mask]
    xdata = xdata[mask]
    ydata = ydata[mask]
    psfxw = psfxw[mask]
    psfyw = psfyw[mask]

    # redefining the zero centroid position
    mid_x, mid_y = np.nanmean(xdata), np.nanmean(ydata)
    xdata -= mid_x
    ydata -= mid_y
    
    #BMJD to JD
    time = time+2400000+0.5
    
    return flux, time, xdata, ydata
