"""
Utility functions
"""

import numpy as np
import datetime

def sph2cart(theta, phi, r=None):
    """Convert spherical coordinates to 3D cartesian
    theta, phi, and r must be the same size and shape, if no r is provided then unit sphere coordinates are assumed (r=1)
    theta: colatitude/elevation angle, 0(north pole) =< theta =< pi (south pole)
    phi: azimuthial angle, 0 <= phi <= 2pi
    r: radius, 0 =< r < inf
    returns X, Y, Z arrays of the same shape as theta, phi, r
    see: http://mathworld.wolfram.com/SphericalCoordinates.html
    """
    if r is None: r = np.ones_like(theta) #if no r, assume unit sphere radius

    #elevation is pi/2 - theta
    #azimuth is ranged (-pi, pi]
    X = np.cos((np.pi/2.)-theta) * np.cos(phi-np.pi)
    Y = np.cos((np.pi/2.)-theta) * np.sin(phi-np.pi)
    Z = np.sin((np.pi/2.)-theta)

    return X, Y, Z

def cart2sph(X, Y, Z):
    """Convert 3D cartesian coordinates to spherical coordinates
    X, Y, Z: arrays of the same shape and size
    returns r: radius, 0 =< r < inf
            phi: azimuthial angle, 0 <= phi <= 2pi
            theta: colatitude/elevation angle, 0(north pole) =< theta =< pi (south pole)
    see: http://mathworld.wolfram.com/SphericalCoordinates.html
    """
    r = np.sqrt(X**2. + Y**2. + Z**2.)
    phi = np.arctan2(Y, X) + np.pi #convert azimuth (-pi, pi] to phi (0, 2pi]
    theta = np.pi/2. - np.arctan2(Z, np.sqrt(X**2. + Y**2.)) #convert elevation [pi/2, -pi/2] to theta [0, pi]

    return r, phi, theta

def vectorize(mat):
    """Convert upper-left triangle of mat to rank 1 vector
    """
    idx = np.triu_indices(mat.shape[0])
    return mat[idx]

def vectorize3D(mat):
    """Convert upper-left triangle of a 3D array to rank 1 vector, assumes the first axis is 
    """
    idx = np.triu_indices(mat.shape[1])
    return mat[:, idx[0], idx[1]]

def convert_arg_range(arg):
    """Split apart command-line lists/ranges into a list of numbers."""
    if arg is None: return None

    arg = arg.split(',')
    outList = []
    for aa in arg:
        rr = list(map(int, aa.split('_')))
        if len(rr)==1: outList.append(rr[0])
        elif len(rr)==2:
            outList.extend(range(rr[0], rr[1]+1))
    return outList

def meanTimeDelta(l):
    """Return the mean of a list of datetime.timedelta objects"""
    tDelta = datetime.timedelta(0)
    for td in l:
        tDelta += td
    return tDelta/len(l)

#Functions taken from healpy/sphtfunc.py
class Alm(object):
    """This class provides some static methods for alm index computation.

    Methods
    -------
    getlm
    getidx
    getsize
    getlmax
    """
    def __init__(self):
        pass

    @staticmethod
    def getlm(lmax,i=None):
        """Get the l and m from index and lmax.
        
        Parameters
        ----------
        lmax : int
          The maximum l defining the alm layout
        i : int or None
          The index for which to compute the l and m.
          If None, the function return l and m for i=0..Alm.getsize(lmax)
        """
        if i is None:
            i=np.arange(Alm.getsize(lmax))
        m=(np.ceil(((2*lmax+1)-np.sqrt((2*lmax+1)**2-8*(i-lmax)))/2)).astype(int)
        l = i-m*(2*lmax+1-m)//2
        return (l,m)

    @staticmethod
    def getidx(lmax,l,m):
        """Returns index corresponding to (l,m) in an array describing alm up to lmax.
        
        Parameters
        ----------
        lmax : int
          The maximum l, defines the alm layout
        l : int
          The l for which to get the index
        m : int
          The m for which to get the index

        Returns
        -------
        idx : int
          The index corresponding to (l,m)
        """
        return m*(2*lmax+1-m)//2+l

    @staticmethod
    def getsize(lmax,mmax = None):
        """Returns the size of the array needed to store alm up to *lmax* and *mmax*

        Parameters
        ----------
        lmax : int
          The maximum l, defines the alm layout
        mmax : int, optional
          The maximum m, defines the alm layout. Default: lmax.

        Returns
        -------
        size : int
          The size of the array needed to store alm up to lmax, mmax.
        """
        if mmax is None or mmax < 0 or mmax > lmax:
            mmax = lmax
        return mmax * (2 * lmax + 1 - mmax) // 2 + lmax + 1

    @staticmethod
    def getlmax(s, mmax = None):
        """Returns the lmax corresponding to a given array size.
        
        Parameters
        ----------
        s : int
          Size of the array
        mmax : None or int, optional
          The maximum m, defines the alm layout. Default: lmax.

        Returns
        -------
        lmax : int
          The maximum l of the array, or -1 if it is not a valid size.
        """
        if mmax is not None and mmax >= 0:
            x = (2 * s + mmax ** 2 - mmax - 2) / (2 * mmax + 2)
        else:
            x = (-3 + np.sqrt(1 + 8 * s)) / 2
        if x != np.floor(x):
            return -1
        else:
            return int(x)

def almVec2array(vec, lmax):
    """Convert the vector output of healpy.map2alm into a 2-D array of the same format as used in the SWHT
    healpy.map2alm returns coefficients for 0=<l<=lmax and 0<=m<=l
    vec: output of healpy.map2alm
    lmax: maximum l number used in healpy.map2alm"""
    lmaxp1 = lmax + 1 #account for the 0 mode
    coeffs = np.zeros((lmaxp1, 2*lmaxp1-1), dtype='complex')

    for l in np.arange(lmaxp1):
        for m in np.arange(l+1):
            #These calls use the healpy Alm() calls which are reproduced in util.py
            #coeffs[l,l-m] = ((-1.)**m) * np.conj(vec[hp.Alm.getidx(lmax,l,m)]) #since the map is real, the a_l,-m = (-1)**m * a_l,m.conj
            #coeffs[l,l+m] = vec[hp.Alm.getidx(lmax,l,m)]
            coeffs[l,l-m] = ((-1.)**m) * np.conj(vec[Alm.getidx(lmax,l,m)]) #since the map is real, the a_l,-m = (-1)**m * a_l,m.conj
            coeffs[l,l+m] = vec[Alm.getidx(lmax,l,m)]

    return coeffs

def array2almVec(arr):
    """Convert a 2-D array of coefficients used in the SWHT into a vector that is the same as that of healpy.map2alm such that healpy.alm2map can be called with the output vector
    healpy.map2alm returns coefficients for 0=<l<=lmax and 0<=m<=l
    arr: 2-D array of SWHT coefficients [lmax + 1, 2*lmax + 1]
    """
    lmax = arr.shape[0] - 1
    ncoeffs = Alm.getsize(lmax)
    vec = np.zeros(ncoeffs, dtype='complex')

    for l in np.arange(lmax + 1):
        for m in np.arange(l + 1):
            vec[Alm.getidx(lmax,l,m)] = arr[l,l+m]

    return vec


import ephem
def pos2uv_flat(antpos):
    """
    Convert 3D antenna positions (approx flat config) to 2D, UV baseline

    Parameters
    ----------
    antpos: (nrants, nrpols, 3) array
        3D positions of antennas

    Returns
    -------
    uv : (nrants, nrants, 3) array
        Flat 3D baselines of configuration
    """
    # antpos is of the form [nants, npol, 3]
    nants = antpos.shape[0]
    # Compute baselines in XYZ
    ant_pos_rep = np.repeat(antpos[:, 0, :], nants, axis=0
                          ).reshape((nants, nants, 3))
    baselines = vectorize(ant_pos_rep - np.transpose(ant_pos_rep, (1, 0, 2)))

    # Compute local UV coord sys by first estimating normal to
    # xyz (uvw) array, using the fact that its normal vec is a
    # null space vector.
    _u_svd, _d_svd, _vt_svd = np.linalg.svd(baselines)
    nrmvec = -_vt_svd[2, :] / np.linalg.norm(_vt_svd[2, :])
    lon_nrm = np.arctan2(nrmvec[1], nrmvec[0])
    lat_nrm = np.arcsin(nrmvec[2])
    # Transform by rotations xyz to local UV crd sys, which has
    # normal along its z-axis and long-axis along x-axis:
    # First rotate around z so nrmvec x is in (+x,+z) quadrant
    # (this means Easting is normal to longitude 0 meridian plane)
    _rz = np.array([[np.cos(lon_nrm), np.sin(lon_nrm), 0.],
                    [-np.sin(lon_nrm), np.cos(lon_nrm), 0.],
                    [0., 0., 1.]])
    # Second rotate around y until normal vec is along z (NCP)
    _tht = (np.pi / 2 - lat_nrm)
    _ry = np.array([[np.cos(_tht), 0., -np.sin(_tht)],
                    [0., 1., 0.],
                    [+np.sin(_tht), 0., np.cos(_tht)]])
    # Third rotate around z so Easting is along (final) x
    _r_xy90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    rot_mat = _r_xy90 @ _ry @ _rz
    uv = baselines @ rot_mat.T
    return uv


def pos2uvw(antpos, dattims):
    # ants is of the form [nants, npol, 3]
    nants = antpos.shape[0]
    # Compute baselines in XYZ
    antPosRep = np.repeat(antpos[:, 0, :], nants, axis=0).reshape(
        (nants, nants, 3))
    baselines = vectorize(antPosRep - np.transpose(antPosRep, (1, 0, 2)))
    uvw = []

    # Setup Greenwich observer for computing GW Sidereal time & Earth rot angle
    GWobs = ephem.Observer()
    GWobs.long = 0.
    GWobs.lat = 0.
    GWobs.elevation = 0.
    if not isinstance(dattims, list):
        dattims = [dattims]
    for dattim in dattims:
        GWobs.epoch = dattim
        GWobs.date = dattim
        GST = GWobs.sidereal_time()
        ERA = float(GST)  # Earth rotation angle

        # Rotation matricies for XYZ -> UVW transform
        era_rotm = -np.array([[np.cos(ERA), -np.sin(ERA), 0.],
                              [np.sin(ERA), +np.cos(ERA), 0.],
                              [0., 0., -1.]])  # rotate about z-axis, xy-flip

        uvw.append(np.dot(era_rotm, baselines.T).T)
    return np.asarray(uvw)

import numpy as np

if __name__ == '__main__':
    print('Running test cases')

    [theta, phi] = np.meshgrid(np.linspace(0, np.pi, num=128, endpoint=False), np.linspace(0, 2.*np.pi, num=128, endpoint=False))
    X, Y, Z = sph2cart(theta, phi)
    r0, phi0, theta0 = cart2sph(X, Y, Z)

    print(np.allclose(theta, theta0))
    print(np.allclose(phi, phi0))

    aa = np.arange(5*5).reshape(5,5)
    print(aa)
    print(vectorize(aa))

    from matplotlib import pyplot as plt
    plt.subplot(221)
    plt.imshow(theta, interpolation='nearest')
    plt.colorbar()
    plt.subplot(222)
    plt.imshow(phi, interpolation='nearest')
    plt.colorbar()
    plt.subplot(223)
    plt.imshow(theta0, interpolation='nearest')
    plt.colorbar()
    plt.subplot(224)
    plt.imshow(phi0, interpolation='nearest')
    plt.colorbar()
    plt.show()

    print('Made it through without any errors.')

