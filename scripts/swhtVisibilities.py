#!/usr/bin/env python
"""
Perform a Spherical Wave Harmonic Transform on LOFAR ACC/XST data or widefield MS data (e.g. PAPER) to form a complex or Stokes dirty image dirty image
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys,os
import SWHT

try:
    import healpy as hp
    healpyEnabled = True
except ImportError:
    healpyEnabled = False

#try:
#    import casacore.tables as tbls
#except ImportError:
#    print 'Warning: could not import casacore.tables, will not be able to read measurement sets'

#import scipy.constants
#cc = scipy.constants.c
cc = 299792458.0 #speed of light, m/s

def main_cli():
    from optparse import OptionParser
    o = OptionParser()
    o.set_usage('%prog [options] ACC/XST/MS/PKL FILE')
    o.set_description(__doc__)
    o.add_option('--station', dest='station', default=None,
        help = 'LOFAR ONLY: station name, e.g. SE607, if this is used then the ant_field and ant_array options are not required, default: None')
    o.add_option('-F', '--ant_field', dest='ant_field', default=None,
        help = 'LOFAR ONLY: AntennaField.conf file for the LOFAR station of the ACC files, default: None')
    o.add_option('-A', '--ant_array', dest='ant_array', default=None,
        help = 'LOFAR ONLY(NOT REQUIRED): AntennaArray.conf file for the LOFAR station geographical coordinates, default: None')
    o.add_option('-D', '--deltas', dest='deltas', default=None,
        help = 'LOFAR ONLY: iHBADeltas.conf file, only required for HBA imaging, default: None')
    o.add_option('-r', '--rcumode', dest='rcumode', default=3, type='int',
        help = 'LOFAR ONLY: Station RCU Mode, usually 3,5,6,7, for XST it will override filename metadata default: 3(LBA High)')
    o.add_option('-s', '--subband', dest='subband', default='0',
        help = 'Select which subband(s) to image, for ACC and MS it will select, for multiple subbands use X,Y,Z and for range use X_Y notation, for XST it will override filename metadata, default:0')
    o.add_option('-p', '--pixels', dest='pixels', default=64, type='int',
        help = 'Width of 2D image in pixels, or number of steps in 3D image, or NSIDE in HEALPix image, default: 64')
    o.add_option('-C', '--cal', dest='calfile', default=None,
        help = 'LOFAR ONLY: Apply a calibration soultion file to the data.')
    o.add_option('-S', '--save', dest='savefig', default=None,
        help = 'Save the figure using this name, type is determined by extension')
    o.add_option('--nodisplay', dest='nodisplay', action='store_true',
        help = 'Do not display the generated image')
    o.add_option('--of', dest='of', default=None,
        help = 'Save complex images in a numpy array in a pickle file or HEALPix map using this name (include .pkl or .hpx extention), default: tempImage.pkl')
    o.add_option('-i', '--int', dest='int_time', default=1., type='float',
        help = 'LOFAR ONLY: Integration time, used for accurate zenith pointing, for XST it will override filename metadata, default: 1 second')
    o.add_option('-c', '--column', dest='column', default='CORRECTED_DATA', type='str',
        help = 'MS ONLY: select which data column to image, default: CORRECTED_DATA')
    o.add_option('--override', dest='override', action='store_true',
        help = 'LOFAR XST ONLY: override filename metadata for RCU, integration length, and subband')
    o.add_option('--autos', dest='autos', action='store_true',
        help = 'Include the auto-correlations in the image')
    o.add_option('--fov', dest='fov', default=180., type='float',
        help = '2D IMAGING MODE ONLY: Field of View in degrees, default: 180 (all-sky)')
    o.add_option('-l', '--lmax', dest='lmax', default=32, type='int',
        help = 'Maximum l spherical harmonic quantal number, rule-of-thumb: lmax ~ (pi/longest baseline resolution), default: 32')
    o.add_option('--lmin', dest='lmin', default=0, type='int',
        help = 'Minimum l spherical harmonic quantal number, usually left as 0, default: 0')
    o.add_option('--ocoeffs', dest='ocoeffs', default=None,
        help = 'Save output image coefficients to a pickle file using this name (include .pkl extention), default: tempCoeffs.pkl')
    o.add_option('-I', '--image', dest='imageMode', default='2D',
        help='Imaging mode: 2D (hemisphere flattened), 3D, healpix, coeff (coefficients) default: 2D')
    o.add_option('--uvwplot', dest='uvwplot', action='store_true',
        help='Display a 3D UVW coverage/sampling plot')
    o.add_option('--psf', dest='psf', action='store_true',
        help='Plot the PSF instead of the image')
    o.add_option('-t', '--times', dest='times', default='0',
        help = 'KAIRA ONLY: Select which integration(s) to image, can use a[seconds] to average, d[step size] to decimate, of a specific range of integrations similar to the subband selection option, default:0 (select the first integration of the file)')
    o.add_option('--pol', dest='polMode', default='I',
        help='Polarization selection: I, Q, U, V, XX, YY, XY, YX, default: I')
    o.add_option('-u', '--uv', dest='local_uv', action='store_true',
                 help='Use local uv coord sys instead of RA-DEC uvw')
    opts, args = o.parse_args(sys.argv[1:])
    swht_visibilities(args, opts)


def swht_visibilities(args, opts):
    ####################
    ## Read Visibilities
    ####################
    visFiles = args # filenames to image
    fDict, visComb, uvwComb, freqs, obsLong, obsLat, LSTangle, imgCoeffs =\
        SWHT.fileio.read_lofvis(visFiles, opts.station, opts.ant_field,
            opts.ant_array, opts.deltas, opts.subband, opts.times,
            opts.override, opts.rcumode, opts.int_time, opts.calfile,
            opts.column, opts.local_uv)

    if opts.uvwplot: # display the total UVW coverage
        fig, ax = SWHT.display.dispVis3D(uvwComb)
        plt.show()

    ####################
    ## Decompose the input visibilities into spherical harmonics visibility coefficeints
    ####################
    decomp = not imgCoeffs
    if decomp:
        # compute the ideal l_max given the average solid angle angular resolution of an l-mode is Omega ~ 4pi / 2l steradian, and if the PSF is circular theta ~ pi / l radians
        blLen = np.sqrt(uvwComb[:,0,:]**2. + uvwComb[:,1,:]**2. + uvwComb[:,2,:]**2.) # compute the baseline lengths (in meters)
        maxBl = np.max(blLen) # maximum baseline length (in meters)
        meanWl = cc / np.mean(freqs) # mean observing wavelength
        maxRes = 1.22 * meanWl / maxBl
        print('MAXIMUM RES: %f (radians) %f (deg)'%(maxRes, maxRes * (180. / np.pi)))
        idealLmax = int(np.pi / maxRes)
        print('SUGGESTED L_MAX: %i, %i (oversample 3), %i (oversample 5)'%(idealLmax, idealLmax*3, idealLmax*5))

        if opts.psf:
           visComb = np.ones_like(visComb) 

        print('AUTO-CORRELATIONS:', opts.autos)
        if not opts.autos: # remove auto-correlations
            autoIdx = np.argwhere(uvwComb[:,0]**2. + uvwComb[:,1]**2. + uvwComb[:,2]**2. == 0.)
            visComb[:,autoIdx] = 0.

        # prepare for SWHT
        print('Performing Spherical Wave Harmonic Transform')
        print('LMAX:', opts.lmax)

        polMode = opts.polMode.upper()
        print('Polarization Mode:', polMode)
        if polMode=='I': polVisComb = visComb[0] + visComb[3]
        elif polMode=='Q': polVisComb = visComb[0] - visComb[3]
        elif polMode=='U': polVisComb = visComb[1] + visComb[2]
        elif polMode=='V': polVisComb = 1j * np.conj(visComb[1] - visComb[2]) #flip imaginary and real
        elif polMode=='XX': polVisComb = visComb[0]
        elif polMode=='XY': polVisComb = visComb[1]
        elif polMode=='YX': polVisComb = visComb[2]
        elif polMode=='YY': polVisComb = visComb[3]

        imgCoeffs = SWHT.swht.swhtImageCoeffs(polVisComb, uvwComb, freqs, lmax=opts.lmax, lmin=opts.lmin) # perform SWHT

        # save image coefficients to file
        if opts.ocoeffs is None: outCoeffPklFn = 'tempCoeffs.pkl'
        else: outCoeffPklFn = opts.pkl
        SWHT.fileio.writeCoeffPkl(outCoeffPklFn, imgCoeffs, [float(obsLong), float(obsLat)], float(LSTangle))

    ####################
    ## Imaging
    ####################
    if opts.of is not None:
        if opts.imageMode.startswith('heal'):
            healpix_image_suffix = '_image.hpx'
            if not opts.of.endswith(healpix_image_suffix):
                opts.of += healpix_image_suffix
        else:
            pkl_image_suffix = '_image.pkl'
            if not opts.of.endswith(pkl_image_suffix):
                opts.of += pkl_image_suffix
        if os.path.exists(opts.of):
            os.remove(opts.of)

    #TODO: not doing the correct projection
    if opts.imageMode.startswith('2'): # Make a 2D hemispheric image
        fov = opts.fov * (np.pi/180.) # Field of View in radians
        px = [opts.pixels, opts.pixels]
        res = fov/px[0] # pixel resolution
        print('Generating 2D Hemisphere Image of size (%i, %i)'%(px[0], px[1]))
        print('Resolution(deg):', res*180./np.pi)
        if opts.local_uv:
            # Put zenith (lcl coordsys) at center of image
            zen = [0., np.pi/2]
            img = SWHT.swht.make2Dimage(imgCoeffs, res, px, phs=zen)
        else:
            # Put North celestial-pole at center of image since it is a cardinal
            # direction that most LOFAR stations have in their LBA FoV, and
            # align RA 00h (GST angle) to be at plot azimuth 0.
            ncp_gst = [0., np.pi/2]
            scp_gst = [0., -np.pi/2]
            img = SWHT.swht.make2Dimage(imgCoeffs, res, px, phs=ncp_gst)
        fig, ax = SWHT.display.disp2D(img, dmode='abs', cmap='jet',
                                      azi_north=True)

        if opts.of:
            # save complex image to pickle file
            print('Writing image to file %s ...'%opts.of, end=' ')
            SWHT.fileio.writeSWHTImgPkl(opts.of, img, fDict, mode='2D')
        print('done')

    elif opts.imageMode.startswith('3'): # Make a 3D equal stepped image
        print('Generating 3D Image with %i steps in theta and %i steps in phi'%(opts.pixels, opts.pixels))
        img, phi, theta = SWHT.swht.make3Dimage(imgCoeffs, dim=[opts.pixels, opts.pixels])
        fig, ax = SWHT.display.disp3D(img, phi, theta, dmode='abs', cmap='jet')

        if opts.of:
            # save complex image to pickle file
            print('Writing image to file %s ...'%opts.of, end=' ')
            SWHT.fileio.writeSWHTImgPkl(opts.of, [img, phi, theta], fDict, mode='3D')
        print('done')

    elif opts.imageMode.startswith('heal'): # plot healpix and save healpix file using the opts.pkl name
        print('Generating HEALPix Image with %i NSIDE'%(opts.pixels))
        # use the healpy.alm2map function as it is much faster, there is a ~1% difference between the 2 functions, this is probably due to the inner workings of healpy
        #m = SWHT.swht.makeHEALPix(imgCoeffs, nside=opts.pixels) # TODO: a rotation issue
        m = hp.alm2map(SWHT.util.array2almVec(imgCoeffs), opts.pixels) # TODO: a rotation issue

        if opts.of:
            # save complex image to HEALPix file
            print('Writing image to file %s ...'%opts.of, end=' ')
            hp.write_map(opts.of, m.real, coord='C') # only writing the real component, this should be fine, maybe missing some details, but you know, the sky should be real.
        print('done')
    
    elif opts.imageMode.startswith('coeff'): # plot the complex coefficients
        fig, ax = SWHT.display.dispCoeffs(imgCoeffs, zeroDC=True, vis=False)

    if not (opts.savefig is None): plt.savefig(opts.savefig)
    if not opts.nodisplay:
        if opts.imageMode.startswith('heal'): hp.cartview(m.real, coord='CG')
        plt.show()


if __name__ == '__main__':
    main_cli()