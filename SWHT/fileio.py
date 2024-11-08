"""
functions to read/write data for the imagers
"""

"""
TODO: hdf5 wrappers for visibilities and images
    visibilities:
        LOFAR (XST or ACC) visibilities
        UVW corrdinates
        Flags
        Station Information
    images:
        SWHT coefficients
        Imaging history
        Observation history
"""

import pickle as pkl
import numpy as np
import datetime
import ephem

from SWHT import ecef, lofarConfig, util

def parse(fn, fmt=None):
    """Parse an input visibility filename to determine meta data and type
    XST files in standard format: <date>_<time>_sb<subband>_xst.dat
    XST files in the SE607 format: <date>_<time>_rcu<id>_sb<subband>_int<integration length>_dur<duration of observation>[_<HBA config in hex>]_xst.dat
    fmt: if None then automatically determines format based on filename, else can be set to 'ms' (measurement set), 'acc' (LOFAR ACC), 'xst' (LOFAR XST)
    returns: dictionary"""
    fDict = {}
    fDict['fn'] = fn
    if fn.lower().endswith('.ms') or fmt=='ms':
        fDict['fmt'] = 'ms'
    elif fmt=='KAIRA':
        # KAIRA is a special LOFAR-like station, the standard XST format is:
        # filename: [YYYYMMDD]_[HHMMSS]_xst.dat
        # 1 second integrations
        # 48 antennas (= 96 RCUs)
        # Nyquist mode I (= LBA array)
        # Subband 195 (= approx. 38.1 MHz)
        # ~3600 integrations per XST file
        fDict['fmt'] = fmt
        metaData = fn.split('/')[-1].split('_')
        fDict['ts'] = datetime.datetime(year=int(metaData[0][:4]), month=int(metaData[0][4:6]), day=int(metaData[0][6:]), hour=int(metaData[1][:2]), minute=int(metaData[1][2:4]), second=int(metaData[1][4:]))
        fDict['rcu'] = 1
        fDict['sb'] = np.array([195])
        fDict['int'] = 1.
        fDict['dur'] = 1.
    elif fn.lower().endswith('.dat') or fn.lower().endswith('.dat.sim') or fmt=='acc' or fmt=='xst':
        metaData = fn.split('/')[-1].split('_')
        fDict['ts'] = datetime.datetime(year=int(metaData[0][:4]), month=int(metaData[0][4:6]), day=int(metaData[0][6:]), hour=int(metaData[1][:2]), minute=int(metaData[1][2:4]), second=int(metaData[1][4:]))
        if metaData[2].startswith('acc'): # the file is a LOFAR ACC file
            fDict['fmt'] = 'acc'
            fDict['shape'] = map(int, metaData[3].split('.')[0].split('x'))
        elif metaData[-1].startswith('xst.dat'):
            if len(metaData) == 4: # Standard XST file
                fDict['fmt'] = 'xst'
                fDict['sb'] = np.array( [int(metaData[2][2:])] )
                fDict['rcu'] = 1  # default, overridden in scripts
                fDict['int'] = 1. # default, overridden in scripts
                fDict['dur'] = 1. # default, overridden in scripts
            else: # the file is a SE607 format LOFAR XST file
                fDict['fmt'] = 'xst'
                fDict['rcu'] = int(metaData[2][3:])
                fDict['sb'] = np.array( [int(metaData[3][2:])] )
                fDict['int'] = float(metaData[4][3:])
                fDict['dur'] = float(metaData[5][3:])
                if len(metaData)==8: # HBA all-sky file, get element identifiers
                    fDict['elem'] = metaData[6][2:]
    elif fn.lower().endswith('.pkl') or fmt=='pkl': # the file is a set of SWHT image coefficients
        fDict['fmt'] = 'pkl'
    elif fn.lower().endswith('coefs.hpx') or fmt=='coefs.hpx':
        # file is a set of SWHT image coefficients in healpy alm format
        fDict['fmt'] = 'coefs.hpx'
    elif fn.lower().endswith('npz') or fmt=='npz':
        fDict['fmt'] = 'npz'
    else:
        # unknown data format, returns warning
        fDict['fmt'] = -1
    return fDict

def writeCoeffPkl(fn, coeffs, phs=[0., 0.], lst=0.):
    """Write SWHT image coefficients to file
    fn: str, pickle filename
    coeffs: 2D array of complex coefficients
    phs: [float, float], RA and Dec (radians) position at the center of the image
    lst: float, local sidereal time of snapshot
    """
    coeffDict = {
        'coeffs': coeffs,
        'phs': phs,
        'lst': lst
    }
    fh = open(fn, 'wb')
    pkl.dump(coeffDict, fh)
    fh.close()
    
def readCoeffPkl(fn):
    """Read SWHT image coefficients from a pickle file, see writeCoeffPkl() for contents"""
    fh = open(fn,'rb')
    coeffDict = pkl.load(fh)
    fh.close()
    return coeffDict

def writeImgPkl(fn, d, fDict, res=None, fttype=None, imtype=None):
    """Write an image cube to a pickle file
    fn: str, pickle filename
    d: numpy array, image data
    fDict: dict, meta data from original visibility file
    res: float, resolution at zenith (radians)
    fftype: str, dft or fft convolution function name
    imtype: str, complex or Stokes"""
    imgDict = {
        'meta': fDict,
        'res': res,
        'fttype': fttype,
        'imtype': imtype,
        'img': d}
    fh = open(fn, 'wb')
    pkl.dump(imgDict, fh)
    fh.close()

def readImgPkl(fn):
    """Read an image cube from a pickle file, see writeImgPkl() for contents"""
    fh = open(fn,'rb')
    imgDict = pkl.load(fh)
    fh.close()
    return imgDict

def writeSWHTImgPkl(fn, d, fDict, mode):
    """Write a SWHT image cube to a pickle file
    fn: str, pickle filename
    d: numpy array, image data
    fDict: dict, meta data from original visibility file
    """
    imgDict = {
        'meta': fDict,
        'mode': mode}
    if mode.startswith('3D'):
        imgDict['img'] = d[0]
        imgDict['phi'] = d[1]
        imgDict['theta'] = d[2]
    else:
        imgDict['img'] = d
    fh = open(fn, 'wb')
    pkl.dump(imgDict, fh)
    fh.close()

def readSWHTImgPkl(fn):
    """Read an image cube from a pickle file, see writeSWHTImgPkl() for contents"""
    fh = open(fn,'rb')
    imgDict = pkl.load(fh)
    fh.close()
    return imgDict

def lofarArrayLatLong(lofarStation, arrayType='LBA'):
    """Return the Latitude, Longitude, Elevation of a LOFAR station
    lofarStation: instance, see lofarConfig.py
    arrayType: string, LOFAR array type

    returns: latitude (degs), longitude (degs), elevation (m)
    """
    #lon, lat, elev = lofarStation.antArrays.location[lofarConfig.rcuInfo[fDict['rcu']]['array_type']]
    arr_xyz = lofarStation.antField.location[arrayType]
    lat, lon, elev = ecef.ecef2geodetic(arr_xyz[0], arr_xyz[1], arr_xyz[2], degrees=True)
    print('LON(deg):', lon, 'LAT(deg):', lat, 'ELEV(m):', elev)

    return lat, lon, elev

def lofarHBAAntPositions(ants, lofarStation, elem):
    """Update the antenna positions using the HBADeltas file
    ants: [nants, 3] array, antenna positions in XYZ
    lofarStation: instance, see lofarConfig.py
    elem: hex/base-16 string of tile element IDs

    returns: updated [N, 3] antenna position array
    """
    if lofarStation.deltas is None:
        print('Warning: HBA element string found, but HBADeltas file is missing, your image is probably not going to make sense')
    else:
        print('Updating antenna positions with HBA element deltas')
        for aid in np.arange(ants.shape[0]):
            delta = lofarStation.deltas[int(elem[aid], 16)]
            delta = np.array([delta, delta])
            ants[aid] += delta

    return ants

def lofarFreqs(fDict, sbs):
    """Compute Frequency information from file meta data and subbands
    fDict: dictionary, file meta data, see parse()
    sbs: 1D array, subband IDs

    returns: [Nsubbands] array with frequency values in Hz
    """
    nchan = lofarConfig.rcuInfo[fDict['rcu']]['nchan']
    bw = lofarConfig.rcuInfo[fDict['rcu']]['bw']
    df = bw/nchan
    freqs = sbs*df + lofarConfig.rcuInfo[fDict['rcu']]['offset'] + (df/2.) # df/2 to centre the band

    return freqs, nchan, bw

def lofarACCSelectSbs(fn, sbs, nchan, nantpol, intTime, antGains=None):
    """Select subband correlation matricies from ACC file
    fn: string, ACC filename
    sbs: [Nsubbands] array
    nchan: int, number of total frequnecy channels
    nantpol: int, number of antenna-polarizations
    intTime: float, integration time in seconds
    antGains: antenna gains from lofarConfig.readCalTable()

    returns:
        sbCorrMatrix: correlation matrix from each subband [Nsubbands, 1, nantpol, nantpol]
        tDeltas: 2D array [Nsubbands, 1], time offsets for each subband from end of file timestep [Nsubbands]
    """
    tDeltas = [] # subband timestamp deltas from the end of file
    corrMatrix = np.fromfile(fn, dtype='complex').reshape(nchan, nantpol, nantpol) # read in the complete correlation matrix
    sbCorrMatrix = np.zeros((sbs.shape[0], nantpol, nantpol), dtype=complex)
    for sbIdx, sb in enumerate(sbs):
        if antGains is None:
            sbCorrMatrix[sbIdx] = corrMatrix[sb, :, :] # select out a single subband, shape (nantpol, nantpol)
        else: # Apply Gains
            sbAntGains = antGains[sb].T
            sbVisGains = np.dot(sbAntGains, np.conj(sbAntGains.T))
            sbCorrMatrix[sbIdx] = np.multiply(sbVisGains, corrMatrix[sb, :, :]) # select out a single subband, shape (nantpol, nantpol)

        # correct the time due to subband stepping
        tOffset = (nchan - sb) * intTime # the time stamp in the filename is for the last subband
        rem = tOffset - int(tOffset) # subsecond remainder
        tDeltas.append(datetime.timedelta(0, int(tOffset), rem*1e6))

    tDeltas = np.reshape(np.array(tDeltas), (sbs.shape[0], 1)) # put in the shape [Nsubbands, Nints]
    sbCorrMatrix = np.reshape(sbCorrMatrix, (sbs.shape[0], 1, nantpol, nantpol)) # add integration axis

    print('CORRELATION MATRIX SHAPE', corrMatrix.shape)
    print('REDUCED CORRELATION MATRIX SHAPE', sbCorrMatrix.shape)

    return sbCorrMatrix, tDeltas

def lofarXST(fn, sb, integration, nantpol, antGains=None):
    """Read in correlation matrix from a XST file
    fn: string, XST filename
    sb: [int], subband ID, 1 element list for consistency with lofarACCSelectSbs()
    integration: integration time in seconds.
    nantpol: int, number of antenna-polarizations
    antGains: antenna gains from lofarConfig.readCalTable()

    returns:
        sbCorrMatrix: correlation matrix from each subband [1, 1, nantpol, nantpol] for consistency with lofarACCSelectSbs()
        tDeltas: 2D array [1, 1], time offset from end of file timestep, set to 0 but kept for consistency with lofarACCSelectSbs()
    """
    corrMatrix = np.fromfile(fn, dtype='complex').reshape(-1, nantpol, nantpol) #read in the correlation matrix
    if antGains is None:
        sbCorrMatrix = corrMatrix[np.newaxis, ...] # shape (1, 1, nantpol, nantpol)
    else: # Apply Gains
        sbAntGains = antGains[sb].T
        sbVisGains = np.dot(sbAntGains, np.conj(sbAntGains.T))
        sbCorrMatrix = np.multiply(sbVisGains, corrMatrix) # shape (1, nantpol, nantpol)
        sbCorrMatrix = np.reshape(sbCorrMatrix, (1, 1, nantpol, nantpol)) # add integration axis
    nrsamps = corrMatrix.shape[0]
    tDeltas = [tic*datetime.timedelta(seconds=int(integration))
               for tic in range(nrsamps)]  # time offsets

    tDeltas = np.array(tDeltas)[np.newaxis] # put in the shape [Nsubbands, Nints]

    print('CORRELATION MATRIX SHAPE', corrMatrix.shape)
    print('REDUCED CORRELATION MATRIX SHAPE', sbCorrMatrix.shape)

    return sbCorrMatrix, tDeltas

def lofarKAIRAXST(fn, sb, nantpol, intTime, antGains=None, times='0'):
    """Read in correlation matrix from a KAIRA format XST file
    fn: string, XST filename
    sb: [int], subband ID, 1 element list for consistency with lofarACCSelectSbs()
    nantpol: int, number of antenna-polarizations
    antGains: antenna gains from lofarConfig.readCalTable()
    times: the KAIRA XST files contain ~3600 integrations of 1 second each, this will result in a slow SWHT, to reduce this a number of options can be used
        i) a[seconds] : average together integrations into correlation matrixs for every block of time, e.g. times='a600' will average for 600 seconds, which is reasonable for the low resolution KAIRA station
        ii) select unique integrations: for multiple integrations use X,Y,Z and for range use X_Y notation, these can be combined, e.g. 0,100,200,300_310,400, default to function is to select only the first integration
        iii) d[step size]: decimate the integrations to select an integration every 'step size', e.g. d600 will select every 600th integration

    returns:
        reducedCorrMatrix: correlation matrix from each subband and integration [1, tids.size, nantpol, nantpol] for consistency with lofarACCSelectSbs()
        tDeltas: 2D array [1, tids.size], time offset from end of file timestep, for consistency with lofarACCSelectSbs()
    """
    corrMatrix = np.fromfile(fn, dtype='complex') # read in the correlation matrix
    nints = corrMatrix.shape[0]/(nantpol * nantpol) # number of integrations
    corrMatrix = np.reshape(corrMatrix, (nints, nantpol, nantpol))

    # determine unique subbands to select
    if times.startswith('a'): # averaging
        intLen = float(times[1:]) * intTime
        nAvgInts = int(nints/intLen)
        print('KAIRA: avergaing XST file to %.2f second integrations, this will produce %i integrations'%(intLen, nAvgInts))
        # clip off extra integrations to make the initial array a factor of the number of averaged integrations
        # reshape array
        # compute the mean for the axis to produce the averaged array
        reducedCorrMatrix = np.mean(corrMatrix[:int(intLen * nAvgInts)].reshape( nAvgInts, intLen, nantpol, nantpol), axis=1)
        tids = np.linspace(intLen/2., intLen * (nAvgInts-0.5), nAvgInts) # take the centre integration time to be the time ID
    elif times.startswith('d'): #decimation
        decimateFactor = int(times[1:])
        tids = np.arange(nints)[::decimateFactor]
        print('KAIRA: decimating XST file to %i integrations'%(tids.shape[0]))
        reducedCorrMatrix = corrMatrix[tids]
    else: # unique IDs
        tids = np.array(util.convert_arg_range(times))
        print('KAIRA: selecting %i unique integrations'%(tids.shape[0]))
        reducedCorrMatrix = corrMatrix[tids]

    tDeltas = [] # integration timestamp deltas from the end of file
    if antGains is not None: # Apply Gains
        sbAntGains = antGains[sb].T
        sbVisGains = np.dot(sbAntGains, np.conj(sbAntGains.T))
        reducedCorrMatrix = sbVisGains * reducedCorrMatrix # Apply gains
    for tIdx, tid in enumerate(tids):
        #if antGains is not None: # Apply Gains
        #    # TODO: this multiplication can be done outside the for loop
        #    reducedCorrMatrix[tIdx] = np.multiply(sbVisGains, reducedCorrMatrix[tid, :, :]) # select out a single integration, shape (nantpol, nantpol)

        # correct the time relative to the EOF timestamp
        tOffset = (nints - tid - 1) * intTime # the timestamp in the filename is for the last integration
        rem = tOffset - int(tOffset) # subsecond remainder
        tDeltas.append(datetime.timedelta(0, int(tOffset), rem*1e6))

    tDeltas = np.array(tDeltas)[np.newaxis] # put in the shape [Nsubbands, Nints]
    reducedCorrMatrix = np.reshape(reducedCorrMatrix, (1, tids.shape[0], nantpol, nantpol)) # add subband axis

    print('ORIGINAL CORRELATION MATRIX SHAPE', corrMatrix.shape)
    print('REDUCED CORRELATION MATRIX SHAPE', reducedCorrMatrix.shape)

    return reducedCorrMatrix, tDeltas

def lofarObserver(lat, lon, elev, ts):
    """Create an ephem Observer for a LOFAR station
    lat: float, latitude (deg)
    lon: float, longitude (deg)
    elev: float, elevation (m)
    ts: datetime, EOF timestamp

    returns: ephem.Observer()
    """
    obs = ephem.Observer() #create an observer at the array location
    obs.long = lon * (np.pi/180.)
    obs.lat = lat * (np.pi/180.)
    obs.elevation = float(elev)
    obs.epoch = ts
    obs.date = ts
    
    return obs

def lofarGenUVW(corrMatrix, ants, obs, sbs, ts, local_uv=False):
    """Generate UVW coordinates from antenna positions, timestamps/subbands
    corrMatrix: [Nsubbands, Nints, nantpol, nantpol] array, correlation matrix for each subband, time integration
    ants: [Nantennas, 3] array, antenna positions in XYZ
    obs: ephem.Observer() of station
    sbs: [Nsubbands] array, subband IDs
    ts: datetime 2D array [Nsubbands, Nints], timestamp for each correlation matrix

    returns:
        vis: visibilities [4, Nsamples*Nints, Nsubbands]
        uvw: UVW coordinates [Nsamples*Nints, 3, Nsubbands]
    """
    nants = ants.shape[0]
    ncorrs = int(nants*(nants+1)/2)
    nints = ts.shape[1]
    uvw = np.zeros((nints, ncorrs, 3, len(sbs)), dtype=float)
    vis = np.zeros((4, nints, ncorrs, len(sbs)), dtype=complex) # 4 polarizations: xx, xy, yx, yy

    for sbIdx, sb in enumerate(sbs):
        for tIdx in np.arange(nints):
            if local_uv:
                _uvw = util.pos2uv_flat(ants)
            else:
                _uvw = util.pos2uvw(ants, ts[sbIdx, tIdx])
            uvw[tIdx, :, :, sbIdx] = _uvw

            # split up polarizations, vectorize the correlation matrix, and drop the lower triangle
            vis[0, tIdx, :, sbIdx] = util.vectorize(corrMatrix[sbIdx, tIdx, 0::2, 0::2])
            vis[1, tIdx, :, sbIdx] = util.vectorize(corrMatrix[sbIdx, tIdx, 1::2, 0::2])
            vis[2, tIdx, :, sbIdx] = util.vectorize(corrMatrix[sbIdx, tIdx, 0::2, 1::2])
            vis[3, tIdx, :, sbIdx] = util.vectorize(corrMatrix[sbIdx, tIdx, 1::2, 1::2])

    vis = np.reshape(vis, (vis.shape[0], vis.shape[1]*vis.shape[2], vis.shape[3])) 
    uvw = np.reshape(uvw, (uvw.shape[0]*uvw.shape[1], uvw.shape[2], uvw.shape[3])) 

    #TODO: i don't think we need to return the LST angle
    return vis, uvw, 0.

def readACC(fn, fDict, lofarStation, sbs, calTable=None, local_uv=False):
    """Return the visibilites and UVW coordinates from a LOFAR station ACC file
    fn: ACC filename
    fDict: dictionary of file format meta data, see parse()
    lofarStation: instance, see lofarConfig.py
    sbs: 1-D array of subband IDs (in range 0-511)
    calTable: station gain calibration table filename

    returns:
        vis: visibilities [4, Nsamples, Nsubbands]
        uvw: UVW coordinates [Nsamples, 3, Nsubbands]
        freqs: frequencies [Nsubbands]
        obsdata: [latitude, longitude, LST]
    """

    # longitude and latitude of array
    lat, lon, elev = lofarArrayLatLong(lofarStation, lofarConfig.rcuInfo[fDict['rcu']]['array_type'])

    # antenna positions
    ants = lofarStation.antField.antpos[lofarConfig.rcuInfo[fDict['rcu']]['array_type']]
    if 'elem' in fDict: # update the antenna positions if there is an element string
        ants = lofarHBAAntPositions(ants, lofarStation, fDict['elem'])
    nants = ants.shape[0]
    print('NANTENNAS:', nants)

    # frequency information
    freqs, nchan, bw = lofarFreqs(fDict, sbs)
    print('SUBBANDS:', sbs, '(', freqs/1e6, 'MHz)')
    npols = 2

    # read LOFAR Calibration Table
    if not (calTable is None):
        antGains = lofarStation.antField.antGains
        if antGains is None: # read the Cal Table only once
            print('Using CalTable:', calTable)
            antGains = lofarConfig.readCalTable(calTable, nants, nchan, npols)
            lofarStation.antField.antGains = antGains
        else:
            print('Using Cached CalTable:', calTable)
    else: antGains = None

    # get correlation matrix for subbands selected
    nantpol = nants * npols
    print('Reading in visibility data file ...', end=' ')
    corrMatrix, tDeltas = lofarACCSelectSbs(fn, sbs, nchan, nantpol, fDict['int'], antGains)
    print('done')
    #print corrMatrix.shape, tDeltas.shape
    
    # create station observer
    obs = lofarObserver(lat, lon, elev, fDict['ts'])
    obsLat = float(obs.lat) #radians
    obsLong = float(obs.long) #radians
    print('Observatory:', obs)

    # get the UVW and visibilities for the different subbands
    vis, uvw, LSTangle = lofarGenUVW(corrMatrix, ants, obs, sbs,
                                     fDict['ts']-np.array(tDeltas),
                                     local_uv=local_uv)

    return vis, uvw, freqs, [obsLat, obsLong, LSTangle], nants

def readXST(fn, fDict, lofarStation, sbs, calTable=None, local_uv=False):
    """Return the visibilites and UVW coordinates from a LOFAR XST format file
    fn: XST filename
    fDict: dictionary of file format meta data, see parse()
    lofarStation: instance, see lofarConfig.py
    sbs: 1-D array of subband IDs (in range 0-511)
    calTable: station gain calibration table filename

    returns:
        vis: visibilities [4, Nsamples, Nsubbands]
        uvw: UVW coordinates [Nsamples, 3, Nsubbands]
        freqs: frequencies [Nsubbands]
        obsdata: [latitude, longitude, LST]
    """

    # longitude and latitude of array
    lat, lon, elev = lofarArrayLatLong(lofarStation, lofarConfig.rcuInfo[fDict['rcu']]['array_type'])

    # antenna positions
    ants = lofarStation.antField.antpos[lofarConfig.rcuInfo[fDict['rcu']]['array_type']]
    antpos_topo = lofarStation.antField.localAntPos[lofarConfig.rcuInfo[fDict['rcu']]['array_type']]
    #ants=antpos_topo
    print('**ANTS**',ants.shape)
    if 'elem' in fDict: # update the antenna positions if there is an element string
        ants = lofarHBAAntPositions(ants, lofarStation, fDict['elem'])
    nants = ants.shape[0]
    print('NANTENNAS:', nants)

    # frequency information
    freqs, nchan, bw = lofarFreqs(fDict, sbs)
    print('SUBBANDS:', sbs, '(', freqs/1e6, 'MHz)')
    npols = 2

    # read LOFAR Calibration Table
    if not (calTable is None):
        antGains = lofarStation.antField.antGains
        if antGains is None: # read the Cal Table only once
            print('Using CalTable:', calTable)
            antGains = lofarConfig.readCalTable(calTable, nants, nchan, npols)
            lofarStation.antField.antGains = antGains
        else:
            print('Using Cached CalTable:', calTable)
    else: antGains = None

    # get correlation matrix for subbands selected
    nantpol = nants * npols
    print('Reading in visibility data file ...', end=' ')
    corrMatrix, tDeltas = lofarXST(fn, fDict['sb'], fDict['int'], nantpol, antGains)
    print('done')
    
    # create station observer
    obs = lofarObserver(lat, lon, elev, fDict['ts'])
    obsLat = float(obs.lat) #radians
    obsLong = float(obs.long) #radians
    print('Observatory:', obs)

    # get the UVW and visibilities for the different subbands
    vis, uvw, LSTangle = lofarGenUVW(corrMatrix, ants, obs, sbs,
                                     fDict['ts']+np.array(tDeltas),
                                     local_uv=local_uv)

    return vis, uvw, freqs, [obsLat, obsLong, LSTangle], nants

def readKAIRAXST(fn, fDict, lofarStation, sbs, calTable=None, times='0',
                 local_uv=False):
    """Return the visibilites and UVW coordinates from a KAIRA LOFAR XST format file
    fn: XST filename
    fDict: dictionary of file format meta data, see parse()
    lofarStation: instance, see lofarConfig.py
    sbs: 1-D array of subband IDs (in range 0-511)
    calTable: station gain calibration table filename
    times: the KAIRA XST files contain ~3600 integrations of 1 second each, this will result in a slow SWHT, to reduce this a number of options can be used
        i) a[seconds] : average together integrations into correlation matrixs for every block of time, e.g. times='a600' will average for 600 seconds, which is reasonable for the low resolution KAIRA station
        ii) select unique integrations: for multiple integrations use X,Y,Z and for range use X_Y notation, these can be combined, e.g. 0,100,200,300_310,400, default to function is to select only the first integration
        iii) d[step size]: decimate the integrations to select an integration every 'step size', e.g. d600 will select every 600th integration

    returns:
        vis: visibilities [4, Nsamples, Nsubbands]
        uvw: UVW coordinates [Nsamples, 3, Nsubbands]
        freqs: frequencies [Nsubbands]
        obsdata: [latitude, longitude, LST]
    """

    # longitude and latitude of array
    lat, lon, elev = lofarArrayLatLong(lofarStation, lofarConfig.rcuInfo[fDict['rcu']]['array_type'])

    # antenna positions
    ants = lofarStation.antField.antpos[lofarConfig.rcuInfo[fDict['rcu']]['array_type']]
    if 'elem' in fDict: # update the antenna positions if there is an element string
        ants = lofarHBAAntPositions(ants, lofarStation, fDict['elem'])
    nants = ants.shape[0]
    print('NANTENNAS:', nants)

    # frequency information
    freqs, nchan, bw = lofarFreqs(fDict, sbs)
    print('SUBBANDS:', sbs, '(', freqs/1e6, 'MHz)')
    npols = 2

    # read LOFAR Calibration Table
    if not (calTable is None):
        antGains = lofarStation.antField.antGains
        if antGains is None: # read the Cal Table only once
            print('Using CalTable:', calTable)
            antGains = lofarConfig.readCalTable(calTable, nants, nchan, npols)
            lofarStation.antField.antGains = antGains
        else:
            print('Using Cached CalTable:', calTable)
    else: antGains = None

    # get correlation matrix for subbands selected
    nantpol = nants * npols
    print('Reading in visibility data file ...', end=' ')
    corrMatrix, tDeltas = lofarKAIRAXST(fn, fDict['sb'], nantpol, fDict['int'], antGains, times=times)
    print('done')

    # create station observer
    obs = lofarObserver(lat, lon, elev, fDict['ts'])
    obsLat = float(obs.lat) #radians
    obsLong = float(obs.long) #radians
    print('Observatory:', obs)

    # get the UVW and visibilities for the different subbands
    vis, uvw, LSTangle = lofarGenUVW(corrMatrix, ants, obs, sbs,
                                     fDict['ts']-np.array(tDeltas),
                                     local_uv=local_uv)

    return vis, uvw, freqs, [obsLat, obsLong, LSTangle], nants

# TODO: add option to not apply rotation, useful for standard FT imaging
def readMS(fn, sbs, column='DATA'):
    """Return the visibilites and UVW coordinates from a Measurement Set
    fn: MS filename
    column: string, data column
    sbs: 1-D array of subband IDs

    returns:
        vis: visibilities [4, Nsamples, Nsubbands]
        uvw: UVW coordinates [Nsamples, 3, Nsubbands]
        freqs: frequencies [Nsubbands]
        obsdata: [latitude, longitude, LST]
    """
    try:
        import casacore.tables as tbls
    except ImportError:
        print('ERROR: could not import casacore.tables, cannot read measurement sets')
        exit(1)

    MS = tbls.table(fn, readonly=True)
    data_column = column.upper()
    uvw = MS.col('UVW').getcol() # [vis id, (u,v,w)]
    vis = MS.col(data_column).getcol() #[vis id, freq id, stokes id]
    vis = vis[:,sbs,:] #select subbands
    MS.close()

    # lat/long/lst information
    ANTS = tbls.table(fn + '/ANTENNA')
    positions = ANTS.col('POSITION').getcol()
    ant0Lat, ant0Long, ant0hgt = ecef.ecef2geodetic(positions[0,0], positions[0,1], positions[0,2], degrees=False) # use the first antenna in the table to get the array lat/long
    ANTS.close()
    SRC = tbls.table(fn + '/SOURCE')
    direction = SRC.col('DIRECTION').getcol()
    obsLat = direction[0,1]
    obsLong = ant0Long
    LSTangle = direction[0,0]
    SRC.close()

    # freq information, convert uvw coordinates
    SW = tbls.table(fn + '/SPECTRAL_WINDOW')
    freqs = SW.col('CHAN_FREQ').getcol()[0, sbs] # [nchan]
    print('SUBBANDS:', sbs, '(', freqs/1e6, 'MHz)')
    SW.close()

    # TODO: check rotation reference is the same as with LOFAR data, north pole is dec=+90, ra=0
    # in order to accommodate multiple observations at different times/sidereal times all the positions need to be rotated relative to sidereal time 0
    print('LST:',  LSTangle)
    rotAngle = float(LSTangle) - obsLong # adjust LST to that of the Observatory longitutude to make the LST that at Greenwich
    # to be honest, the next two lines change the LST to make the images come out but i haven't worked out the coordinate transforms, so for now these work without justification
    rotAngle += np.pi
    rotAngle *= -1
    # Rotation matrix for antenna positions
    rotMatrix = np.array([[np.cos(rotAngle), -1.*np.sin(rotAngle), 0.],
                          [np.sin(rotAngle), np.cos(rotAngle),     0.],
                          [0.,               0.,                   1.]]) #rotate about the z-axis
    uvwRot = np.dot(uvw, rotMatrix).reshape(uvw.shape[0], uvw.shape[1], 1)
    uvwRotRepeat = np.repeat(uvwRot, len(sbs), axis=2)

    return np.transpose(vis, (2,0,1)), uvwRotRepeat, freqs, [obsLat, obsLong, LSTangle]


def readNPZ(fn, sbs=[0], jump=1, local_uv=None):
    """
    Parameters
    ----------
    fn: str
        Filename of numpy zipped data
    sbs: list
        Subbands list.
    jump: int
        Skip 'jump' samples.
    local_uv: bool
        Use topocentric UV coord. sys. or not.

    Returns
    -------
    vis:
        Visibility array
    uvw:
        UVW baseline postions
    freqs:
        Frequencies
    obsLong:
        Observation longitude
    obsLat:
        Observation latitude
    LSTangle:
        Local sidereal angle
    nants:
        Number of antennas
    """
    # Load dataset
    ds = np.load(fn)
    ldattype = ds['datatype']

    # Create times
    sdt = datetime.datetime.utcfromtimestamp(
        (ds['start_datetime'] - np.datetime64('1970-01-01T00:00:00Z'))
        / np.timedelta64(1, 's'))
    delta_secs = ds['delta_secs']
    if ldattype == 'acc':
        delta_secs = delta_secs[:, sbs]
    delta_secs = delta_secs.flatten()
    ts = [sdt + datetime.timedelta(seconds=int(dt)) for dt in delta_secs]

    # Add subband axis:
    freq_dep = ds['frequencies']
    freq_all = np.sort(np.unique(freq_dep))
    freqs = freq_all[sbs]
    if not isinstance(freqs, np.ndarray):
        freqs = np.asarray(freqs)
    ts = np.asarray(ts)[np.newaxis]

    # Create visibility matrices
    arrfiles = [f for f in ds.files if f.startswith('arr_')]
    # Try to avoid loading in every file fully by skipping over jump samples
    # so compute a list indices to extract from all arr 'jumpsmps'
    # as if all the arrs were concatenated.
    if ldattype == 'xst':
        nrsmps_in_file = ds['arr_0'].shape[0]
        jumpsmps = np.asarray(range(nrsmps_in_file*len(arrfiles))[::jump])
        nrsmps = 0
    _corrmatarray = np.empty((0,)+ds['arr_0'].shape[1:])  # Empty arr to append2
    for arridx, arrfile in enumerate(arrfiles):
        if ldattype == 'acc':
            if arridx % jump:
                continue
        _arr = ds[arrfile]
        if ldattype == 'acc':
            _arr = _arr[sbs]
        else:
            # Jump over 'jump' samples taking into account previous file samples
            jumpsmps_in_file = jumpsmps[(jumpsmps >= nrsmps)
                                    & (jumpsmps < nrsmps+nrsmps_in_file)]-nrsmps
            _arr = _arr[jumpsmps_in_file]
            nrsmps += nrsmps_in_file
        _corrmatarray = np.append(_corrmatarray, _arr, axis=0)
    corshp = _corrmatarray.shape
    corrMatrix = np.zeros((len(freqs), corshp[0], corshp[1] * corshp[3],
                           corshp[2] * corshp[4]), complex)
    corrMatrix[0, :, 0::2, 0::2] = _corrmatarray[..., 0, 0, :, :]
    corrMatrix[0, :, 0::2, 1::2] = _corrmatarray[..., 0, 1, :, :]
    corrMatrix[0, :, 1::2, 0::2] = _corrmatarray[..., 1, 0, :, :]
    corrMatrix[0, :, 1::2, 1::2] = _corrmatarray[..., 1, 1, :, :]
    ants = ds['positions'][:, np.newaxis, :]
    vis, uvw, LSTangle = lofarGenUVW(corrMatrix, ants, None,
                                     sbs, ts[:, ::jump], local_uv=local_uv)
    obsLat, obsLong, h = ecef.ecef2geodetic(*np.mean(ants, axis=(0,1)))
    LSTangle = 0.
    nants = ants.shape[-1]
    return vis, uvw, freqs, obsLong, obsLat, LSTangle, nants


import SWHT

def read_lofvis(vis_files, station, ant_field, ant_array, deltas, subband, times,
            override, rcumode, int_time, calfile, column, local_uv=False):
    dataFmt = None
    if (not (station is None)) or (not (ant_field is None)): # If using LOFAR data, get station information
        lofarStation = SWHT.lofarConfig.getLofarStation(name=station,
                                affn=ant_field, aafn=ant_array, deltas=deltas)
        antGains = None # Setup variable so that the gain table isn't re-read for every file if used
        if lofarStation.name=='KAIRA': dataFmt='KAIRA'
    # parse subbands
    sbs = np.array(SWHT.util.convert_arg_range(subband))

    # setup variables for combined visibilities and uvw samples
    visComb = np.array([]).reshape(4, 0, len(sbs))
    uvwComb = np.array([]).reshape(0, 3, len(sbs))
    imgCoeffs = None

    for vid, visFn in enumerate(vis_files):
        print('Using %s (%i/%i)'%(visFn, vid+1, len(vis_files)))
        fDict = SWHT.fileio.parse(visFn, fmt=dataFmt)

        # Pull out the visibility data in a (u,v,w) format
        if fDict['fmt']=='acc': # LOFAR station all subbands ACC file visibilities
            fDict['rcu'] = rcumode # add the RCU mode to the meta data of an ACC file
            fDict['sb'] = sbs # select subbands to use
            fDict['int'] = int_time # set integration length (usually 1 second)

            vis, uvw, freqs, obsInfo, nants = SWHT.fileio.readACC(visFn, fDict,
                lofarStation, sbs, calTable=calfile, local_uv=local_uv)
            [obsLat, obsLong, LSTangle] = obsInfo

            # add visibilities to previously processed files
            visComb = np.concatenate((visComb, vis), axis=1)
            uvwComb = np.concatenate((uvwComb, uvw), axis=0)

        elif fDict['fmt']=='xst': # LOFAR XST format visibilities
            if override: # Override XST filename metadata
                fDict['rcu'] = rcumode
                fDict['sb'] = sbs
                fDict['int'] = int_time
            else:
                sbs = fDict['sb']

            vis, uvw, freqs, obsInfo, nants = SWHT.fileio.readXST(visFn, fDict,
                    lofarStation, sbs, calTable=calfile, local_uv=local_uv)
            [obsLat, obsLong, LSTangle] = obsInfo

            # add visibilities to previously processed files
            visComb = np.concatenate((visComb, vis), axis=1)
            uvwComb = np.concatenate((uvwComb, uvw), axis=0)

        elif fDict['fmt']=='KAIRA': # KAIRA LOFAR XST format visibilities
            if override: # Override XST filename metadata
                fDict['rcu'] = rcumode
                fDict['sb'] = sbs
                fDict['int'] = int_time
            else:
                sbs = fDict['sb']

            vis, uvw, freqs, obsInfo = SWHT.fileio.readKAIRAXST(visFn, fDict,
                        lofarStation, sbs, times=times, local_uv=local_uv)
            [obsLat, obsLong, LSTangle] = obsInfo

            # add visibilities to previously processed files
            visComb = np.concatenate((visComb, vis), axis=1)
            uvwComb = np.concatenate((uvwComb, uvw), axis=0)

        elif fDict['fmt']=='ms': # MS-based visibilities
            fDict['sb'] = sbs

            vis, uvw, freqs, obsInfo = SWHT.fileio.readMS(visFn, sbs, column=column)
            [obsLat, obsLong, LSTangle] = obsInfo

            # add visibilities to previously processed files
            visComb = np.concatenate((visComb, vis), axis=1)
            uvwComb = np.concatenate((uvwComb, uvw), axis=0)

        elif fDict['fmt']=='pkl':
            print('Loading Image Coefficients file:', visFn)
            coeffDict = SWHT.fileio.readCoeffPkl(visFn)
            imgCoeffs = coeffDict['coeffs']
            LSTangle = coeffDict['lst']
            obsLong = coeffDict['phs'][0]
            obsLat = coeffDict['phs'][1]

        elif fDict['fmt'] == 'npz':
            #vis, uvw, freqs, obsLong, obsLat, LSTangle, nants \
             #   = SWHT.fileio.readNPZ(visFn, calTable=calfile, local_uv=local_uv)
            # Defer till after returning
            freqs = 0.
            obsLong = 0.
            obsLat = 0.
            LSTangle = 0.
        else:
            raise ValueError('ERROR: unknown data format, exiting')

    return (fDict, visComb, uvwComb, freqs, obsLong, obsLat, LSTangle,
            imgCoeffs)


if __name__ == '__main__':
    print('Running test cases...')

    # LOFAR ACC
    fDict = parse('../examples/20150607_122433_acc_512x192x192.dat')
    print(fDict)

    # Measurement Set
    fDict = parse('../examples/zen.2455819.26623.uvcRREM.MS')
    print(fDict)

    # SE607 HBA XST
    fDict = parse('../examples/20150915_191137_rcu5_sb60_int10_dur10_elf0f39fe2034ea85fc02b3cc1544863053b328fd83291e880cd0bf3c3d3a50a164a3f3e0c070c73d073f4e43849c0e93b_xst.dat')
    print(fDict)

    # KAIRA XST
    fDict = parse('20160228_040005_xst.dat', fmt='KAIRA')
    print(fDict)

    print('...Made it through without errors')

