# Copyright 2023 Andreas Koehler, MIT license

from obspy.signal.util import next_pow_2
from obspy.signal.headers import clibsignal
from obspy.signal.invsim import cosine_taper
from obspy.core import Stream
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib  import cm
import scipy
from scipy import signal
import scipy.linalg      as linalg
import multitaper.utils      as utils 
import multitaper.mtspec     as spec

# Code for computing the covariances/cross-spectral matrix for all array data
# using FFT or muti-taper code from Prieto et al.
# I had to modify MTCross of the multitaper code slighlty to allow for pre-computation of tapers.
# This is not used: from multitaper import MTCross
# Modified MTCross function added below.


class CSM_methods(Enum):
    """
    Define the valid methods for the CSM.
    Can be one of:
    'CBF' - relative beam power
    'CCBF' - relative beam power in dB
    'CAPON' - absolute beam power
    """
    CBF = 'CBF'
    CCBF = 'CCBF'
    CAPON = 'CAPON'

class CSM_specmethods(Enum):
    """
    Define the valid spectral methods for the CSM.
    Can be one of:
    'FFT'
    'PRIETO' - Multi-taper method from Prieto
    """
    FFT = 'FFT'
    PRIETO = 'PRIETO'

def qc_stream(stream):
    """
    QC stream input data
     :param stream: Data stream containing traces
     :type  stream: ObspyStream
    """
    if not isinstance(stream, Stream):
        raise TypeError('Input stream is not a valid Stream')
    if len(stream) < 2:
        raise ValueError('Multiple traces are required')
    npts = set([tr.stats.npts for tr in stream])
    deltas = set([tr.stats.delta for tr in stream])
    if len(deltas) > 1:
        raise ValueError('Traces should have the same sample rate')
    if len(npts) > 1:
        raise ValueError('Traces should have the same number of points')
    if npts == 0:
        raise ValueError('No data in Trace')
    for tr in stream:
        if isinstance(tr.data, np.ma.masked_array):
            raise ValueError('At least one trace has masked data')
        eps = 1e-16
        if np.all(np.abs(tr.data) < eps) :
            raise ValueError('At least one trace has only zero data')
        data = tr.data - tr.data[0]
        if np.all(np.abs(data) < eps) :
            raise ValueError('At least one trace has constant data')


def _cross_spectral_matrix(stream, nfreq, nlow, nf, method, coarflag):
    """Returns the cross-spectral matrix for a given input stream using numpy rfft
            Function should not be called directly. Use class CrossSpectralMatrix() instead.
            This class calls this function and computes input parameters accordingly.
     :param stream: Obspy data stream containing traces of an array (should already be filtered)
     :type  stream: Stream
     :param nfreq: total number of frequency points to be used for spectrum calculation
     :type nfreq: int
     :param nlow: index of lowest frequency point in cross-spectral matrix
     :type nlow: int
     :param nf: number of frequency points in cross-spectral matrix
     :type nf: int
     :param method: prepare cross-spectral matrix for method of Capon et al ('CAPON'),
      normal FK ('CBF'), or cross-correlation beamforming ('CCBF')
     :type method: str
     :param coarflag: co-array flags in nchan x nchan array (0 = use, 1 = do not use)
     :type coarflag: np.ndarray
     :return: cross-spectral matrix and scaling factor needed for calculating the semblance
     :rtype: np.ndarray, float

    """
    # computing the covariances/cross-spectral matrix between all receivers: adapted from obspy

    if method == 'CCBF' and coarflag is None:
        raise ValueError('Co-array flags cannot be None for CCBF method!')

    nchan = len(stream)

    # 0.22 matches 0.2 of historical C bbfk.c
    tap = cosine_taper(stream[0].stats.npts, p=0.22)

    # compute spectra
    ft = np.empty((nchan, nf), dtype=np.complex128)
    for i, tr in enumerate(stream):
        dat = tr.data
        dat = (dat - dat.mean()) * tap
        ft[i, :] = np.fft.rfft(dat, nfreq)[nlow:nlow + nf]
        ft[i, :] /= np.sqrt(nfreq)
    ft = np.ascontiguousarray(ft, np.complex128)

    dpow = 0.
    _r = np.empty((nf, nchan, nchan), dtype=np.complex128)
    for i in range(nchan):
        for j in range(i, nchan):
            _r[:, i, j] = ft[i, :] * ft[j, :].conj()
            if method == 'CAPON':
                _r[:, i, j] /= np.abs(_r[:, i, j].sum())
            if i != j:
                _r[:, j, i] = _r[:, i, j].conjugate()
            else:
                dpow += np.abs(_r[:, i, j].sum())
            # set to zero for CCBF. Still need dpow in case of relpower normalization.
            if method == 'CCBF':
                if coarflag[i, j] == 1:
                    _r[:, j, i] = np.zeros(nf, dtype=np.complex128)
                    _r[:, i, j] = np.zeros(nf, dtype=np.complex128)
    dpow *= nchan
    if method == 'CAPON':
        # P(f) = 1/(e.H R(f)^-1 e) = Capon
        for n in range(nf):
            _r[n, :, :] = np.linalg.pinv(_r[n, :, :], rcond=1e-6)
    return _r, dpow


def _cross_spectral_matrix_prieto(stream, nfreq, nlow, nf, method, coarflag, kspec = 5, tbp = 3.5, vn = None, lamb = None):
    """Returns the cross-spectral matrix for a given input stream using
        method of Prieto et al (multi-taper)
       NEW: function should not be called directly. Use class CrossSpectralMatrix() instead.
            This class calls this function and computes input parameters accordingly.

     :param stream: data stream containing traces of an array (should already be filtered)
     :type  stream: Obspytream
     :param nfreq: total number of frequency points to be used for spectrum calculation
     :type nfreq: int
     :param nlow: index of lowest frequency point to be included in cross-spectral matrix
     :type nlow: int
     :param nf: number of frequency points to be included in cross-spectral matrix
     :type nf: int
     :param method: prepare cross-spectral matrix for method of Capon et al ('CAPON'),
      normal FK ('CBF'), or cross-correlation beamforming ('CCBF')
     :type method: str
     :param coarflag: co-array flags in nchan x nchan array (0 = use, 1 = do not use)
     :type coarflag: np.darray
     :param kspec : number of taper for multitaper method
     :type kspec : int
     :param tbp : time-bandwidth product for multitaper method
     :type tbp : float
     :param vn : DPSS tapers and eigenvalues: Slepian sequences -> possible to pre-compute to speed up processing
     :type vn : ndarray [npts,kspec]
     :param lamb : Eigenvalues of Slepian sequences -> possible to pre-compute to speed up processing
     :type lamb : ndarray [kspec]
     :return: cross-spectral matrix and scaling factor needed for calculating the semblance
     :rtype: np.ndarray, float

    """

    if method == 'CCBF' and coarflag is None:
        raise ValueError('Co-array flags cannot be None for CCBF method!')

    nchan = len(stream)
    _r = np.empty((nf, nchan, nchan), dtype=np.complex128)
    dpow = 0.
    tap = cosine_taper(stream[0].stats.npts, p=0.22)
    for i in range(nchan):
        data1 = stream[i].data
        data1 = (data1 - data1.mean()) * tap
        for j in range(i, nchan):
            data2 = stream[j].data
            data2 = (data2 - data2.mean()) * tap
            out  = MTCross(
                data1,
                data2,
                nw=tbp,
                kspec=kspec,
                dt=1./stream[0].stats.sampling_rate,
                nfft=nfreq,
                iadapt=1,
                vn=vn,
                lamb=lamb
            )
            # check Parseval's theorem
            # print(np.sum(data1**2)/2.)
            # print(np.sum(out.Sxx))
            _r[:, i, j] = out.Sxy[nlow:nlow + nf].flatten()
            if method == 'CAPON':
                _r[:, i, j] /= np.abs(_r[:, i, j].sum())
            if i != j:
                _r[:, j, i] = _r[:, i, j].conjugate()
            else:
                dpow += np.abs(_r[:, i, j].sum())
            if method == 'CCBF':
                if coarflag[i, j] == 1:
                    _r[:, j, i] = np.zeros(nf, dtype=np.complex128)
                    _r[:, i, j] = np.zeros(nf, dtype=np.complex128)
    dpow *= nchan
    if method == 'CAPON':
        # P(f) = 1/(e.H R(f)^-1 e) = Capon
        for f in range(nf):
            _r[f, :, :] = np.linalg.pinv(_r[f, :, :], rcond=1e-6)

    return _r, dpow


class CrossSpectralMatrix():
    """
    Represents a spatial covariance or cross-sepctral matrix (CSM) of a seismic array record

    Contains methods and attributes applicable to the CSM

    """

    def __init__(self, stream, method = 'CBF', spec_method = "FFT", compute = True, flow = None, fhigh = None,
                 precompute_multitaper = True, kspec = 5, tbp = 3.5 ):
        """
        Initialize CSM
        :param stream: Array stream for which the CSM should be computed for all frequencies
        :type stream: ObspyStream
        :param method: the method to use 'CBF' (correlation beamforming), 'CCBF' (cross-correlation beamforming without
        autocorrelations ), 'CAPON' (Method of Capon et al.)
        :type method: str
        :param spec_method : which method should be used for spectrum computation FFT ("FFT")
                             or multi-taper method from Prieto "PRIETO"
        :type spec_method: str
        :param compute: specifies whether to compute the CSM upon initialisation. If false, only data preparation is
        done. CSM computation can be called using the "compute_CSM" method from this class.
        :type compute: bool
        :param flow: lowest frequency band for which a CSM should be computed
                     (closest frequency sampling point will be chosen)
        :type flow: float
        :param fhigh: highest frequency band for which a CSM should be computed
                      (closest frequency sampling point will be chosen)
        :type fhigh: float
        :param precompute_multitaper : precompute multitaper to speed up processing for Prieto method
                                       if False, matrix computation takes ages, but there migth be a reason to keep this option
        :type precompute_multitaper : bool
        :param kspec : number of taper for multitaper method of Prieto
        :type kspec : int
        :param tbp : time-bandwidth product for multitaper method of Prieto
        :type tbp : float
        """
        qc_stream(stream)

        try:
            CSM_methods(method)
        except ValueError:
            raise NotImplementedError("%s is an unsupported method" % method)

        try:
            CSM_specmethods(spec_method)
        except ValueError:
            raise NotImplementedError("%s is an unsupported spectral method" % spec_method)


        if flow is not None and fhigh is not None and flow >= fhigh :
            raise ValueError("flow must be lower than fhigh")
        self.nchan = len(stream)
        self.fs = stream[0].stats.sampling_rate
        self.npts = stream[0].stats.npts
        self.flow = flow
        self.fhigh = fhigh
        if flow is None : self.nlow = int(1)  # avoid using the offset as default (nlow=1)
        if spec_method == "FFT":
            self._prep_for_fft(stream)
        elif spec_method == "PRIETO":
            self._prep_for_prieto(stream, precompute_multitaper = precompute_multitaper, kspec = kspec, tbp = tbp)
        else:
            raise ValueError("%s is an unsupported method to compute the cross-spectral matrix" % spec_method)

        if compute == True: self.compute_CSM(stream, method)
        self.chanlist=[tr.stats.station for tr in stream]
        self.ccbf = False
        self.dist_limit_in_km = None
        self.capon = False

    def _prep_for_fft(self,stream):
        """
        Private method for preparing for the CSM computation using FFT spectral method
        :param stream: Array stream for which the CSM should be computed for all frequencies
        :type stream: ObspyStream
        """
        self.nfreq = next_pow_2(self.npts)
        flist=np.array(range(self.nfreq))*self.fs/self.nfreq
        if self.flow is not None : self.nlow = np.argmin(np.abs(flist-self.flow))
        if self.fhigh is not None : self.nhigh = np.argmin(np.abs(flist-self.fhigh))
        else : self.nhigh = int(self.nfreq // 2 - 1) # nyquist , highest frequency for rfft
        self.flow = flist[self.nlow]
        self.fhigh = flist[self.nhigh] # update input frequency to closest sampling point
        self.flist=flist[self.nlow:self.nhigh+1] # update input frequency to closest sampling point
        self.nf = self.nhigh - self.nlow + 1  # include upper and lower frequency
        self.spec_method = "FFT"
        self.deltaf = self.fs / float(self.nfreq)
 
    def _compute_from_fft(self, stream):
        """
        Private method for CSM computation with FFT spectral method
        :param stream: Array stream for which the CSM should be computed for all frequencies
        :type stream: ObspyStream
        """
        # Check that the method or the data hasn't changed from the preparation
        if self.spec_method != "FFT" or self.nfreq != next_pow_2(self.npts): self._prep_for_fft(stream)
            
        # 'coarflag' and 'method' options are no longer used or fixed to 'CBF'!
        # This is now done with a class method outside this function
        self.data,self.dpow = _cross_spectral_matrix(stream, self.nfreq, self.nlow, self.nf, method='CBF', coarflag=None)


    def _prep_for_prieto(self, stream, next_pow = False, precompute_multitaper = True, kspec = 5, tbp = 3.5):
        """
        Private method for CSM computation with multi-taper method
        :param stream: Array stream for which the CSM should be computed for all frequencies
        :type stream: ObspyStream
        :param next_pow: True for using more frequency sampling points for spectral estimation (next power of two)
        :type next_pow : bool
        :param precompute_multitaper : precompute multitaper to speed up processing
        :type precompute_multitaper : bool
        :param kspec : number of taper for multitaper method
        :type kspec : int
        :param tbp : time-bandwidth product for multitaper method
        :type tbp : float
        """
        if next_pow is None: self.nfreq = self.npts
        else : self.nfreq = next_pow_2(self.npts)
        flist=np.array(range(self.nfreq))*self.fs/self.nfreq
        if self.flow is not None : self.nlow = np.argmin(np.abs(flist-self.flow))
        if self.fhigh is not None : self.nhigh = np.argmin(np.abs(flist-self.fhigh))
        else : self.nhigh = int(self.nfreq // 2 - 1)
        self.flow = flist[self.nlow] # update input frequency to closest sampling point
        self.fhigh = flist[self.nhigh] # update input frequency to closest sampling point
        self.flist=flist[self.nlow:self.nhigh+1]
        self.nf = self.nhigh - self.nlow + 1  # include upper and lower frequency
        self.spec_method = "PRIETO"
        self.deltaf = self.fs / float(self.nfreq)
        self.kspec = kspec
        self.tbp = tbp
        if precompute_multitaper :
            self.vn, self.lamb = utils.dpss(self.npts,self.tbp,self.kspec)
        else :
            self.vn = None
            self.lamb = None

    def _compute_from_prieto(self, stream):
        """
        Private method for CSM computation with multi-taper method
        :param stream: Array stream for which the CSM should be computed for all frequencies
        :type stream: ObspyStream
        """
        # Check that the method or the data hasn't changed from the preparation
        if self.spec_method != "PRIETO" or self.npts != stream[0].stats.npts : self._prep_for_prieto(stream)
          
        # 'coarflag' and 'method' options are no longer used or fixed to 'CBF'!
        # This is now done with a class method outside this function
        self.data,self.dpow = _cross_spectral_matrix_prieto(stream, self.nfreq, self.nlow, self.nf, method='CBF', coarflag=None,
                                                            kspec=self.kspec, tbp=self.tbp, vn=self.vn, lamb=self.lamb)
        
    def compute_CSM(self, stream, method = 'CBF'):
        """
        Public method to computing the CSM for either CBF, Capon, CCBF
        :param stream: Array stream for which the CSM should be computed using the parameters set in the
        CrossSpectralMatrix class
        :type stream: ObspyStream
        :param method: the method to use 'CBF' (correlation beamforming), 'CCBF' (cross-correlation beamforming without
        autocorrelations), 'CAPON' (Method of Capon et al.)
        :type method: str
        """

        try:
            CSM_methods(method)
        except ValueError:
            raise NotImplementedError("%s is an unsupported method" % method)

        # Run the appropriate computation, depending on the spectral method that was specified during the class
        # initialisation
        if self.spec_method == "PRIETO": 
            self._compute_from_prieto(stream)
        elif self.spec_method == "FFT":
            self._compute_from_fft(stream)

        # in case of CCBF autocorrelations are set to zero
        # (see Gibbons et al, 2018, GRL, Ruigrok et al. 2016, JSeis)
        if method == 'CCBF':
            self.set_autocorrelation_to_zero()
        elif method == 'CAPON':
            self.prepare_capon()

    def set_autocorrelation_to_zero(self):
        """
        Public method to prepare CSM for cross-correlation beamforming (CCBF)
        which mean we set the auto-correlations (matrix diagonals) to zero
        """
        for i in range(self.nchan) : self.data[:,i,i] = np.zeros(self.nf, dtype=np.complex128)
        self.ccbf = True


    def select_single_frequency(self,f_ind):
        """
        Public method to extract the CSM at a single frequency
        :param f_ind: index of frequency (check self.flist to find wished index)
        :type f_ind: int
        :return: CSM data
        :rtype: numpy array
        """
        return self.data[f_ind]

    def get_dpow_fband(self,f_ind_1,f_ind_2):
        """
        Public method to get sum of autocorrelations (power) for a frequency band of interest
        Useful when CSM is computed once and relativ beampower should be computed for different frequency bands
        :param f_ind_1: index of lower frequency (check self.flist to find wished index)
        :type f_ind_1: int
        :param f_ind_2: index of upper frequency (check self.flist to find wished index)
        :type f_ind_2: int
        :return: Power in frequency band
        :rtype: float
        """
        dpow = 0
        for i in range(self.nchan): dpow += np.abs(self.data[f_ind_1:f_ind_2+1, i, i].sum())
        dpow *= self.nchan
        return dpow

    def principal_eigen_vectors(self):
        """
        Public method to compute first principal Eigenvector of CSM for all frequencies
        (used in empirical matched field processing)
        :return: principal Eigenvectors for all frequencies
        :rtype: numpy array
        """
        pev_all=[]
        for find in range(self.nf):
            _, v = np.linalg.eigh(self.data[find])
            pev=v[:,-1]
            pev_all.append(pev)
        return np.array(pev_all)

    def csm_coarray_pattern(self,coarray,f_ind,plot=False):
        """Returns coarray pattern for cross-spectral matrix for a chosen frequency
         :param coarray: Dictionary containing the coarray
         :type coarray: dict
         :param f_ind: frequency index in csm
         :type f_ind :  int
         :param plot : True or False for plotting csm coarray pattern
         :return : coarray dictionary where keys 'station1-station2' contain distance in km, azimuth in radians, phase and coherency
         :rtype : dict
        """

        coarray_pattern={}
        for key in coarray:
            if key.split('-')[0] in self.chanlist and key.split('-')[1] in self.chanlist :
                i=self.chanlist.index(key.split('-')[0])
                j=self.chanlist.index(key.split('-')[1])
                coarray_pattern[key] = {'distance' : coarray[key]['distance'],
                                        'azimuth' : coarray[key]['azimuth'],
                                        'phase' : np.angle(self.data[f_ind][i][j]),
                                        'coherency' : np.abs(self.data[f_ind][i][j])*np.abs(self.data[f_ind][i][j]) /
                                                     (np.abs(self.data[f_ind][i][i])*np.abs(self.data[f_ind][j][j]))}
        if plot :
            fig = plt.figure()
            ax = fig.add_subplot(projection='polar')
            sc=ax.scatter([coarray_pattern[key]['azimuth'] for key in coarray_pattern], \
                          [coarray_pattern[key]['distance'] for key in coarray_pattern], \
                          marker='o',c=[coarray_pattern[key]['phase'] for key in coarray_pattern], \
                          cmap = cm.seismic,edgecolors='k',vmin=-3.14,vmax=3.14, \
                          s=[coarray_pattern[key]['coherency']*50 for key in coarray_pattern])
            plt.colorbar(sc);
            plt.show()
        return coarray_pattern

def _angle_between(p1, p2):
    ang=np.arctan2(*(p2-p1)[::-1])
    return ang % (2 * np.pi)

def get_coarray(geometry, plot=False, full=True):
    """Returns coarray of a seismic array
     :param geometry: Dictionary containing the dx and dy offsets from the array reference stations
     :type geometry: dict
     :param plot : True or False for plotting coarray geometry
     :type: bool
     :param full: False if station pair should appear only once not twice.
     :type: bool
     :return : coarray dictionary where keys 'station1-station2' contain distance in km and azimuth in radians
     :rtype : dict
    """

    keys=[key for key in geometry]
    coarray={}
    for i in range(len(keys)):
        loc1=np.array([geometry[keys[i]]['dx'],geometry[keys[i]]['dy']])
        if full : lst = range(len(keys))
        else : lst = range(i+1,len(keys))
        for j in lst:
            if i != j :
                loc2=np.array([geometry[keys[j]]['dx'],geometry[keys[j]]['dy']])
                coarray[keys[i]+'-'+keys[j]] = {'distance' : np.linalg.norm(loc1-loc2),'azimuth' : _angle_between(loc1,loc2)}
    if plot :
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        ax.scatter([coarray[key]['azimuth'] for key in coarray],[coarray[key]['distance'] for key in coarray],marker='o')
        plt.show()
    return coarray

# Copyright 2022 Germán A. Prieto, MIT license
# Modified by Andreas Koehler
#-----------------------------------------------------

class MTCross:

    """

    A class for bi-variate Thomson multitaper estimates. 
    It performs main steps in bi-variate multitaper estimation, 
    including cross-spectrum, coherency and transfer function.

    **Attributes**

    *Parameters*

    npts   : int
        number of points of time series
    nfft   : int
        number of points of FFT. Dafault adds padding. 
    nw     : flaot
        time-bandwidth product
    kspec  : int
        number of tapers to use

    *Time series*

    x      : ndarray [npts]
        time series
    xvar   : float
        variance of time series
    dt     : float
        sampling interval

    *Frequency vector*

    nf     : int
        number of unique frequency points of spectral 
        estimate, assuming real time series
    freq   : ndarray [nfft]
        frequency vector in Hz
    df     : float
        frequncy sampling interval

    *Method*

    iadapt : int
        defines methos to use
        0 - adaptive multitaper
        1 - unweighted, wt =1 for all tapers
        2 - wt by the eigenvalue of DPSS
    wl : float, optional
        water-level for stabilizing deconvolution (transfer function).
        defined as proportion of mean power of Syy

    *Spectral estimates*

    Sxx : ndarray [nfft]
        Power spectrum of x time series
    Syy : ndarray [nfft]
        Power spectrum of y time series
    Sxy : ndarray, complex [nfft]
        Coss-spectrum of x, y series
    cohe  : ndarray [nfft]
        MSC, freq coherence. Normalized (0.0,1.0)
    phase : ndarray [nfft]
        the phase of the cross-spectrum    
    cohy : ndarray, complex [nfft]
        the complex coherency, normalized cross-spectrum 
    trf  : ndarray, compolex [nfft]
        the transfer function Sxy/(Syy_wl), with water-level optional
    se : ndarray [nfft,1] 
        degrees of freedom of estimate
    wt : ndarray [nfft,kspec]
        weights for each eigencoefficient at each frequency

    **Methods**

       * init      : Constructor of the MTCross class
       * mt_deconv : Perform the deconvolution from the self.trf, by iFFT
       * mt_corr   : compute time-domain via iFFT of cross-spectrum, 
                     coherency, and transfer function

    **Modified**
    
	German Prieto
	January 2022

    |

    """

    def __init__(self,x,y,nw=4,kspec=0,dt=1.0,nfft=0,iadapt=0,wl=0.0,vn=None,lamb=None):
        """
        The constructor of the MTCross class.

        It performs main steps in bi-variate multitaper estimation, 
        including cross-spectrum, coherency and transfer function.
        
        MTCross class variable with attributes described above. 

        **Parameters**
        
        x : MTSpec class, or ndarray [npts,]
            Time series signal x.
            If ndarray, the MTSpec class is created.
        y : MTSpec class, or ndarray [npts,]
            Time series signal x
            If ndarray, the MTSpec class is created.
        nw : float, optional
            time bandwidth product, default = 4
            Only needed if x,y are ndarray
        kspec : int, optional
            number of tapers, default = 2*nw-1
            Only needed if x,y are ndarray
        dt : float, optional
            sampling interval of x, default = 1.0
            Only needed if x,y are ndarray
        nfft : int, optional
            number of frequency points for FFT, allowing for padding
            default = 2*npts+1
            Only needed if x,y are ndarray
        iadapt : int, optional
            defines methos to use, default = 0
            0 - adaptive multitaper
            1 - unweighted, wt =1 for all tapers
            2 - wt by the eigenvalue of DPSS
        wl : float, optional
            water-level for stabilizing deconvolution (transfer function).
            defined as proportion of mean power of Syy

        |

        """
        
        #-----------------------------------------------------
        # Check if input data is MTSPEC class
        #-----------------------------------------------------

        if (type(x) is not type(y)):
            raise ValueError("X and Y are not similar types")

        if (type(x) is np.ndarray):
            
            #-----------------------------------------------------
            # Check dimensions of input vectors
            #-----------------------------------------------------

            xdim  = x.ndim
            ydim  = y.ndim
            if (xdim>2 or ydim>2):
                raise ValueError("Arrays cannot by 3D")
            if (xdim==1):
                x = x[:, np.newaxis]
            if (ydim==1):
                y = y[:, np.newaxis]
            if (x.shape[0] != y.shape[0]):
                raise ValueError('Size of arrays must be the same')
            ndim = x.ndim 
            nx   = x.shape[1]
            ny   = y.shape[1]
            npts = x.shape[0]
            if (nx>1 or ny>1):
                raise ValueError("Arrays must be a single column")

            x = spec.MTSpec(x,nw,kspec,dt,nfft,iadapt=iadapt,vn=vn,lamb=lamb)
            y = spec.MTSpec(y,nw,kspec,dt,nfft,iadapt=iadapt,vn=x.vn,lamb=x.lamb)


        #------------------------------------------------------------
        # Now, check MTSPEC variables have same sizes
        #------------------------------------------------------------

        if (x.npts != y.npts):
            raise ValueError("npts must coincide")
        if (x.dt != y.dt):
            raise ValueError("dt must coincide")
        if (x.nfft != y.nfft):
            raise ValueError("nfft must coincide")
        if (x.nw != y.nw):
            raise ValueError("NW must coincide")
        if (x.kspec != y.kspec):
            raise ValueError("KSPEC must coincide")

        #------------------------------------------------------------
        # Parameters based on MTSPEC class, not on input
        #------------------------------------------------------------
 
        iadapt = x.iadapt
        dt     = x.dt
        kspec  = x.kspec
        nfft   = x.nfft
        npts   = x.npts
        nw     = x.nw

        #------------------------------------------------------------
        # Create the cross and auto spectra
        #------------------------------------------------------------

        wt = np.minimum(x.wt,y.wt)
        se = utils.wt2dof(wt)

        wt_scale = np.sum(np.abs(wt)**2, axis=1)  # Scale weights to keep power 
        for k in range(kspec):
            wt[:,k] = wt[:,k]/np.sqrt(wt_scale)

        # Weighted Yk's
        dyk_x = np.zeros((nfft,kspec),dtype=complex)
        dyk_y = np.zeros((nfft,kspec),dtype=complex)
        for k in range(kspec):
            dyk_x[:,k] = wt[:,k] * x.yk[:,k]
            dyk_y[:,k] = wt[:,k] * y.yk[:,k]

        # Auto and Cross spectrum
        Sxy      = np.zeros((nfft,1),dtype=complex)
        Sxx      = np.zeros((nfft,1),dtype=float)
        Syy      = np.zeros((nfft,1),dtype=float)
        Sxx[:,0] = np.sum(np.abs(dyk_x)**2, axis=1) 
        Syy[:,0] = np.sum(np.abs(dyk_y)**2, axis=1) 
        Sxy[:,0] = np.sum(dyk_x * np.conjugate(dyk_y),axis=1)

        # Get coherence and phase
        cohe  = np.zeros((nfft,1),dtype=float)
        cohy  = np.zeros((nfft,1),dtype=complex)
        trf   = np.zeros((nfft,1),dtype=complex)
        phase = np.zeros((nfft,1),dtype=float)
        
        w_lev = wl*np.mean(Syy[:,0])
        phase[0] = np.arctan2(np.imag(Sxy[0]),np.real(Sxy[0])) 
        cohe[0]  = np.abs(Sxy[0])**2 / (Sxx[0]*Syy[0])
        cohy[0]  = Sxy[0] / np.sqrt(Sxx[0]*Syy[0])
        trf[0]   = Sxy[0] / (Syy[0]+w_lev)


        phase = phase * (180.0/np.pi)

        #-----------------------------------------------------------------
        # Save all variables in self
        #-----------------------------------------------------------------

        self.freq   = x.freq
        self.dt     = dt
        self.df     = x.df
        self.nf     = x.nf
        self.nw     = nw
        self.kspec  = kspec
        self.nfft   = nfft
        self.npts   = npts
        self.iadapt = iadapt

        self.Sxx    = Sxx
        self.Syy    = Syy
        self.Sxy    = Sxy
        self.cohe   = cohe
        self.cohy   = cohy
        self.trf    = trf
        self.phase  = phase
        self.se     = se
        self.wt     = wt

        del Sxx, Syy, Sxy, cohe, phase, se, wt

    #-------------------------------------------------------------------------
    # Finished INIT mvspec
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    # Deconvolution
    # TF = Sx/Sy 
    #    although actually we compute Sx*conj(Sy)/(Sy^2)
    # Take the IFFT to convert to the time domain. 
    # Assumes a real deconvolved signal (real input signals). 
    #-------------------------------------------------------------------------

    def mt_deconv(self): 

        """
        Generate a deconvolution between two time series, returning
        the time-domain signal.
        
        MTCross has already pre-computed the cross-spectrum and 
        the transfer function. 

        **Returns**
        
        dfun : ndarray [nfft]
            time domain of the transfer function. 
            delay time t=0 in centered in the middle.

        **References**
        
        The code more or less follows the paper
        Receiver Functions from multiple-taper spectral corre-
        lation estimates. J. Park and V. Levin., BSSA 90#6 1507-1520

        It also uses the code based on dual frequency I created in
        GA Prieto, Vernon, FL , Masters, G, and Thomson, DJ (2005), 
        Multitaper Wigner-Ville Spectrum for Detecting Dispersive 
        Signals from Earthquake Records, Proceedings of the 
        Thirty-Ninth Asilomar Conference on Signals, Systems, and 
        Computers, Pacific Grove, CA., pp 938-941. 

        | 

        """

        nfft  = self.nfft
        trf   = self.trf

        dfun  = scipy.fft.ifft(trf[:,0],nfft) 
        dfun  = np.real(scipy.fft.fftshift(dfun))
        dfun  = dfun[:,np.nexaxis]
        dfun  = dfun/float(nfft) 

        return dfun 


    def mt_corr(self): 

        """
        Compute time-domain via iFFT of cross-spectrum, 
        coherency, and transfer function
 
        Cross spectrum, coherency and transfer function 
        already pre-computed in MTCross.

        **Returns**
        
        xcorr : ndarray [nfft]
            time domain of the transfer function. 
        dcohy : ndarray [nfft]
            time domain of the transfer function. 
        dfun : ndarray [nfft]
            time domain of the transfer function. 
            
        Delay time t=0 in centered in the middle.

        **Notes**
        
        The three correlation-based estimates in the time domain
            - correlation (cross-spectrum)
            - deconvolution (transfer function)
            - norm correlation (coherency)
        Correlation:
            - Sxy = Sx*conj(Sy)
        Deconvolution:
            - Sxy/Sy = Sx*conj(Sy)/Sy^2
        Coherency
            - Sxy/sqrt(Sx*Sy)
        
        | 

        """

        nfft = self.nfft
        cohy = self.cohy
        trf  = self.trf
        xc   = self.Sxy

        xcorr  = scipy.fft.ifft(xc[:,0],nfft) 
        xcorr  = np.real(scipy.fft.fftshift(xcorr))
        xcorr  = xcorr[:,np.newaxis]
        xcorr  = xcorr/float(nfft) 

        dcohy  = scipy.fft.ifft(cohy[:,0],nfft) 
        dcohy  = np.real(scipy.fft.fftshift(dcohy))
        dcohy  = dcohy[:,np.newaxis]
        dcohy  = dcohy/float(nfft) 

        dconv  = scipy.fft.ifft(trf[:,0],nfft) 
        dconv  = np.real(scipy.fft.fftshift(dconv))
        dconv  = dconv[:,np.newaxis]
        dconv  = dconv/float(nfft) 

        return xcorr, dcohy, dconv
