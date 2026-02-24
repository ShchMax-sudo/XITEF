import os
import tarfile
import shutil
from math import floor, cos, sin, pi, ceil, log, sqrt, radians, exp
from time import time
from random import choice, random
import sys
import traceback
from scipy.stats import poisson
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astroquery.esa.xmm_newton import XMMNewton
from FenwickTree import FenwickTree

sys.path.append("sifting")

import sifting

np.seterr(divide = 'ignore')

class Config:
    def __init__(self):
        self.firstLayerCheckTimeInterval = 10                                                                           # Time interval in which photons count for the first layer check
        self.goodPhotonsCountThreshold = 4                                                                              # The amount of photons needed to be in 10 seconds interval to count the current one as a “probably good”
        self.probabilityThreshold = 10                                                                                  # pGlob of a Poisson probability threshold for a photon to count as “good”
        self.transientCountThreshold = 7                                                                                # Minimum amount of photons in a transient candidate
        self.flagFilter = 12                                                                                            # An upper threshold for the quality flag

        self.pglobCoefficient = 1                                                                                       # Recalculation coefficient from local to global probabilities (pglob = survivalFunction * pglobCoefficient / timeInterval)

        self.transientDuration = 10                                                                                     # Expectied transient duration
        self.timeThreshold = 10                                                                                         # Time interval used for transient composition (needed to be same order as expected transient duration)

        self.innerRingScale = 3                                                                                         # How many times background inner ring radius is larger than transient circle radius
        self.outerRingScale = 5                                                                                         # How many times background outer ring radius is larger than transient circle radius
        self.timeScale = 1                                                                                              # How many times background checking duration is larger than transient duration

        self.transientBackgroundUncertainty = 10  # 1 / uncertainty of a background during pglob calculations
        self.pglobThresh = 0.05                                                                                         # Max global probability of transient for it to be saved

        # BTI calculation
        self.hed = (10, 12)                                                                                             # High-energy diapason
        self.BTIthreshold = 0.4                                                                                         # Counts per second
        self.binDur = 100                                                                                               # Duration of one bin

        self.addFakeTransient = False                                                                                   # If it is needed to add fake transient
        self.ftransientTime = 10                                                                                        # Duration of a fake transient
        self.ftransientProb = 0.05                                                                                      # Global probability of a fake transient
        self.showDebug = False                                                                                          # If it is needed to show debug plots and prints
        self.printDebug = False                                                                                         # If it is needed to print program run information in file
        self.showVision = False                                                                                         # If it is needed to run Vision after image processing
        self.showTransients = False                                                                                     # If it is needed to show plots for each transient
        self.removeStars = True                                                                                         # If it is needed to remove stars at the first stage of processing
        self.removeBTI = False                                                                                          # If it is needed to remove photons within BTIs.


class TransientFinder:
    # There are 3 coordinate systems: RA-DEC (FK5), Local and Picture
    # RA-DEC – right ascension, declination coordinates given in degrees in FK5 system
    # Local – local for pn-camera x, y coordinates
    # Picture – coordinates on resulting picture
    # RA-DEC is used outside the processing (for input and output)
    # Picture uses only to build “picture” of whole observation (is unusable in processing too)
    # Local is the main coordinate system using in a self program
    # If it is not critical to use other coordinate systems, self should be preferred

    # Program logic:
    # 1. Get detections – download current observation files (PIEVLI + OBSMLI)
    # 2. Initiation of WCS – calculating all necessary parameters for coordinates transformation from headers
    # 3. Events parsing – parsing all necessary photon data from datasheet
    # 4. Removal of stars – all photons near to the detected “stars” in OBSMLI (in PSF area) are being removed
    #    This not for time optimisation purpose, but for removing potential photons with inhomogeneous background
    # 5. Bad events removal – all photons with non-zero quality flag or with out-of-picture coordinates are being removed
    # 6. Calculation of the mean background during BTI and GTI
    # 7. Adding fake transient (if addFakeTransient is True)
    # 8. Calculating the number of photons within the 10-second interval (for each photon)
    # 9. Calculating the number of photons during the entire observation (for each photon)
    # 10. pGlob calculation for each photon
    # 11. Filtering out probable photons using pGlob
    # 12. Constructing transients from improbable photons list
    # 13. Found transients pGlob calculation
    # 14. Attempt to analyse what was found

    # Overall program comments:
    # When determining time characteristic of transients (or other similar thing) “mean” means mean and “d” means dispersion, not duration (half-transient duration)

    distanceCheckGeneral = (lambda e1, e2, psf: abs(e1[0] - e2[0]) <= psf and
                                                abs(e1[1] - e2[1]) <= psf)                                              # Function for taxicab distance checking

    # Class for calculations related to CCD or CCD coordinates
    class CCD:
        GTI = None
        xyToCCDcoeff = None
        num = 0
        GTIevents = []
        BTIevents = []

        def __init__(self, num):
            self.num = num
            self.GTIevents = []
            self.BTIevents = []

        # Calculates necessary coefficients for CCD coordinates transform
        def initCoords(self):
            if len(self.GTIevents) == 0 and len(self.BTIevents) == 0:
                return
            if len(self.GTIevents) < 3:
                raise Exception("Not enough events in CCD #" + str(self.num))
            event0, event1, event2 = self.GTIevents[:3]
            matrix = np.array([
                [event0[0], event0[1], 1, 0, 0, 0],
                [0, 0, 0, event0[0], event0[1], 1],
                [event1[0], event1[1], 1, 0, 0, 0],
                [0, 0, 0, event1[0], event1[1], 1],
                [event2[0], event2[1], 1, 0, 0, 0],
                [0, 0, 0, event2[0], event2[1], 1],
            ])
            coords = np.array([event0[6], event0[7], event1[6], event1[7],
                               event2[6], event2[7]])
            self.xyToCCDcoeff = np.linalg.solve(matrix, coords)

        # Converts local coordinates to CCD coordinates
        def XYtoCCD(self, x, y):
            return (round(self.xyToCCDcoeff[0] * x + self.xyToCCDcoeff[1] * y
                          + self.xyToCCDcoeff[2]),
                    round(self.xyToCCDcoeff[3] * x + self.xyToCCDcoeff[4] * y
                          + self.xyToCCDcoeff[5]))

        # Initialises GTI arrays
        def initGTI(self, gtis, observationStart, observationEnd):
            self.GTI = self.GTI = [(observationStart - 1, 1)]
            for gti in gtis:
                self.GTI.append((gti[0], 0))
                self.GTI.append((gti[1], 1))
            self.GTI.append((observationEnd + 1, 0))

        # Lower bound binary search
        @staticmethod
        def lowerBound(t, arr, ind):
            L = 0
            R = len(arr)
            while R - L > 1:
                m = (R + L) // 2
                if arr[m][ind] <= t:
                    L = m
                else:
                    R = m
            return L

        # Checks if event is in GTI
        def inGTI(self, event):
            _, _, t = event[:3]
            return not self.GTI[self.lowerBound(t, self.GTI, 0)][1]

        # Obsolete and incorrect method
        def GTIpart(self, tm, td):
            ts = tm - td
            te = tm + td
            i = self.lowerBound(ts, self.GTI, 0)
            gti = 0
            bti = 0
            while ts < te:
                end = te
                if i + 1 < len(self.GTI):
                    end = min(te, self.GTI[i + 1][0])
                if self.GTI[i][1]:
                    bti += end - ts
                else:
                    gti += end - ts
                i += 1
                ts = end
            return gti, bti

        # Adds an event to one of the lists (need to be performed before initCoords)
        def addEvent(self, event):
            if self.inGTI(event):
                self.GTIevents.append(event)
            else:
                self.BTIevents.append(event)

        # Returns number of events in GTIs and BTIs between tmin and tmax
        def getGTIBTI(self, tmin, tmax):
            return (self.lowerBound(tmax, self.GTIevents, 2)
                    - self.lowerBound(tmin, self.GTIevents, 2) + 1,
                   self.lowerBound(tmax, self.BTIevents, 2)
                    - self.lowerBound(tmin, self.BTIevents, 2) + 1)

    # Background calculations class
    class Background:
        ccdnum = 0
        backgrounds = None
        nums = None
        duration = 0

        def __init__(self, ccdnum, duration, bincount, localToPic,
                     eventToCCDnum, localToPicScale, picToLocal):
            self.bincount = bincount
            self.localToPic = localToPic
            self.eventToCCDnum = eventToCCDnum
            self.localToPicScale = localToPicScale
            self.picToLocal = picToLocal
            self.ccdnum = ccdnum
            self.duration = duration
            self.backgrounds = np.zeros((ccdnum, bincount, bincount),
                                        dtype=float)
            self.nums = np.zeros(ccdnum, dtype=int)

        # Appends an event to background precalculation
        def add(self, event):
            i = self.eventToCCDnum(event)
            self.nums[i] += 1
            xp, yp = self.localToPic(event)
            self.backgrounds[i, xp, yp] += 1

        # Calculates background coefficients
        def init(self):
            for i in range(self.ccdnum):
                if self.backgrounds[i].max() != 0:
                    self.backgrounds[i] /= self.nums[i]
            self.backgrounds = np.rollaxis(self.backgrounds, 0, 3)

        # Calculates background coefficient within the circle with given radius and centre
        def calc(self, x, y, radius, backgrounds):
            sum_ = 0
            if len(backgrounds) != self.ccdnum:
                raise Exception(
                    "Incorrect numbers of ccd backgrounds: "
                    + str(len(backgrounds)) + " instead of " + str(self.ccdnum))
            num = 0
            bck = np.asarray(backgrounds)
            xp, yp = self.localToPic([x, y])
            k = ceil(radius * self.localToPicScale)
            for i in range(xp - k, xp + k + 1):
                for j in range(yp - k, yp + k + 1):
                    f = False
                    for dx in [0, 1]:
                        for dy in [0, 1]:
                            xl, yl = self.picToLocal([i + dx, j + dy])
                            f |= ((x - xl) ** 2 + (y - yl) ** 2) ** 0.5 <= radius
                    if f:
                        num += 1
                        sum_ += np.dot(bck, self.backgrounds[i][j])
            return sum_ / num * (pi * radius ** 2 * (self.localToPicScale ** 2))

    # Convert local telescope coordinates to picture coordinates
    def __init__(self, obsId, config, bincount=512):
        self.obsID = obsId
        self.bincount = bincount                                                                                        # Number of bins per axis in output picture
        self.localToPicScale = None                                                                                     # Local to picture coordinate transform scale
        self.meanFieldBackground = None                                                                                 # Mean background during the whole observation
        self.xmin, self.xmax, self.ymin, self.ymax = None, None, None, None                                             # Min and max possible values of local coordinates
        self.refx, self.refy = None, None                                                                               # Local coordinates of LOS
        self.xdel, self.ydel = None, None                                                                               # Angular measure of one local pixel (coordinates per degree)
        self.raref, self.decref = None, None                                                                            # FK5 coordinates of LOS (degrees)
        self.observationBegin = None                                                                                    # Date of exposure beggining
        self.picPsf = None                                                                                              # Point Spread Function radius in units of picture pixels
        self.localPsf = None                                                                                            # Point Spread Function Radius in units of local pixels
        self.observationPeriod = None                                                                                   # Period between the earliest and the latest photons detected
        self.fullImageArea = None                                                                                       # Area of image in local coordinate pixels (12 of one pn chip area)
        self.psfArea = None                                                                                             # Area of psf in local coordinate pixels
        self.distanceCheck = None                                                                                       # Function that checks if two points are close to each other
        self.pglobCoefficient = 1                                                                                       # Coefficient used to convert local probabilities to global
        self.fakex = None                                                                                               # X coordinate of the fake transient
        self.fakey = None                                                                                               # Y coordinate of the fake transient
        self.faketmin = None                                                                                            # Start time of the fake transient
        self.faketmax = None                                                                                            # End time of the fake transient
        self.CCDs = None                                                                                                # List of CCDs
        self.GTIbkg = None                                                                                              # Good Time Interval background map
        self.BTIbkg = None                                                                                              # Bad Time Interval background map
        self.result = None                                                                                              # Result of the program
        self.timeStamp = None                                                                                           # Time stamp used to measure local work time
        self.globalTimeStamp = None                                                                                     # Time stamp used to measure whole program work time
        self.config = config                                                                                            # Configure parameters
        self.workDir = os.getcwd()                                                                                      # Path to the working directory

    # Converts local telescope coordinates to local coordinates
    def localToPic(self, xyte):
        return (floor((self.ymax - xyte[1]) * self.localToPicScale),
                floor((xyte[0] - self.xmin) * self.localToPicScale))

    # Converts picture coordinates to local telescope coordinates
    def picToLocal(self, xyte):
        return np.asarray([xyte[1] / self.localToPicScale + self.xmin, self.ymax
                           - xyte[0] / self.localToPicScale] + xyte[2:])

    # Creates a picture of given photons
    # If xys is not empty, creates red psf borders around given photons
    def showEvents(self, photons, title, gamma=0.7, xys=None):
        if xys is None:
            xys = []
        pixels = np.zeros((self.bincount, self.bincount), dtype=float)
        for event in photons:
            x, y = self.localToPic(event)
            pixels[x, y] += 1
        for i in range(self.bincount):
            for j in range(self.bincount):
                pixels[i, j] = pixels[i, j] ** (1 - gamma)
        plt.imshow(pixels, cmap="Greys")
        pixels = np.zeros((self.bincount, self.bincount), dtype=float)
        self.picPsf *= 3
        for xy in xys:
            x, y = self.localToPic(xy)
            for k in range(-self.picPsf, self.picPsf + 1):
                pixels[x - self.picPsf, y + k] = 1
                pixels[x + self.picPsf, y + k] = 1
                pixels[x + k, y - self.picPsf] = 1
                pixels[x + k, y + self.picPsf] = 1
        if xys:
            plt.imshow(pixels, alpha=pixels, cmap="Reds")
        self.picPsf //= 3
        plt.title(title)
        plt.show()

    # Downloads necessary files
    def getDetections(self, observationID):
        if not os.path.exists(self.workDir + "/data/" + observationID):
            os.mkdir(self.workDir + "/data/" + observationID)
        path = self.workDir + "/data/" + observationID + "/"
        if not os.path.exists(path + "PIEVLI.FTZ"):
            XMMNewton.download_data(observationID, filename=f"PIEVLI{observationID}.FTZ",
                                    level="PPS", extention="FTZ", name="PIEVLI")
            os.rename(self.workDir + f"/PIEVLI{observationID}.FTZ", path + "PIEVLI.FTZ")
        if self.config.removeStars:
            if not os.path.exists(path + "OBSMLI.tar"):
                XMMNewton.download_data(observationID, filename=f"OBSMLI{observationID}.tar",
                                        level="PPS", extention="FTZ", name="OBSMLI")
                os.rename(self.workDir + f"/OBSMLI{observationID}.tar", path + "OBSMLI.tar")
            tar = tarfile.open(path + "OBSMLI.tar", "r")
            if not os.path.exists(path + "OBSMLI.FTZ"):
                for member in tar.getmembers():
                    if member.name[-3:] == "FTZ":
                        tar.extract(member, path=".")
                ftz_path = (self.workDir + "/" + observationID + "/pps/" + "P" + observationID
                            + "EPX000OBSMLI0000.FTZ")
                os.rename(ftz_path, path + "OBSMLI.FTZ")
                shutil.rmtree(self.workDir + "/" + observationID)

    # Removes unnecessary files
    @staticmethod
    def removeDetections(observationID):
        directory = os.getcwd() + "/data/" + observationID
        if os.path.exists(directory):
            shutil.rmtree(directory)

    # Creates a picture of given stars
    def showStars(self, starsPos):
        pixels = np.zeros((self.bincount, self.bincount), dtype=float)
        for star in starsPos:
            x, y = self.localToPic(star)
            pixels[x, y] += 1
        for i in range(self.bincount):
            for j in range(self.bincount):
                pixels[i, j] = (pixels[i, j])
        plt.imshow(pixels, cmap="Greys")
        plt.show()

    # Initializes necessary parameters for coordinates conversions
    def initWCS(self, header):
        self.xmin, self.xmax = header["REFXLMIN"], header["REFXLMAX"]
        self.ymin, self.ymax = header["REFYLMIN"], header["REFYLMAX"]
        self.refx, self.refy = header["REFXCRPX"], header["REFYCRPX"]
        self.xdel, self.ydel = header["REFXCDLT"], header["REFYCDLT"]
        self.raref, self.decref = header["REFXCRVL"], header["REFYCRVL"]
        xm = self.xmax - self.xmin + 1
        ym = self.ymax - self.ymin + 1
        self.localToPicScale = self.bincount / max(xm, ym)
        self.localPsf = ceil(abs(15 / 3600 / self.xdel))                                                                # 15 seconds psf
        self.picPsf = ceil(abs(self.localPsf * self.localToPicScale))
        self.fullImageArea = 200 * 64 * (4.1 / 3600) ** 2 * 12 / abs(
            self.xdel * self.ydel)                                                                                      # Every of 12 pn chips has 200×64 size, and every chip pixel covers 4.1 arcsec of scy
        self.psfArea = self.localPsf ** 2 * pi
        self.distanceCheck = (
            lambda e1, e2: TransientFinder.distanceCheckGeneral(e1, e2,
                                                                self.localPsf))                                         # Checks if events are in one PSF (taxicab type)
        self.observationBegin = Time(header["DATE-OBS"], format="isot", scale="tt")

    # Converts the local telescope XY coordinates to FK5 RA DEC coordinates
    def localToFK5(self, x, y):
        dec = (y - self.refy) * self.ydel + self.decref
        ra = (x - self.refx) * self.xdel / cos(radians(dec)) + self.raref
        return ra, dec

    # Converts FK5 RA DEC coordinates to local telescope XY coordinates
    def FK5ToLocal(self, ra, dec):
        x = (ra - self.raref) * cos(radians(dec)) / self.xdel + self.refx
        y = (dec - self.decref) / self.ydel + self.refy
        return x, y

    # Adds fake transient
    def addFakeTransient(self, photons, pglob, transTime, ccds, size=None):
        if size is None:
            size = self.localPsf
        originPhoton = choice(photons)
        self.fakex = originPhoton[0]
        self.fakey = originPhoton[1]
        additionalPhotons = []
        self.faketmin = transTime * random()
        self.faketmax = transTime - self.faketmin
        self.faketmin = originPhoton[2] - self.faketmin
        self.faketmax += originPhoton[2]
        faketmean = (self.faketmin + self.faketmax) / 2
        faketd = self.faketmax - faketmean
        ccdNumber = self.eventToCCDnum(originPhoton)
        cntGood = 0
        for photon in photons:
            if (((photon[0] - self.fakex) ** 2 + (photon[1] - self.fakey) ** 2)
                    <= size ** 2 and self.faketmin <= photon[2] <= self.faketmax
                    and not photon[4]):
                cntGood += 1

        cntBackground = self.calcBackground(self.fakex, self.fakey,
                                            faketmean, faketd)

        L = self.config.transientCountThreshold - cntGood - 1                                                           # To ensure that there are at least transientCountThreshold photons
        R = len(photons)
        while R - L > 1:
            m = (L + R) // 2
            if self.pGlob(cntGood + m, cntBackground, faketd * 2,
                          faketd * 2 * self.config.timeScale) > pglob:
                L = m
            else:
                R = m

        if self.config.showDebug:
            format_ = [R, cntGood, cntBackground,
                       self.pGlob(cntGood + R, cntBackground, faketd * 2,
                                  faketd * 2 * self.config.timeScale), faketd]
            print("Added {} to {} photons, background: {}. Pglob: {}, faketd: {}".format(*format_))

        times = [(self.faketmax - self.faketmin) * random() + self.faketmin for _ in range(R)]

        for i in range(R):
            photon = originPhoton.copy()
            photon[2] = times[i]
            r, phi = (size / 3 * sqrt(-2.0 * log(random())), random())                                                  # Box-Miller transform; dispersion of normal distribution is third of PSF size
            r %= size                                                                                                   # To be sure that added photons are not out of psf
            x, y = (r * cos(2 * pi * phi) + self.fakex,
                    r * sin(2 * pi * phi) + self.fakey)
            photon[0], photon[1] = x, y
            photon[6], photon[7] = ccds[ccdNumber].XYtoCCD(x, y)
            additionalPhotons.append(photon)

        result = np.append(photons, np.asarray(additionalPhotons), axis=0)
        return self.timeSort(result)

    # Checks if transient with given parameters are similar to added fake transient
    def checkFakeTransient(self, x, y, tmean, tdev):
        return ((x - self.fakex) ** 2 + (y - self.fakey) ** 2 <= self.localPsf ** 2
                and not (tmean - tdev > self.faketmax or tmean + tdev < self.faketmin))

    # Sorts array of events by their time
    @staticmethod
    def timeSort(arr):
        return arr[np.argsort(arr[:, 2])]

    # Gets all photons from given observation
    def getEvents(self, observationID):
        eventsFile = fits.open(self.workDir + "/data/" + observationID + "/PIEVLI.FTZ")
        self.initWCS(eventsFile[0].header)
        photons = eventsFile[1].data
        photons = np.vstack((photons["X"], photons["Y"], photons["TIME"],
                             photons["PI"], photons["FLAG"], photons["CCDNR"],
                             photons["RAWX"], photons["RAWY"])).transpose((1, 0))

        times = photons[:, 2]
        timemin = np.min(times)
        timemax = np.max(times)
        self.observationPeriod = timemax - timemin
        self.pglobCoefficient *= self.fullImageArea / self.psfArea * self.observationPeriod                             # Look-elsewhere effect Pglob coefficient

        photons = self.timeSort(photons)
        return photons

    # Removes photons near to detected stars
    def removeStars(self, photons, observationID, show=False):
        targets_file = fits.open("data/" + observationID + "/OBSMLI.FTZ")
        stars = np.vstack((targets_file[1].data["RA"],
                           targets_file[1].data["DEC"])).transpose((1, 0))
        stars = np.asarray([np.asarray(list(self.FK5ToLocal(i[0], i[1])) + [0, 0]) for i in stars])
        mask = set()
        for star in stars:
            x, y = self.localToPic(star)                                                                                # Picture coordinates are used to optimise star masking
            for i in range(-self.picPsf, self.picPsf):
                for j in range(-self.picPsf, self.picPsf):
                    if i ** 2 + j ** 2 <= self.picPsf ** 2:
                        mask.add((i + x, j + y))
        count = 0
        for e in photons:
            if self.localToPic(e) not in mask:
                count += 1
        clearedEvents = np.zeros((count, len(photons[0])),
                                 dtype=type(photons[0]))
        for e in photons:
            if self.localToPic(e) not in mask:
                count -= 1
                clearedEvents[count] = e
        if show:
            self.showStars(stars)
        return clearedEvents

    # Slow method to count all photons near to given within time diapason
    def countValidEvents(self, photons, e, selectionTime=float("inf")):
        selectionTime /= 2.0
        count = 0
        for event in photons:
            if abs(event[2] - e[2]) <= selectionTime:
                count += self.distanceCheck(e, event)
        return count

    # Fast method to calculate all photons near to each other within time diapason
    # If selection time is None, count is performed regardless of time
    # Chosen indexes contains information about what photons should be processes
    def calcPhotons(self, photons, selectionTime=None, chosenIndexes=None):
        f = selectionTime is None
        if not f:
            selectionTime /= 2.0
        if f:
            ft = FenwickTree(photons, self.xmax, self.ymax)
        else:
            ft = FenwickTree([], self.xmax, self.ymax)
        result = np.zeros((len(photons)), int)
        left = 0
        right = 0
        maxResult = 0
        for i, photon in enumerate(photons):
            if chosenIndexes is not None and not chosenIndexes[i]:
                continue
            if not f:
                current_time = photon[2]
                while current_time - photon[2] > selectionTime:
                    x, y = ft.event_to_xy(photons[left])
                    ft.add(x, y, -1)
                    left += 1
                while (right < len(photons) and photons[right][2] - current_time
                       <= selectionTime):
                    x, y = ft.event_to_xy(photons[right])
                    ft.add(x, y, 1)
                    right += 1
            x, y = ft.event_to_xy(photon)
            result[i] = ft.count(x - self.localPsf, x + self.localPsf,
                                 y - self.localPsf, y + self.localPsf)
            maxResult = max(maxResult, float(result[i]))
        return result

    # Shows histogram of the number of photons versus time
    def showEventsDiagram(self, photons, xy, binTime, timeMin=None, timeMax=None):
        cnt = 0
        result = np.zeros((len(photons)), float)
        for photon in photons:
            if self.distanceCheck(photon, xy):
                if (((timeMin is not None) and timeMin <= photon[2] <= timeMax)
                        or timeMin is None):
                    result[cnt] = photon[2]
                    cnt += 1
        result = result[:cnt]
        result = result - result.min()
        if timeMin is None:
            plt.hist(result, bins=ceil((result.max() - result.min()) / binTime))
        else:
            plt.hist(result)
        plt.show()

    # Removes photons, which are likely not in transients
    # This method is obsolete due to frequent removal of good photons
    def firstLayerCheck(self, photons):
        firstLayerCheckTimeInterval = self.config.transientDuration
        meanBackground = len(photons) / 12 / self.observationPeriod * firstLayerCheckTimeInterval
        left = 0
        right = 0
        candidates = np.zeros(len(photons), dtype=bool)
        binCounts = [0] * 12
        maxx = 64
        maxy = 200
        border = 2
        for i in range(len(photons)):
            while photons[left, 2] < (photons[i, 2] - firstLayerCheckTimeInterval / 2):
                binCounts[int(photons[left, 5]) - 1] -= 1
                left += 1
            while ((right < len(photons))
                   and (photons[right, 2] < (photons[i, 2] + firstLayerCheckTimeInterval / 2))):
                binCounts[int(photons[right, 5]) - 1] += 1
                right += 1
            if ((border < photons[i, 6] < maxx - border)
                    and (border < photons[i, 7] < maxy - border)):
                if binCounts[int(photons[i, 5]) - 1] >= meanBackground * 1.2:
                    candidates[i] = True
            else:
                candidates[i] = True

        return candidates

    # Removes unnecessary events such as out of bounds or bad flag events
    def removeBadEvents(self, photons):
        candidates = np.zeros(len(photons), dtype=bool)
        for i, photon in enumerate(photons):
            if (photon[0] > self.xmax or photon[1] > self.ymax or photon[0] < 0
                    or photon[1] < 0):
                candidates[i] = False
            elif photon[4] > self.config.flagFilter:
                candidates[i] = False
            else:
                candidates[i] = True
        return photons[candidates]

    # Writes necessary information about photons into cache file
    def writePhotons(self, photons, filePath="/cache/input.txt"):
        filePath = self.workDir + filePath
        with open(filePath, "w") as file:
            print(len(photons), file=file)
            for photon in photons:
                print(f"{photon[0]} {photon[1]} {photon[2]}", file=file)
            print(self.xmax, self.ymax, self.localPsf, file=file)
            file.close()

    # Writes flags for each photon into cache file
    def writeFlags(self, chosenIndexes, selectionTime, filePath="/cache/flags.txt"):
        filePath = self.workDir + filePath
        with open(filePath, "w") as file:
            print(len(chosenIndexes), file=file)
            for val in chosenIndexes:
                print("1" if val else "0", file=file)
            print(selectionTime, file=file)
            file.close()

    # Uniform global probability calculation function
    def pGlob(self, eventCnt, backgroundCnt, eventDuration, backgroundDuration):
        if type(eventCnt) is not list:
            eventCnt = [eventCnt]
            backgroundCnt = [backgroundCnt]
            eventDuration = [eventDuration]
            backgroundDuration = [backgroundDuration]
        prob = log(self.pglobCoefficient / sum(eventDuration))
        for i, cnt in enumerate(eventCnt):
            prob += poisson.logsf(round(cnt) - 1, backgroundCnt[i]
                                  / backgroundDuration[i] * eventDuration[i])
        return exp(prob)

    # Gets number of given event CCD
    @staticmethod
    def eventToCCDnum(event):
        return int(event[5]) - 1

    # Calculates background with the fanciest method I've ever programmed
    def calcBackground(self, x, y, tm, td, both=False):
        gtis = []
        btis = []
        for i in range(12):
            gti, bti = self.CCDs[i].getGTIBTI(tm - td, tm + td)
            gtis.append(gti)
            btis.append(bti)
        gti = self.GTIbkg.calc(x, y, self.localPsf, gtis)
        bti = self.BTIbkg.calc(x, y, self.localPsf, btis)
        if both:
            return gti, bti
        return gti + bti

    # Calculates Good Time Intervals
    def GTIcalc(self, events):
        pregtis = []
        cnt = 0
        begin = events[0][2]
        cts = []
        for i in range(len(events) + 1):
            event = [None, None, events[-1][2]]
            if i != len(events):
                event = events[i]
            if event[0] is None or event[2] - begin > self.config.binDur:
                cts.append(cnt / (event[2] - begin))
                if cnt / (event[2] - begin) < self.config.BTIthreshold:
                    pregtis.append((begin, event[2]))
                begin = event[2]
                cnt = 0
                if event[0] is None:
                    break
            cnt += self.config.hed[0] <= event[4] <= self.config.hed[1]
        if len(pregtis) == 0:
            raise Exception("There are no GTIs???")
        gtis = [pregtis[0]]
        for gti in pregtis[1:]:
            if gti[0] == gtis[-1][1]:
                gtis[-1] = (gtis[-1][0], gti[1])
            else:
                gtis.append(gti)
        if self.config.showDebug:
            plt.plot(cts)
            plt.title("High energy photons count rate")
            plt.show()
        return gtis

    # Initialises program result variable
    def initResults(self, obsID):
        self.result = {"type": "Unknown", "observationID": obsID, "transients": []}

    # Adds transient to program result variable
    def addTransientToResult(self, x, y, tmean, tdev, cntgti, backgroundgti,
                             cntbti, backgroundbti, prob):
        ra, dec = self.localToFK5(x, y)
        self.result["transients"].append({
            "x": float(x),
            "y": float(y),
            "tmean": float(tmean),
            "tdev": float(tdev),
            "timebegin": (self.observationBegin + u.s * (float(tmean) - float(tdev))).isot,
            "cntGTI": int(cntgti),
            "backgroundGTI": float(backgroundgti),
            "cntBTI": int(cntbti),
            "backgroundBTI": float(backgroundbti),
            "prob": float(prob),
            "ra": float(ra),
            "dec": float(dec),
        })

    # More human readable str() function for integers
    def sstr(self, i: int, arr=None):
        if arr is None:
            arr = []
        if i == 0:
            if not arr:
                return "0"
            return "`".join([str(arr[0])] + [str(i // 100) + str(i % 100 // 10)
                                             + str(i % 10) for i in arr[1:]])
        return self.sstr(i // 1000, [i % 1000] + arr)

    # Main transient finding function
    def process(self):
        self.timeStamp = time()
        self.globalTimeStamp = time()
        self.initResults(self.obsID)

        # Gets events from certain observation
        self.CCDs = [self.CCD(i) for i in range(12)]
        try:
            self.getDetections(self.obsID)
        except Exception:
            self.result["type"] = "NetworkError"
            self.result["traceback"] = traceback.format_exc()
            return
        events = None
        try:
            events = self.getEvents(self.obsID)
        except Exception:
            self.result["type"] = "PIEVLIError"
            self.result["traceback"] = traceback.format_exc()
            return
        if self.config.showDebug:
            self.showEvents(events, "Image before processing")

        # Removes star events
        size1 = len(events)
        try:
            if self.config.removeStars:
                events = self.removeStars(events, self.obsID, show=False)
        except Exception:
            self.result["type"] = "OBSMLIError"
            self.result["traceback"] = traceback.format_exc()
            return
        size2 = len(events)
        if self.config.showDebug:
            print("Number of events with/without stars:", self.sstr(size1),
                  self.sstr(size2))
        self.result["EventsWithStars"] = size1
        self.result["EventsWithoutStars"] = size2
        events = self.removeBadEvents(events)
        events = self.timeSort(events)

        # Background computation
        self.GTIbkg = self.Background(12, self.observationPeriod, self.bincount,
                                      self.localToPic, self.eventToCCDnum,
                                      self.localToPicScale, self.picToLocal)
        self.BTIbkg = self.Background(12, self.observationPeriod, self.bincount,
                                      self.localToPic, self.eventToCCDnum,
                                      self.localToPicScale, self.picToLocal)

        if True:
            gtis = self.GTIcalc(events)
            tmin = events[0][2]
            tmax = events[-1][2]
            for ccd in self.CCDs:
                ccd.initGTI(gtis, tmin, tmax)

        gtievents = np.zeros(len(events), dtype=bool)

        for i, event in enumerate(events):
            self.CCDs[self.eventToCCDnum(event)].addEvent(event)
            if self.CCDs[self.eventToCCDnum(event)].inGTI(event):
                self.GTIbkg.add(event)
                gtievents[i] = True
            else:
                self.BTIbkg.add(event)
                gtievents[i] = False

        if self.config.removeBTI:
            events = events[gtievents]
            gtievents = gtievents[gtievents]
            self.result["EventsWithoutBTI"] = len(events)

        for ccd in self.CCDs:
            ccd.initCoords()

        self.GTIbkg.init()
        self.BTIbkg.init()

        if self.config.addFakeTransient:
            events = self.addFakeTransient(events, self.config.ftransientProb,
                                           self.config.ftransientTime,
                                           self.CCDs, self.localPsf)

            gtievents = np.zeros(len(events), dtype=bool)

            for i, event in enumerate(events):
                if self.CCDs[self.eventToCCDnum(event)].inGTI(event):
                    gtievents[i] = True
                else:
                    gtievents[i] = False

            if self.config.removeBTI:
                events = events[gtievents]
                gtievents = gtievents[gtievents]
                self.result["EventsWithoutBTI"] = len(events)

        meanFieldBackground = events.size / self.observationPeriod / self.fullImageArea

        # Removes non-transient events and gets number of “near” events within 10-seconds interval
        indexes = np.ones(len(events), dtype=bool)                                                                      # There was an optimisation; it was bad, so it got removed

        tenSeconds = sifting.calc(*events[:, :3].transpose(), self.xmax,
                                  self.ymax, self.localPsf, indexes,
                                  self.config.transientDuration)

        chosenIndexes = np.zeros(len(events), dtype=bool)
        for i in range(len(events)):
            if tenSeconds[i] >= self.config.goodPhotonsCountThreshold:
                chosenIndexes[i] = True

        if self.config.showDebug:
            self.showEvents(events, "Image after star removal")
            self.showEvents(events[chosenIndexes], "Improbable events")

        self.result["ImprobableEvents"] = len(events[chosenIndexes])

        averageBackground = sifting.calc(*events[:, :3].transpose(), self.xmax,
                                         self.ymax, self.localPsf, chosenIndexes, 0)

        self.result["EventsSiftingTime"] = time() - self.timeStamp
        self.timeStamp = time()

        if self.config.showDebug:
            print("Processing time: ", round(self.result["EventsSiftingTime"]), "s", sep="")

        # Calculates the probability of every event to be from a background
        probabilityCoefficients = np.zeros((len(events)), dtype=float)

        for i in range(len(events)):
            if averageBackground[i] == 0:
                probabilityCoefficients[i] = float("inf")
                continue
            probabilityCoefficients[i] = self.pGlob(tenSeconds[i], averageBackground[i],
                                                    self.config.transientDuration,
                                                    self.observationPeriod)

        # Distributes events into transients
        potentialIndexes = []
        for i in range(len(events)):
            if probabilityCoefficients[i] <= self.config.probabilityThreshold:
                potentialIndexes.append(i)

        # Checks for number of potential events in fake transient
        if self.config.addFakeTransient:
            fakecnt = 0
            for event in events[potentialIndexes]:
                if (self.faketmin <= event[2] <= self.faketmax
                        and (self.fakex - event[0]) ** 2 + (self.fakey - event[1]) ** 2
                        <= self.localPsf ** 2):
                    fakecnt -= -1

            if self.config.showDebug:
                print(f"{fakecnt} photons of fake transient found")

            self.result["FakeCount"] = fakecnt

        # Continues to distribute
        cluster = sifting.clustering(*events[potentialIndexes, :3].transpose(),
                                     -np.log10(probabilityCoefficients[potentialIndexes]),
                                     self.localPsf * 3, self.config.timeThreshold)
        pretransients = [[] for _ in range(np.max(cluster) + 1)]
        for i, clust in enumerate(cluster):
            pretransients[clust].append(events[potentialIndexes[i]])

        self.result["Pretransients"] = len(pretransients)

        transients = []
        for transient in pretransients:
            if len(transient) >= self.config.transientCountThreshold:
                transients.append(transient)

        self.result["TransientCandidates"] = len(transients)
        self.result["TransientComputationTime"] = time() - self.timeStamp
        self.timeStamp = time()

        if self.config.showDebug:
            print("Number of transients:", len(transients))
            print("Transient computation time: ",
                  round(self.result["TransientComputationTime"]), "s", sep="", )

        # Calculates global probabilities of transients
        def transCalc(innerSigmaCoefficient=1, outerSigmaCoefficient=3, bcgUncertainty=100):
            if len(transients) == 0:
                if self.config.showDebug:
                    print("No transients at all...")
                return [], []

            timeStamp = time()

            transients.sort(key=lambda x: -len(x))

            probs = []
            goodTransients = []
            inf = 10000

            xms = []
            yms = []
            timems = []
            timeds = []
            for j, photons in enumerate(transients):
                xm, ym = 0, 0
                for photon in photons:
                    xm += photon[0]
                    ym += photon[1]
                xm /= len(photons)
                ym /= len(photons)

                beginTime = min([i[2] for i in photons])
                endTime = max([i[2] for i in photons])
                timem = (beginTime + endTime) / 2
                timed = (endTime - beginTime) / 2

                xms.append(xm)
                yms.append(ym)
                timems.append(timem)
                timeds.append(timed)

            timeCoefficient = (bcgUncertainty ** 2 / (
                    meanFieldBackground * self.psfArea *
                    (outerSigmaCoefficient ** 2 - innerSigmaCoefficient ** 2))
                               / (min(timeds) * 2))
            counts = sifting.count(xms, yms, timems, timeds, gtievents,
                                   *events[:, :3].transpose(), self.localPsf,
                                   innerSigmaCoefficient, outerSigmaCoefficient,
                                   timeCoefficient)
            flags = []

            if self.config.addFakeTransient:
                self.result["FakeTransientStatus"] = "NotFound"

            for j in range(len(transients)):
                cntTransientsGTI, cntTransientsBTI = (counts[j], counts[j + len(transients)])

                cntObsGTI, cntObsBTI = self.calcBackground(xms[j], yms[j],
                                                           timems[j], timeds[j],
                                                           both=True)

                try:
                    pglob = self.pGlob([cntTransientsGTI, cntTransientsBTI],
                                       [cntObsGTI, cntObsBTI], [timeds[j], timeds[j]],
                                       [timeds[j], timeds[j]])
                except ValueError:
                    pglob = inf
                probs.append(pglob)

                isFake = False

                if pglob <= self.config.pglobThresh:
                    if self.config.addFakeTransient and self.checkFakeTransient(xms[j], yms[j], timems[j], timeds[j]):
                        if self.config.showDebug:
                            print("-" * 10)
                            print("Found fake transient:", round(xms[j], 1),
                                  round(yms[j], 1), round(timems[j], 1),
                                  round(timeds[j], 2), pglob, cntTransientsGTI,
                                  cntObsGTI, cntTransientsBTI, cntObsBTI)

                            print(round(self.fakex, 1), round(self.fakey, 1),
                                  round((self.faketmax + self.faketmin) / 2, 1),
                                  round((self.faketmax - self.faketmin) / 2), 2)

                            print("-" * 10)
                        self.result["FakeTransientStatus"] = "Probable"
                        isFake = True
                    else:
                        goodTransients.append((xms[j], yms[j], timems[j],
                                               timeds[j], pglob))
                        self.addTransientToResult(xms[j], yms[j], timems[j],
                                                  timeds[j], cntTransientsGTI,
                                                  cntObsGTI, cntTransientsBTI,
                                                  cntObsBTI, pglob)
                        flags.append(True)
                else:
                    flags.append(False)
                    if (self.config.addFakeTransient
                        and self.checkFakeTransient(xms[j], yms[j], timems[j], timeds[j])):
                        if self.config.showDebug:
                            print("-" * 10)
                            print("Not Found:", round(xms[j], 1), round(yms[j], 1),
                                  round(timems[j], 1), round(timeds[j], 2), pglob,
                                  cntTransientsGTI, cntObsGTI, cntTransientsBTI,
                                  cntObsBTI)

                            print(round(self.fakex, 1), round(self.fakey, 1),
                                  round((self.faketmax + self.faketmin) / 2, 1),
                                  round((self.faketmax - self.faketmin) / 2), 2)

                            print("-" * 10)
                        self.result["FakeTransientStatus"] = "Improbable"
                        isFake = True

                if isFake:
                    self.result["FakeTransient"] = {
                        "x": xms[j],
                        "y": yms[j],
                        "tmean": timems[j],
                        "tdev": timeds[j],
                        "pglob": pglob,
                        "xset": self.fakex,
                        "yset": self.fakey,
                        "tmeanset": (self.faketmax + self.faketmin) / 2,
                        "tdevset": (self.faketmax - self.faketmin) / 2,
                    }

            self.result["TransientProcessingTime"] = time() - timeStamp

            if self.config.showDebug:
                print('Number of "good" transients:', len(goodTransients))
                print("Processing time: ",
                      round(self.result["TransientProcessingTime"], 2), "s", sep="")
            return goodTransients, flags

        goodTransients, _ = transCalc(self.config.innerRingScale,
                                      self.config.outerRingScale,
                                      self.config.transientBackgroundUncertainty)

        # Shows 2D and 3D distribution of photons belonging to transient
        def showTransient(photons, x, y, tmean, tdev, sigma, tcoeff, sigmacoeff, prob):
            xi, yi, xa, ya = (x - sigmacoeff * sigma, y - sigmacoeff * sigma,
                              x + sigmacoeff * sigma, y + sigmacoeff * sigma)
            pixels = np.zeros((xa - xi + 1, ya - yi + 1), dtype=int)
            pixelsTransient = np.zeros((xa - xi + 1, ya - yi + 1), dtype=float)
            photonsWithinFov = []
            for photon in photons:
                if (xi <= photon[0] < xa and yi <= photon[1] < ya
                        and abs(photon[2] - tmean) <= tdev * tcoeff):
                    arr = pixels
                    if ((photon[0] - x) ** 2 + (photon[1] - y) ** 2 <= sigma ** 2
                            and abs(photon[2] - tmean) <= tdev):
                        arr = pixelsTransient
                    photonsWithinFov.append(photon)
                    try:
                        for j in range(1, 3):
                            for k in range(1, 3):
                                arr[int(photon[0] - xi + j), int(photon[1] - yi + k)] += 1
                                arr[int(photon[0] - xi - j), int(photon[1] - yi + k)] += 1
                                arr[int(photon[0] - xi + j), int(photon[1] - yi - k)] += 1
                                arr[int(photon[0] - xi - j), int(photon[1] - yi - k)] += 1
                        arr[int(photon[0] - xi), int(photon[1] - yi)] += 1
                    except Exception:
                        pass
            plt.xlabel(str(10 ** prob))
            plt.imshow(pixels, cmap="Greys", norm="asinh")
            plt.imshow(pixelsTransient, alpha=pixelsTransient.astype(bool).astype(float),
                       cmap="Reds", norm="asinh")
            pixels = np.zeros((xa - xi + 1, ya - yi + 1), dtype=float)
            lineWidth = 2
            n, m = xa - xi + 1, ya - yi + 1
            for i in range(0, n):
                for j in range(0, m):
                    if abs(((n // 2 - i) ** 2 + (m // 2 - j) ** 2) ** 0.5 - sigma) <= lineWidth:
                        pixels[i, j] = 1
            plt.imshow(pixels, alpha=pixels, cmap="Reds")
            plt.show()
            t0 = tmean - tdev * tcoeff
            ax = plt.figure().add_subplot(projection="3d")
            xs = [int(i[1] - yi) for i in photonsWithinFov]
            ys = [int(xi - i[0]) for i in photonsWithinFov]
            ts = [int(i[2] - t0) for i in photonsWithinFov]
            x1, y1, t1, x2, y2, t2 = [], [], [], [], [], []
            for i, _ in enumerate(ts):
                if ((xs[i] + yi - y) ** 2 + (xi - ys[i] - x) ** 2 <= sigma ** 2
                        and abs(ts[i] + t0 - tmean) <= tdev):
                    x1.append(xs[i])
                    y1.append(ys[i])
                    t1.append(ts[i])
                else:
                    x2.append(xs[i])
                    y2.append(ys[i])
                    t2.append(ts[i])
            x1, y1, t1, x2, y2, t2 = (np.asarray(i) for i in [x1, y1, t1, x2, y2, t2])
            _, stemlines, _ = ax.stem(x2, y2, t2, linefmt=":", basefmt=" ", markerfmt="C1.")
            # ax.stem(x1, y1, t1, linefmt=":", basefmt=" ", markerfmt="C2.")
            # plt.setp(stemlines, visible=False)
            plt.show()

        coordinates = []

        for k, transient in enumerate(goodTransients):
            transient = goodTransients[k]
            if self.config.showTransients:
                showTransient(events, int(transient[0]), int(transient[1]),
                              *transient[2:4], int(self.localPsf), 3, 3, transient[4])
            coordinates.append(tuple(transient[0:2]))

        if self.config.showTransients:
            self.showEvents(events, "Transients",
                            xys=[(transient[0], transient[1]) for transient in goodTransients])

        self.result["CodeTime"] = time() - self.globalTimeStamp

        if self.config.showDebug:
            print("Code completion time: ", round(self.result["CodeTime"], 2), "s", sep="")

        if self.config.showVision:
            import Vision as vs

            visualiser = vs.Vision(events, self.bincount, self.picPsf, self.localToPic)
            visualiser.addAuxiliary(goodTransients)
            visualiser.initGUI()

        self.result["type"] = "Success"
