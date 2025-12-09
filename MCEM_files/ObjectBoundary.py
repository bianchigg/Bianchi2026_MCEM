import numpy as np
import math


class ObjectPoint:
    def __init__(self, position, albedo, columnInitialTemp, columnDepth, columnPointSpacing):
        # gives the point an initial position, np array
        self.position = position
        # column spacing and depth
        self.columnDepth = columnDepth
        self.columnPointSpacing = columnPointSpacing
        # gives the point an initial column temperature profile, np.array
        self.columnTempProfile = columnInitialTemp * np.ones(int(columnDepth / columnPointSpacing))
        # the point's initial albedo
        self.albedo = albedo
        # number of temperature profile points
        self.noTempPoints = int(columnDepth / columnPointSpacing)
        # define shaded condition, set to False initially
        self.isShaded = False
        # if wetted or dry perimeter condition
        self.isWet = False
        # add undercut condtion
        self.isUndercut = False
        # surface energy at point
        self.surfaceEnergy = 0
        # net sw
        self.sw = 0
        # net lw
        self.lw = 0
        # sensible
        self.sens = 0
        # latent
        self.latent = 0
        # water shear energy
        self.shear = 0
        # angle of incidence
        self.aoi = 0

    def tangent_gradient(self, LpointPos, RpointPos):
        # need to note difference for end points
        # using math.dist to avoid issue with symmetry at x =0
        dx = math.dist((LpointPos[0],), (RpointPos[0],))
        dy = math.dist((LpointPos[1],), (RpointPos[1],))
        dx = LpointPos[0] - RpointPos[0]
        dy = LpointPos[1] - RpointPos[1]

        if dx == 0:
            return 'not_defined'

        return np.round(dy / dx, 4)  # /(np.sqrt(dx**2 + dy**2)) does it need to be normalised?

    def norm_gradient(self, LpointPos, RpointPos):

        tangent_gradient = self.tangent_gradient(LpointPos, RpointPos)
        if tangent_gradient == 0:
            return 'not_defined'
        if tangent_gradient == 'not_defined':
            return 0

        grad = - 1 / tangent_gradient

        if np.isinf(grad) == True:
            grad = 'not_defined'
        elif np.isnan(grad) == True:
            grad = 'not_defined'
        # elif grad<0:
        #     grad = -grad
        return grad

    def surface_tilt(self, LpointPos, RpointPos):
        # measured from horizontal with 0 being flat on x and pi/2 being straight line down. VAlue should always
        # be positive as surface azimuth will account for position. Range is 0 - pi/2
        dx = LpointPos[0] - RpointPos[0]
        dy = LpointPos[1] - RpointPos[1]

        if dx == 0:
            return np.pi / 2
        # output in radians
        tilt = np.arctan2(abs(dy), abs(dx))
        # if tilt < 0:
        # tilt += np.pi
        return tilt


point = ObjectPoint(np.array([0, 5]), albedo=0.8, columnInitialTemp=263.15, columnDepth=3, columnPointSpacing=0.01)
