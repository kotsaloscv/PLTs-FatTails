###############################################################################
# This python script implements the statistical analysis presented in:
# Kotsalos et al.: Anomalous Platelet Transport & Fat-Tailed Distributions
# arXiv: 
# 
# It reads the positions of PLTs (center of mass) through time (stored in PLTs_tau_X directory)
# Every file in the directory stores the individual trajectory of a PLT in the form: (t,x,y,z)
# This is how our simulations are built (tailored on DNS output):
# t: refers to fluid time steps. It should be multiplied by the fluid time-step to return physical time
# x,z: vorticity, flow directions, respectively
# y: wall-bounded direction
# The above order can change with the right modifications.
# 
# The given PLTs directory inlcudes the positions of 95 platelets from a simulation
# executed for 1s physical time, in a box of 50^3 um^3, and constant shear rate
# 100 s^-1. The DNS are sampled at 1ms window and the numerical time step is
# 0.125 us according to the Lattice Boltzmann Method (see paper for a complete overview).
###############################################################################
# Contact:
# Christos Kotsalos
# Computer Science Department
# University of Geneva
# 7 Route de Drize
# 1227 Carouge, Switzerland
# kotsaloscv@gmail.com
# christos.kotsalos@unige.ch
###############################################################################
# Attribution 4.0 International (CC BY 4.0)
# You are free to:
# Share — copy and redistribute the material in any medium or format
# Adapt — remix, transform, and build upon the material for any purpose, even commercially.
# Under the following terms:
# Attribution — You must give appropriate credit, provide a link to the license, 
# and indicate if changes were made. You may do so in any reasonable manner, 
# but not in any way that suggests the licensor endorses you or your use.
###############################################################################

import glob
import sys
import os
import re
import copy
import time
import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy   import stats
from astropy import stats as astats
from joblib import Parallel, delayed


def numericalSort(value):
    '''
    Access the contents of a folder sorted numerically
    '''
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def upper_bounce_back(pos, domainSize):
    '''
    Used by PLTs class for top wall bounce back boundary condition
    '''
    if (pos > domainSize):
        jump = pos/domainSize
        if (jump > 2.):
            return -domainSize
        else:
            return domainSize - (jump - 1.) * domainSize
    else:
        return pos

def distFromWallsPLTs(y, b, t):
    '''
    Used by PLTs class to compute avg distance from walls
    y : PLT position
    b : bottom (0.)
    t : top (domainSize)
    '''
    return (y - b) if ( (y - b) < (t - y) ) else (t - y)

class PLTs:
    '''
    Random Walk Simulations:
    Simulate the Impact-R platelet function analyser
    PLTs that cross the lower boundary are considered deposited
    '''

    def __init__(self, numPLTs_, dt_, domainSize_,
                 distros_invECDF_, distros_tail_, xmins_):
        '''
        the domain can be split in multiple zones,
        each zone can have different distributions
        '''
        self.numPLTs = numPLTs_ # number of PLTs
        self.dt = dt_ # simulation time step in ms

        self.domainSize = domainSize_ # normally 820um as the Impact-R device

        self.distros_invECDF = distros_invECDF_ # inverse Empirical Distribution Function per zone
        self.distros_tail    = distros_tail_ # tail distribution per zone (refer to PLT velocities)
        self.xmins           = xmins_ # lower bound (x_min) for tail per zone

        self.zones = len(distros_invECDF_)

        pos = np.random.uniform(0.0, self.domainSize, self.numPLTs) # initialise PLT positions in the domain

        self.y  = []
        self.y0 = []
        zones_tmp = np.linspace(0.0, self.domainSize, self.zones+1)
        # Distribute PLTs per zone
        for z in range(self.zones):
            inds = np.where((pos >= zones_tmp[z]) & (pos < zones_tmp[z+1]))
            self.y.append(pos[inds])
            self.y0.append(np.copy(self.y[-1]))

        # MSD & average distance from walls
        self.t             = [0.,]
        self.MSD           = [0.,]
        self.distFromWalls = [0.,]
        
        distFromWallsPLTs_np = np.vectorize(distFromWallsPLTs)
        for z in range(self.zones):
            self.distFromWalls[-1] += np.sum(distFromWallsPLTs_np(self.y[z], 0.0, self.domainSize))
        self.distFromWalls[-1] /= self.numPLTs

    def advance(self, T):
        '''
        Advance the system through time
        '''
        upper_bounce_back_np = np.vectorize(upper_bounce_back)
        distFromWallsPLTs_np = np.vectorize(distFromWallsPLTs)
        
        # main time loop
        for t in range(T):
            
            toZones = []
            toZones0 = []
            for z in range(self.zones):
                toZones.append(np.array([], dtype=np.float64))
                toZones0.append(np.array([], dtype=np.float64))

            zones_tmp = np.linspace(0.0, self.domainSize, self.zones+1)
            # advancing per zone
            for z in range(self.zones):
                # initially generate velocities from the Empirical Distribution Function
                vels = self.distros_invECDF[z]['inv_ecdf'](np.random.uniform(self.distros_invECDF[z]['lb'], self.distros_invECDF[z]['ub'], size=self.y[z].shape[0]))
                # check which velocities are above the lower bound.
                # below no change
                # above we go with the tail distribution
                inds = np.where(vels >= self.xmins[z])
                
                # generate velocities (tail)
                vels[inds] = self.distros_tail[z]['distro'].rvs(*self.distros_tail[z]['params'], size=inds[0].shape[0])

                #**
                # Sanity check
                inds = np.where((~np.isnan(vels)) & (~np.isinf(vels)))
                vels = vels[inds]
                self.y[z]  = self.y[z][inds]
                self.y0[z] = self.y0[z][inds]
                #**

                # up/down: symmetric phenomenon
                vels *= np.random.choice([-1.,1.], size=self.y[z].shape[0])
            
                # advance in time
                self.y[z] += vels*self.dt

                # Top Boundary Condition (BC)
                # Either Bounce Back (BB) or trapping in Cell Free Layer (CFL)
                if   ( (sim_BC_ == 'tBB') or (sim_BC_ == 'NA') ):
                    self.y[z] = upper_bounce_back_np(self.y[z], self.domainSize)
                elif (sim_BC_ == 'tCFL'):
                    inds = np.where(self.y[z] < self.domainSize)
                    self.y[z]  = self.y[z][inds]
                    self.y0[z] = self.y0[z][inds]

                # Bottom BC ~> "Deposition"
                inds = np.where(self.y[z] > 0.0)
                self.y[z]  = self.y[z][inds]
                self.y0[z] = self.y0[z][inds]

                # Transfer PLTs between zones (conservation of mass)
                for z_ in range(self.zones):
                    if (z_ == z):
                        continue
                    inds = np.where((self.y[z] >= zones_tmp[z_]) & (self.y[z] < zones_tmp[z_+1]))
                    toZones[z_]  = np.append(toZones[z_] , self.y[z][inds])
                    toZones0[z_] = np.append(toZones0[z_], self.y0[z][inds])
                    self.y[z]  = np.delete(self.y[z] , inds)
                    self.y0[z] = np.delete(self.y0[z], inds)
                
            for z in range(self.zones):
                self.y[z]  = np.append(self.y[z] , toZones[z])
                self.y0[z] = np.append(self.y0[z], toZones0[z])

            bulk_PLTs = 0
            for z in range(self.zones):
                bulk_PLTs += self.y[z].shape[0]
            if (bulk_PLTs == 0):
                break

            # MSD & average distance from walls
            self.t.append(self.t[-1]+self.dt)
            self.MSD.append(0.)
            self.distFromWalls.append(0.)
            for z in range(self.zones):
                self.MSD[-1]           += np.sum(np.power(self.y[z] - self.y0[z], 2.0))
                self.distFromWalls[-1] += np.sum(distFromWallsPLTs_np(self.y[z], 0.0, self.domainSize))
            self.MSD[-1]           /= bulk_PLTs # MSD refers to the non-deposited PLTs, since we care about the effective diffusivity
            self.distFromWalls[-1] /= self.numPLTs # This division is not a bug, the deposited PLTs contribute to the avg distance from the walls.
    
    def depositedPLTs(self):
        '''
        return the number of deposited PLTs,
        deposition: PLTs that cross the bottom boundary are considered deposited (just a simplification)
        '''
        bulk_PLTs = 0
        for z in range(self.zones):
            bulk_PLTs += self.y[z].shape[0]

        return self.numPLTs - bulk_PLTs

    def meta_quantities(self):
        '''
        return MSD & average distance from walls
        '''
        from scipy import optimize
        # non_linear fitting
        def non_linear_(x, a, b):
            return a*np.power(x, b)
        # linear fitting
        def linear_(x, a, b):
            return a*x + b
        
        X = np.array(self.t)

        # MSD
        Y = np.array(self.MSD)
        MSD_params, _ = optimize.curve_fit(non_linear_, X, Y)

        # distFromWalls
        Y = np.array(self.distFromWalls)
        distFromWalls_params, _ = optimize.curve_fit(linear_, X, Y)

        return tuple(MSD_params), tuple(distFromWalls_params)


def find_xminOpt_distro(data, dist_name, upper_bound = np.inf):
    '''
    find the lower bound (x_min) for tail fitting,
    the tail varies per distribution
    '''
    distro = eval('stats.' + dist_name)
    data.sort()

    min_sample_size = 100

    data_len = data.shape[0]
    if (data_len <= min_sample_size):
        min_sample_size = data_len - 1

    def find_xmin_core(xmin_optimal):
        if (xmin_optimal > upper_bound):
            return 0.0 # lowest p-value
        
        import warnings
        warnings.filterwarnings('ignore')
        
        data_tmp   = data[data >= xmin_optimal]
        params_tmp = distro.fit(data_tmp)
        
        return stats.kstest(data_tmp, dist_name, params_tmp)[1]

    # multi-processing part (num_threads)
    p_vals = Parallel(n_jobs=num_threads)(delayed(find_xmin_core)(xmin_optimal) for xmin_optimal in data[:-min_sample_size])

    # Optimal xmin is the one that maximizes the p-value and minimizes the statistic (D_value)
    return data[np.argmax(p_vals)]


def LLR_test(data, dist_name_1, dist_name_2):
    '''
    Log-Likelihood Ratio test, see Clauset_2009 (Power-Law Distributions in Empirical Data)
    '''
    n = data.shape[0]

    distro1 = eval('stats.' + dist_name_1)
    distro2 = eval('stats.' + dist_name_2)

    params1 = distro1.fit(data)
    params2 = distro2.fit(data)

    logLik1_i = distro1.logpdf(data, *params1)
    logLik1   = np.sum(logLik1_i)
    logLik1_  = logLik1 / n
    logLik2_i = distro2.logpdf(data, *params2)
    logLik2   = np.sum(logLik2_i)
    logLik2_  = logLik2 / n

    LLR = logLik1 - logLik2

    sigma_sq = np.sum(np.power((logLik1_i - logLik1_) - (logLik2_i - logLik2_), 2.)) / n

    from scipy.special import erfc
    p = erfc(abs(LLR) / np.sqrt(2.*n*sigma_sq))

    # The lower the p, the more trust on the direction of LLR
    return LLR, p


def meta_process(tau):
    '''
    Main processing kernel
    '''
    print('Analyzing tau (ms): ', tau)

    import warnings
    warnings.filterwarnings('ignore')

    # Folder where you store the PLT positions (center of mass - COM) per DNS time steps
    data_location = which_bodies + '_tau_' + str(tau) + '/'

    numBodies = 0
    Bodies = []
    for log in sorted(glob.glob(data_location + '*_ComPos.log'), key=numericalSort):
        Bodies.append(numBodies)
        numBodies += 1

    # Build the data frames and fill them
    absolute_pos  = pd.DataFrame(columns=Bodies, dtype=np.float64)
    distFromWalls = pd.DataFrame(columns=Bodies, dtype=np.float64)
    MSD           = pd.DataFrame(columns=Bodies, dtype=np.float64)
    
    # Perform distributions checking in zones
    zones_vels          = []
    zones_distros       = []
    zones_MSD           = []
    zones_distFromWalls = []
    for z in range(zones_):
        zones_distros.append(np.array([], dtype=np.float64))
        zones_vels.append(np.array([], dtype=np.float64))
        zones_MSD.append(np.array([], dtype=np.float64))
        zones_distFromWalls.append(np.array([], dtype=np.float64))

    # Var that help us find the mean free path/ time (MFP/T) in comparison with the ground truth (gT, path from DNS)
    integrals_tau = []
    
    numBodies = 0
    for log in sorted(glob.glob(data_location + '*_ComPos.log'), key=numericalSort):

        df = pd.read_csv(log, delimiter=',', header=None, names=names_, usecols=usecols_, dtype={'t': np.float64, 'y': np.float64, 'z': np.float64})

        # Time in the original files interprets to how many DNS fluid time steps,
        # this is why we multiply here with DNS fluid time step to convert it into physical time in ms
        df = df.loc[df['t']*dt_f >= From_]
        df = df.loc[df['t']*dt_f <= To_]
        df = df.reset_index(drop=True)

        absolute_pos[numBodies] = df['y'].copy()

        if (do_what == 'MFP'):
            integrals_tau.append(np.trapz(df['y'], df['t']*dt_f))
        
        if ( (do_what == 'distros') or (do_what == 'MSD') or (do_what == 'distFromWalls') ):

            MSD[numBodies] =  pd.Series((df['y'] - df['y'].iloc[0]) * (df['y'] - df['y'].iloc[0]))

            distFromWalls[numBodies] = df['y'].apply(lambda y: (y - bottom_wall) if ( (y - bottom_wall) < (top_wall - y) ) else (top_wall - y))

            pos        = absolute_pos[numBodies].to_numpy()
            pos_rolled = np.roll(pos, 1)

            # velocity in um/ms
            vel = (pos - pos_rolled) / tau
            vel[0] = np.nan
            
            # Exclude erroneous jumps
            dp = np.absolute(pos-pos_rolled)
            inds = np.where(dp < ((top_wall - bottom_wall) - 5.0))
            pos = pos[inds]
            vel = vel[inds]
            
            inds = np.where((~np.isnan(vel)) & (~np.isinf(vel)))
            pos = pos[inds]
            vel = vel[inds]

            zones_tmp = np.linspace(bottom_CFL, top_CFL, zones_+1)
            for z in range(zones_):
                inds = np.where((pos >= zones_tmp[z]) & (pos < zones_tmp[z+1]))
                zones_distros[z] = np.append(zones_distros[z], vel[inds])
        
        numBodies += 1
    #######################################################################

    #######################################################################
    df_t = np.arange(From_, To_+tau, tau)
    #######################################################################

    #######################################################################
    # Compute MSD & distFromWalls per Zone
    if ( (do_what == 'distros') or (do_what == 'MSD') or (do_what == 'distFromWalls') ):

        zones_tmp = np.linspace(bottom_CFL, top_CFL, zones_+1)
        MSD_t_avg = []
        distFromWalls_t_avg = []
        for z in range(zones_):
            MSD_t_avg.append([])
            distFromWalls_t_avg.append([])

        for i, t_ in enumerate(df_t):
            for b_ in range(len(Bodies)):
                try:
                    pos = absolute_pos[b_].iloc[i]
                except:
                    continue

                for z in range(zones_):
                    if ( (pos >= zones_tmp[z]) and (pos < zones_tmp[z+1]) ):
                        MSD_t_avg[z].append(MSD[b_].iloc[i])
                        distFromWalls_t_avg[z].append(distFromWalls[b_].iloc[i])
            
            for z in range(zones_):
                # If no particles in the zone, then np.mean returns nan
                zones_MSD[z] = np.append(zones_MSD[z], np.mean(MSD_t_avg[z]))
                MSD_t_avg[z] = []

                zones_distFromWalls[z] = np.append(zones_distFromWalls[z], np.mean(distFromWalls_t_avg[z]))
                distFromWalls_t_avg[z] = []
        
        # Cleaning
        for z in range(zones_):
            if (np.where(np.isnan(zones_MSD[z]))[0].shape[0] != 0):
                zones_MSD[z]           = zones_MSD[z][:np.where(np.isnan(zones_MSD[z]))[0][0]]
            if (np.where(np.isnan(zones_distFromWalls[z]))[0].shape[0] != 0):
                zones_distFromWalls[z] = zones_distFromWalls[z][:np.where(np.isnan(zones_distFromWalls[z]))[0][0]]

        for z in range(zones_):
            
            from scipy import optimize
            # non_linear fitting
            def non_linear_(x, a, b):
                return a*np.power(x, b)
            # linear fitting
            def linear_(x, a, b):
                return a*x + b

            # MSD
            Y = zones_MSD[z]
            X = np.copy(df_t)[:Y.shape[0]]
            X -= From_

            best_vals_non_linear, _ = optimize.curve_fit(non_linear_, X, Y)
            #best_vals_linear    , _ = optimize.curve_fit(linear_    , X, Y)

            zones_MSD[z] = tuple(best_vals_non_linear)

            if (do_what == 'MSD'):
                # Dump data
                #np.savetxt(fName_ + '.csv', np.array(list(zip(X,Y))), delimiter=',')
                plt.plot(X,Y)
                plt.plot(X, non_linear_(X, *best_vals_non_linear), linestyle='--', label='non-linear (a*x^b), params as [a,b] : '+str(best_vals_non_linear))
                #plt.plot(X, linear_(X, *best_vals_linear)        , linestyle='--', label='linear (a*x + b), params as [a,b] : '+str(best_vals_linear))
                plt.legend()
                plt.show()


            # distFromWalls
            Y = zones_distFromWalls[z]
            X = np.copy(df_t)[:Y.shape[0]]
            X -= From_
            
            #best_vals_non_linear, _ = optimize.curve_fit(non_linear_, X, Y)
            best_vals_linear    , _ = optimize.curve_fit(linear_    , X, Y)
            
            zones_distFromWalls[z] = tuple(best_vals_linear)

            if (do_what == 'distFromWalls'):
                # Dump data
                #np.savetxt(fName_ + '.csv', np.array(list(zip(X,Y))), delimiter=',')
                plt.plot(X,Y)
                #plt.plot(X, non_linear_(X, *best_vals_non_linear), linestyle='--', label='non-linear (a*x^b), params as [a,b] : '+str(best_vals_non_linear))
                plt.plot(X, linear_(X, *best_vals_linear)        , linestyle='--', label='linear (a*x + b), params as [a,b] : '+str(best_vals_linear))
                plt.legend()
                plt.show()
    #######################################################################

    #######################################################################
    if (do_what == 'distros'):

        # significance level for p-values
        sign_lvl = 0.1

        # For the PLT random walk simulations
        distros_invECDF = []
        distros_tail    = []
        xmins           = []

        zones_tmp = np.linspace(bottom_CFL, top_CFL, zones_+1)
        for z in range(zones_):
            print("#######################################################################")
            print("Zone ", z)
            print("Limits: (", zones_tmp[z], ",", zones_tmp[z+1], ")")
            print('------------------------------------------------------------')

            data = np.absolute(zones_distros[z])
            print("Mean absolute velocity (current zone) [um/ms]                             : ", np.mean(data))
            print("Diffusion Coefficient (v^2*dt*0.5) [um^2/ms]                              : ", (np.mean(data)**2.)*tau*0.5)
            print("MSD non-linear fitting (a*x^b), params as (a,b) [um^2,ms]                 : ", zones_MSD[z])
            print("Avg Distance from Walls linear fitting (a*x + b), params as (a,b) [um,ms] : ", zones_distFromWalls[z])

            print('------------------------------------------------------------')
            print("Checking for sign.")
            data = zones_distros[z]

            sign_ = np.sign(data)
            positive_ = sign_[sign_ > 0.]
            negative_ = sign_[sign_ < 0.]

            print('Positive velocities (%) : ' , round(positive_.shape[0]/sign_.shape[0], 2) * 100.)
            print('Negative velocities (%) : ' , round(negative_.shape[0]/sign_.shape[0], 2) * 100.)

            print('------------------------------------------------------------')
            print("Checking for normality.")
            
            not_normal = 0
            normal     = 0

            # Shapiro-Wilk Test
            stat, p = stats.shapiro(data)
            if (p > sign_lvl):
                normal += 1
            else:
                not_normal += 1

            # D’Agostino’s K^2 Test
            stat, p = stats.normaltest(data)
            if (p > sign_lvl):
                normal += 1
            else:
                not_normal += 1

            # Anderson-Darling Test
            result = stats.anderson(data)
            for i in range(len(result.critical_values)):
                if result.statistic < result.critical_values[i]:
                    normal += 1
                else:
                    not_normal += 1

            kurt = stats.kurtosis(data)
            print('kurtosis of dataset (whole range, i.e., body & tail) : ', kurt)
            print('Number of successful normality tests                 : ', normal)
            print('Number of failed normality tests                     : ', not_normal)

            print("End of Checking for normality.")

            print('------------------------------------------------------------')
            print("Analyze the tail of the distribution.")
            
            data = np.absolute(zones_distros[z])

            from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter
            data.sort() # in-place sorting
            ecdf = ECDF(data)
            inv_ecdf = monotone_fn_inverter(ecdf, data)
            distros_invECDF.append({'inv_ecdf':inv_ecdf, 'lb':ecdf(np.min(data)), 'ub':ecdf(np.max(data))})
            
            #######################################################################
            tail_P = 0.90 # no need to search the whole domain for the lower bound (x_min). Search from the 90th percentile and above.
            print("Number of samples to do statistics (whole range, i.e., body & tail) : ", data.shape[0])
            print("Number of samples to do statistics (tail-only)                      : ", data[data >= inv_ecdf(tail_P)].shape[0])
            #######################################################################

            print('------------------------------------------------------------')
            # https://en.wikipedia.org/wiki/Heavy-tailed_distribution#Common_heavy-tailed_distributions
            # We focus on fat-tails and more specifically on power laws (see paper for more)
            # heavy-tails term: kept it for legacy reasons
            wikipedia_heavy_tailed_distros = [
                'halfcauchy',
                'burr12', 'burr',
                'pareto',
                'lognorm',
                'weibull_min',
                'fisk',
                'invweibull',
                'levy',
                'invgauss' # see Klaus_2011 (Statistical Analyses Support Power Law Distributions Found in Neuronal Avalanches)
            ]

            handpicked_distros = wikipedia_heavy_tailed_distros + ['expon', 'halfnorm']

            for dist_name in handpicked_distros:
                print(dist_name)
                distro = eval('stats.' + dist_name)

                '''
                if (distro.numargs >= 2):
                    print('Skip distro.')
                    print('Avoid overfitting from distros with multiple parameters (numargs >= 2).')
                    print('------------------------------------------------------------')
                    continue
                '''

                if ( (distro.a < 0.) or (distro.b != np.inf) ):
                    print('Skip distro.')
                    print('Bounds not appropriate.')
                    print('------------------------------------------------------------')
                    continue

                #######################################################################
                # Optimal fitting
                # Computationally expensive part!
                if (dist_name != 'halfnorm'):
                    xmin_optimal = find_xminOpt_distro(data[data >= inv_ecdf(tail_P)], dist_name)
                else:
                    xmin_optimal = 0.
                #######################################################################

                #######################################################################
                # Relaxed fitting based on optimal one
                # When ecdf(xmin_opt) > 95%, it's a good idea to try a relaxed version
                # at ecdf(xmin_opt) ~ 90%
                if (dist_name != 'halfnorm'):
                    # round down
                    tail_i = 0.04
                    xmin_relaxed_lb = inv_ecdf( (int(ecdf(xmin_optimal)*10.) / 10.) - tail_i/2. )
                    xmin_relaxed_ub = inv_ecdf( (int(ecdf(xmin_optimal)*10.) / 10.) + tail_i/2. )
                    # More educated choice of xmin_relaxed
                    xmin_relaxed = find_xminOpt_distro(data[data >= xmin_relaxed_lb], dist_name, xmin_relaxed_ub)
                else:
                    xmin_relaxed = 0.
                #######################################################################

                data_optimal = data[data >= xmin_optimal]
                params_optimal = distro.fit(data_optimal)

                data_relaxed = data[data >= xmin_relaxed]
                params_relaxed = distro.fit(data_relaxed)

                #*** KS-test
                p_val_optimal = stats.kstest(data_optimal, dist_name, params_optimal)[1]
                p_val_relaxed = stats.kstest(data_relaxed, dist_name, params_relaxed)[1]
                #***

                strongly_rejected_opt = False
                negative_d = 'None'
                negative_p = 1.
                for dist_name_ in handpicked_distros:
                    if (dist_name_ == dist_name):
                        continue
                    # Check dist_name vs dist_name_
                    # Which model is better fit
                    LLR, p = LLR_test(data_optimal, dist_name, dist_name_)
                    if ( (LLR < 0.) and (p < negative_p) ):
                        negative_d = dist_name_
                        negative_p = p

                # significance lvl as in Klaus_2011 (Statistical Analyses Support Power Law Distributions Found in Neuronal Avalanches)
                if ( negative_p < 0.01 ):
                    strongly_rejected_opt = True

                print('_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.')
                print('Optimal fitting                             ')
                print('Number of samples xmin_optimal            : ', data_optimal.shape[0])
                print('params_optimal                            : ', params_optimal)
                print('xmin_optimal                              : ', xmin_optimal)
                print('ecdf(xmin_optimal)                        : ', round(ecdf(xmin_optimal)*100, 2), ' (%)')
                print('(p-val) kstest - tail only - xmin_optimal : ', round(p_val_optimal, 2))
                print(' . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .')
                print('strongly_rejected                         : ', 'True' if (strongly_rejected_opt) else 'False')
                print('As good as possible alternative (dist,p)  : ', (negative_d, round(negative_p,5)))

                print('_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.')
                print('Relaxed fitting                             ')
                print('Number of samples xmin_relaxed            : ', data_relaxed.shape[0])
                print('params_relaxed                            : ', params_relaxed)
                print('xmin_relaxed                              : ', xmin_relaxed)
                print('ecdf(xmin_relaxed)                        : ', round(ecdf(xmin_relaxed)*100, 2), ' (%)')
                print('(p-val) kstest - tail only - xmin_relaxed : ', round(p_val_relaxed, 2))

                print(' . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .')
                relaxed_accept = 0
                repeat_ = 2500 # See Clauset_2009 (Power-Law Distributions in Empirical Data)
                for _ in range(repeat_):
                    synthetic_data = inv_ecdf(np.random.uniform(ecdf(np.min(data)), ecdf(np.max(data)), size=data.shape[0]))
                    toCompare_with = inv_ecdf(np.random.uniform(ecdf(np.min(data)), ecdf(np.max(data)), size=data.shape[0]))

                    # 1. optimal model: simulates the observed data with the ecdf up to xmin_optimal and then with the selected distro
                    # 2. relaxed model: simulates the observed data with the ecdf up to xmin_relaxed and then with the selected distro
                    # The reference model is the optimal one.
                    
                    optimal_model = np.copy(synthetic_data)
                    inds = np.where(optimal_model >= xmin_optimal)
                    optimal_model[inds] = distro.rvs(*params_optimal, size=inds[0].shape[0])
                    optimal_model = optimal_model[ (~np.isnan(optimal_model)) & (~np.isinf(optimal_model)) ]
                    optimal_model = optimal_model[ optimal_model < (((top_wall - bottom_wall) - 5.0) / tau) ]
                    D_opt = astats.kuiper_two(toCompare_with, optimal_model)[0]

                    relaxed_model = np.copy(synthetic_data)
                    inds = np.where(relaxed_model >= xmin_relaxed)
                    relaxed_model[inds] = distro.rvs(*params_relaxed, size=inds[0].shape[0])
                    relaxed_model = relaxed_model[ (~np.isnan(relaxed_model)) & (~np.isinf(relaxed_model)) ]
                    relaxed_model = relaxed_model[ relaxed_model < (((top_wall - bottom_wall) - 5.0) / tau) ]
                    D_rel = astats.kuiper_two(toCompare_with, relaxed_model)[0]

                    if (D_rel <= D_opt):
                        relaxed_accept += 1

                p_val_relaxed = round(relaxed_accept/repeat_, 2)
                print('p-val of relaxed model                    : ', p_val_relaxed)

                print('_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.')
                print('One-Zone Simulation for optimal model.')
                # int(4808*0.82): 4808 number of activated PLTs per ul (see Chopard_2017 - A physical description of the adhesion and aggregation of platelets). We deal with 0.82ul -> 4808*0.82
                # tau is the time step of the random walks in ms
                # 820um is the height of Impact-R PLT function analyser (and thus the *0.82)
                PLTs_ = PLTs(int(4808*0.82), tau, 820.0, [{'inv_ecdf':inv_ecdf, 'lb':ecdf(np.min(data)), 'ub':ecdf(np.max(data))}], [{'distro':distro, 'params':params_optimal}], [xmin_optimal])
                try:
                    PLTs_.advance(int(20000/tau))
                    depositedPLTs_opt = int(PLTs_.depositedPLTs()/0.82)
                    MSD_fitting_prms, distFromWalls_prms = PLTs_.meta_quantities()
                except:
                    depositedPLTs_opt = 0
                    MSD_fitting_prms, distFromWalls_prms  = (), ()
                print('deposited PLTs (per uL)                   : ', depositedPLTs_opt)
                print('MSD non-linear fitting [um^2,ms]          : ', MSD_fitting_prms)
                print("Avg Dist Walls linear fitting [um,ms]     : ", distFromWalls_prms)
                
                print('_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.')
                print('One-Zone Simulation for relaxed model.')
                PLTs_ = PLTs(int(4808*0.82), tau, 820.0, [{'inv_ecdf':inv_ecdf, 'lb':ecdf(np.min(data)), 'ub':ecdf(np.max(data))}], [{'distro':distro, 'params':params_relaxed}], [xmin_relaxed])
                try:
                    PLTs_.advance(int(20000/tau))
                    depositedPLTs_rel = int(PLTs_.depositedPLTs()/0.82)
                    MSD_fitting_prms, distFromWalls_prms = PLTs_.meta_quantities()
                except:
                    depositedPLTs_rel = 0
                    MSD_fitting_prms, distFromWalls_prms  = (), ()
                print('deposited PLTs (per uL)                   : ', depositedPLTs_rel)
                print('MSD non-linear fitting [um^2,ms]          : ', MSD_fitting_prms)
                print("Avg Dist Walls linear fitting [um,ms]     : ", distFromWalls_prms)

                print('------------------------------------------------------------')
  
            print("#######################################################################")
    #######################################################################

    #######################################################################
    if (do_what == 'MFP'):
        avg_ = 0.
        for PLT in Bodies:
            avg_ += (abs(integrals_gT[PLT] - integrals_tau[PLT]) / abs(integrals_gT[PLT])) * 100.
        avg_ /= numBodies
        ground_truth_diff.append(avg_)
    #######################################################################


#######################################################################
#######################################################################
#######################################################################
# main
#######################################################################
#######################################################################
#######################################################################

# Command line args

# RBCs || PLTs (we focus on PLTs)
which_bodies = sys.argv[1]

# tau to analyze (in milliseconds)
tau_ = float(sys.argv[2])

# different kind of analysis: (see paper for more info on the different types)
# - MFP: Mean Free Path/Time
# - distros: PLT velocities distributions
# - MSD: Mean Square Displament & Diffusion Coefficient from DNS data
# - distrFromWalls: average distance from walls from DNS data
do_what = sys.argv[3]

# box50: 1 49, The walls of a box with dimensions 50^3 um^3 are located at 1um & 49um from the boundaries of the domain
bottom_wall = float(sys.argv[4])
top_wall    = float(sys.argv[5])

# CFL thickness (in micrometers-um) or zone to exclude to avoid wall effects
CFL_b = float(sys.argv[6])
CFL_t = float(sys.argv[7])

# From/To (in milliseconds), if the DNS was executed for 1000ms physical time, you may want to start the meta-processing at 300ms
From_ = float(sys.argv[8])
To_ = float(sys.argv[9])

# number of zones to analyze velocity distibution (split the domain into zones)
zones_ = int(sys.argv[10])

# Top Boundary: tBB (top Bounce Back) or tCFL (top Trapping in the CFL) || NA
sim_BC_ = sys.argv[11]

# Chanel Length (lateral dimension)
chanel_len = float(sys.argv[12])

# Fluid timestep (us) from DNS
dt_f = float(sys.argv[13])

# Number of threads for xmin_optimal
num_threads = int(sys.argv[14])

#######################################################################

# file name to write the output
fName_ = (
sys.argv[ 1] + '_' + 
sys.argv[ 2] + '_' + 
sys.argv[ 3] + '_' + 
sys.argv[ 4] + '_' + 
sys.argv[ 5] + '_' + 
sys.argv[ 6] + '_' + 
sys.argv[ 7] + '_' + 
sys.argv[ 8] + '_' + 
sys.argv[ 9] + '_' + 
sys.argv[10] + '_' +
sys.argv[11] + '_' +
sys.argv[12] + '_' +
sys.argv[13] + '_' +
sys.argv[14]
)

if (do_what == 'distros'):
    sys.stdout = open(fName_ + '.log', "w")

#######################################################################
# for pandas dataframes
names_   = ['t', 'x', 'y', 'z']
usecols_ = ['t', 'y', 'z']

# in milli-seconds
if (do_what == 'MFP'):
    tau = [0.01, 0.1, 1.0, 10.0]
else:
    tau = [tau_]

# Fluid timestep
dt_f *= 0.001 # in milli-seconds

bottom_CFL = bottom_wall + CFL_b
top_CFL = top_wall - CFL_t
#######################################################################

#######################################################################
# For numerical ordering of files in a folder
numbers = re.compile(r'(\d+)')

# Computation of MFP: Mean Free Path/Time (an approximation at least)
ground_truth = tau[0]
integrals_gT = []
ground_truth_diff = []

if (do_what == 'MFP'):
    groundTruth_data_location = which_bodies + '_tau_' + str(ground_truth) + '/'
    for log in sorted(glob.glob(groundTruth_data_location + '*_ComPos.log'), key=numericalSort):
        df = pd.read_csv(log, delimiter=',', header=None, names=names_, usecols=usecols_)
        df = df.loc[df['t']*dt_f >= From_]
        df = df.loc[df['t']*dt_f <= To_]
        df = df.reset_index(drop=True)
        integrals_gT.append(np.trapz(df['y'], df['t']*dt_f))

for time_window in tau:
    meta_process(time_window)

if (do_what == 'MFP'):
    print(ground_truth_diff)
    plt.plot(range(len(ground_truth_diff)),ground_truth_diff)
    plt.xticks(range(len(ground_truth_diff)))
    plt.savefig(fName_ + '.pdf')
    plt.show()

#######################################################################

if (do_what == 'distros'):
    sys.stdout.close()