import numpy as np
from scipy.special import iv
from scipy.special import erfc
from pfas_leach_screening import utils
from pfas_leach_screening.solvers import equilibrium_solver, kinetic_solver

def analytical_soln(t,z,t0,pfas_tot,Ci,L,v,theta,rhob,D,Kd,alphas,Fs,R,Rs,Raw,spaflag,cflag):
    # Compute the aqueous concentration (C1), solid-phase adsorption in the kinetic 
    # sorption domain (C2), and the total concentration (C_tot)
    
    # Processes & conditions included: advection, dispersion, kinetic SPA,
    # equilibrium AWIA, and nonzero spatially variable initial condition
    
    # output parameters
    # C1                    aqueous concentration in space (micro mol/cm^3)
    #                       NB: C1 here is the C in Guo et al (2022) AWR.         
    # C2                    kinetic solid-phase adsorption (micro mol/g)
    #                       NB: C2 here is the Cs,2 in Guo et al (2022) AWR.  
    # C_tot                 total concentration per bulk volume in space (micro mol/cm^3) 
    
    # input parameters
    # t         time vector (second)
    #           NB: t can be a vector representing different points in time or
    #               a single value representing one point in time
    # z         length vector (cm)
    #           NB: z can be a vector representing different points in space or
    #               a single value representing one point in space (e.g.,outlet)
    # t0        duration of active contamination (second)
    #           NB: set t0 to zero if the simulation does not include an active
    #               contamination period
    # pfas_tot  total (single-component) PFAS mass that will be released (mg)
    #           NB: set pfas_tot to zero if the simulation does not include an
    #               active contamination period
    # Ci        initial aqueous concentration in space (micro mol/cm^3)
    # L         length of the domain (cm)
    # v         interstitial pore velocity (cm/s)
    #           NB: v = q/theta
    # theta     water content (-)
    # rhob      bulk density (g/cm^3)
    # D         dispersion coefficient (cm^2/s)
    # Kd        solid-phase adsorption coefficient (cm^3/g)
    # alphas    first-order rate constant for kinetic SPA (1/s)
    # Fs        fraction of sorbent for which sorption is instantaneous (-)
    # R         total retardation factor (-)
    # Rs        retardation factor associated with SPA
    # Raw       retardation factor associated with AWIA
    # spaflag   A flag to denote equilibrium or kinetic SPA
    #           NB: 0 - SPA is equilibrium, 1 - SPA is kinetic
    # cflag     A flag to denote volume-averaged or flux-averaged concentration
    #           NB: 0 - concentration is volume-averaged, 1 - concentration is flux-averaged
    #           

    # Copyright 2021-2022 Bo Guo (University of Arizona, Email: boguo@arizona.edu or guobo07@gmail.com).

    # This file is part of the implementation for the PFAS-LEACH-Screening analytical 
    # solver presented in the article Guo et al. (2022) AWR

    # Guo, B., Zeng, J., Brusseau, M.L. and Zhang, Y., 2022. 
    # A screening model for quantifying PFAS leaching in the vadose zone and 
    # mass discharge to groundwater. Advances in Water Resources, 160, p.104102.

    # Development of PFAS-LEACH-Screening is supported by the ESTCP Project ER21-5041. 
    # Project Webpage: https://www.serdp-estcp.org/Program-Areas/Environmental-Restoration/ER21-5041

    # The PFAS-LEACH-Screening analytical solver is distributed in the hope that 
    # it will be useful, but WITHOUT ANY WARRANTY without even the implied warranty 
    # of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    # GNU General Public License for more details, <http://www.gnu.org/licenses/>.
    
    # Compute releasing contaminant concentration (micro mol/cm^3)
    if t0 > 0:
        # if an active contamination period is considered
        C10 = pfas_tot / (t0*v*theta)
    else:
        # if the simulation does not include an active contamination period
        C10 = 0
    
    #Compute dimensionless variables
    Z = z/L            # dimensionless length
    T = v*t/L          # dimensionless time
    T0 = v*t0/L        # dimensionless duration of contamination period
    P = v*L/D          # Peclect number
    
    # Define dimensionless variables for kinetic SPA
    betas = (1+Fs*Rs)/(1+Rs)
    ws = alphas*(1-betas)*(1+Rs)*L/v   # Damköhler number
    beta = (betas*(1+Rs)+Raw) / (1+Rs+Raw)
    
    # Initialize solutions for the aqueous concentration for BVP and IVP problems
    C1_bvp = np.zeros((len(Z),len(T)))                
    C1_ivp = np.zeros((len(Z),len(T)))
    # Initialize solutions for adsorbed concentration at the kinetic sorption domain
    C2_bvp = np.zeros((len(Z),len(T)))
    C2_ivp = np.zeros((len(Z),len(T)))
    C2 = None #TODO
    # SPA is equilibrium
    if spaflag == 0:
        C1, C_tot = equilibrium_solver(R, Z, T, P, T0, C10, Ci, theta)
        # Solution for the boundary value problem
        # Define the solution for a constant boundary condition as a function
        #eqbvpfunc = lambda T: 0.5 * erfc((R*Z-T)/(2*(T*R/P)**(1/2))) + ((T*P)/(np.pi*R))**(1/2) * np.exp(-(R*Z-T)**2/(4*T*R/P)) - 1/2 * (1 + P*Z + P*T/R) * np.exp(P*Z) * erfc((R*Z + T)/(2*(T*R/P)**(1/2)))
        #for i in range(len(T)):
        #    if T[i] <= T0:
        #        C1_bvp[:,i] = C10 * eqbvpfunc(T[i])
        #    else:
        #        C1_bvp[:,i] = C10 * eqbvpfunc(T[i]) - C10 * eqbvpfunc(T[i]-T0)
        #if max(Ci) != 0:
        #    # Solution for the initial value problem
        #    for i in range(len(T)):
        #        for j in range(len(Z)):
        #            kesi = np.linspace(0,1,len(Ci))       
        #            eqivpfunc = lambda Z, T: (np.exp(-(R*Z-R*kesi-T)**2/(4*T*R/P)) + np.exp(-P*kesi - (R*Z+R*kesi-T)**2/(4*T*R/P)))/(2*np.sqrt(np.pi*T/P/R)) - P/2*np.exp(P*Z)*erfc((R*Z+R*kesi+T)/(2*np.sqrt(T*R/P)))            
        #            C1_ivp[j,i] = np.trapz(eqivpfunc(Z[j],T[i])*Ci,kesi)
    # SPA is kinetic
    elif spaflag == 1:
        C1, C2, C_tot = kinetic_solver(R, Z, T, P, T0, C10, Ci, ws, betas, beta, cflag, Rs, Fs, Kd, theta, rhob)
# =============================================================================
#         m = 30 # number of modified bessel function terms used   
#         for i in range(len(Z)):
#             for j in range(len(T)):
#                 # Solution for the boundary value problem
#                 if T[j] <= T0:
#                     C1_bvp[i,j], C2_bvp[i,j] = utils.ABfunc(Z[i],T[j],ws,betas,beta,P,R,Rs,m,cflag)
#                 elif T[j] > T0:
#                     C1_bvp[i,j], C2_bvp[i,j] = utils.ABfunc(Z[i],T[j],ws,betas,beta,P,R,Rs,m,cflag)
#                     A, B = utils.ABfunc(Z[i],T[j]-T0,ws,betas,beta,P,R,Rs,m,cflag)
#                     C1_bvp[i,j] = C1_bvp[i,j] - A
#                     C2_bvp[i,j] = C2_bvp[i,j] - B
#                 if max(Ci) != 0:
#                     # Solution for the initial value problem
#                     kesi = np.linspace(0,1,len(Ci))
#                     tau = np.linspace(0,T[j],100)
#                     neqivpfunc = lambda Z, T: (np.exp(-P*beta*R*(Z-kesi-T/(beta*R))**2/(4*T)) + np.exp(-kesi*P -P*beta*R*(Z+kesi-T/(beta*R))**2/(4*T)))/(2*np.sqrt(np.pi*T/(beta*R*P))) - P/2*np.exp(P*Z)*erfc((Z+kesi+T/(beta*R))/(2*np.sqrt(T/(beta*R)/P)))
#                     
#                     Hfunc = lambda T, tau: Rs*(1-Fs)/(beta*R)*np.exp(-ws*(T-tau)/(1-betas)/(1+Rs) -ws*tau*(1-Fs)*Rs/(1-betas)/(beta*R)/(1+Rs)) * (iv(0,2*ws/(1-betas)/(1+Rs)*np.sqrt(Rs*(1-Fs)*(T-tau)*tau)/(beta*R)) +iv(1,2*ws/(1-betas)/(1+Rs)*np.sqrt(Rs*(1-Fs)*(T-tau)*tau)/(beta*R))*tau/np.sqrt(Rs*(1-Fs)*(T-tau)*tau/(beta*R)))
#                     
#                     Hs2func = lambda T, tau: np.exp(-ws*(T-tau)/(1-betas)/(1+Rs) -ws*tau*(1-Fs)*Rs/(1-betas)/(beta*R)/(1+Rs)) * (iv(0,2*ws/(1-betas)/(1+Rs)*np.sqrt(Rs*(1-Fs)*(T-tau)*tau)/(beta*R)) + np.sqrt(Rs*(1-Fs)*(T-tau)/(beta*R)/tau)*iv(1,2*ws/(1-betas)/(1+Rs)*np.sqrt(Rs*(1-Fs)*(T-tau)*tau)/(beta*R)))
#                     
#                     GfuncT = np.trapz(neqivpfunc(Z[i],T[j])*Ci,kesi)
#                     if betas == 1:
#                         C1_ivp[i,j] = GfuncT
#                     else:
#                         C1_ivp[i,j] = np.exp(-ws*T[j]*(1-Fs)*Rs/(1-betas)/(beta*R)/(1+Rs)) * GfuncT
#                         C2_ivp[i,j] = (1-Fs)*Kd*Ci[i]*np.exp(-ws*T[j]/(1-betas)/(1+Rs))
#                         Gfunctau = np.zeros((len(tau),1))
#                         for k in range(1,len(tau)-1):
#                             Gfunctau[k] = np.trapz(neqivpfunc(Z[i],tau[k])*Ci,kesi)
#                         C1_ivp[i,j] = C1_ivp[i,j] + ws/(1-betas)/(1+Rs) * np.trapz(Hfunc(T[j],tau[1:-1])* Gfunctau[1:-1],tau[1:-1])
#                         C2_ivp[i,j] = C2_ivp[i,j] + ws/(1-betas)/(1+Rs) * (1-Fs)*Kd* np.trapz(Hs2func(T[j],tau[1:-1])* Gfunctau[1:-1],tau[1:-1])
#         # Convert dimensionless C1_bvp and C2_bvp to original dimensions
#         C1_bvp = C10*C1_bvp
#         C2_bvp = (1-Fs)*Kd*C10*C2_bvp
# =============================================================================
# =============================================================================
#     # Combine solutions for BVP and IVP
#     C1 = C1_bvp + C1_ivp
#     C2 = C2_bvp + C2_ivp
#     # Compute the total concentration per bulk volume
#     if spaflag == 0:
#         C_tot = C1*R*theta + rhob*C2
#     elif spaflag == 1:
#         C_tot = C1*beta*R*theta + rhob*C2  
# =============================================================================

    return C1, C2, C_tot
