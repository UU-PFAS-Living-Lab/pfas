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

import numpy as np
from pfas_leach_screening.analytical_soln import analytical_soln
from pfas_leach_screening.utils import Aaw_func_thermo 
from pfas_leach_screening.utils import Aaw_func_tracer
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Example simulation of PFOA release and contamination in Accusand under AZ climate

# Flag for the type of air-water interfacial area
# Aawflag = 'tracer-based'          # "tracer-based" or "thermodynamic-based". See Guo et al (2022) AWR 
Aawflag = 'tracer-based'

# time units
hour = 3600
day = 24*hour
year = 365*day

# PFAS properties
M = 414.07                                  # Molecular weight of PFOS [g/mol]
b = 0.19                                    # [-]
a = 62.11/M                                 # mg/L
sigma0 = 71                                 # dyn/cm
C0 = 0.9                                    # mg/L
C_rep = 0.016                               # representative aqueous concentration [mg/L]
                                            # NB: C_rep is not a required parameter. It is only needed
                                            #     if solid-phase and air-water interfacial adsorption are nonlinear

# PFAS solid-phase adsorption parameters
Kf = 0.1                                    # Freundlich Kf [mg/kg / (mg/L)^N]
n_fr = 0.87                                 # Freundlich power
Fs = 0.4                                    # fraction of instantaneous sites
alphas = 5.9/hour                           # first-order rate constant for kinetic adsorption

# total simulation time
tf = 80*year

Rg = 8.31                                   # gas constant [J/K/mol]
Tp = 293.15                                 # temperature [K]
L = 400                                     # vadose zone depth [cm]

# hydraulic parameters
thetar = 0.015                              # Residual water content [cm^3/cm^3]
alpha = 0.04479                             # VG parameter
n_vg = 4                                    # VG parameter
m = 1 - 1/n_vg                              # VG parameter
# coefficients to compute tracer-measured air-water interfacial area
x0 = 633.96                                 # [-]               -> Aaw fitting parameter
x1 = -1182.5                                # [-]               -> Aaw fitting parameter
x2 = 548.54                                 # [-]               -> Aaw fitting parameter
# initial guess of effective saturation
se0 = 0.1                                   
rhob = 1.65                                 # bulk density [g/cm^3] 
poro = 0.294                                # porosity [cm^3/cm^3]
thetas = 0.294                              # saturated water saturation [cm^3/cm^3]
Ks = 2.0964E-2                              # saturated hydraulic conductivity
alphaL = 100*0.83*(np.log10(L/100))**2.414  # dispersivity [cm]
q = 1916.2/tf                               # water infiltration rate        

Kd = Kf*C_rep**(n_fr-1)                     # solid-phase adsorption coefficient [cm^3/g]
Kaw = 0.1*sigma0*b/(Rg*Tp*(a+C_rep/M))      # air-water interfacial coefficient [cm^3/cm^2]
                                            # NB: if C_rep is not known, set C_rep = 0
                                            
# solve for effective saturation corresponding to the infiltration rate
kr = q/Ks                                   # relative permeability
relperm = lambda se: se**0.5 * (1 - (1 - se**(1/m))**m)**2 - kr
se = fsolve(relperm,se0)
theta = se[0] * (poro - thetar) + thetar    # water content [cm^3/cm^3]
sw = theta/poro                             # water saturation [cm^3/cm^3]
v = q/theta                                 # pore velocity [cm/s]
D = v*alphaL                                # dispersion coefficient [cm^2/s]
if Aawflag == 'tracer-based':
    Aaw = Aaw_func_tracer(sw,x2,x1,x0)      # air-water interfacial area [cm^2/cm^3]
elif Aawflag == 'thermodynamic-based':
    sf = 4.15                               # scaling factor to correct the thermodyanmic-based Aaw (see Guo et al., 2022) [-]
    Aaw = Aaw_func_thermo(sigma0,poro,alpha,n_vg,theta,thetar,thetas,sf)
t0 = 30*year                                # Contamination period [s].  Set it to zero if contamination had occured.

Rs = rhob*Kd/theta                          # retardation factor assuming no AWIA
Raw = Aaw*Kaw/theta                         # retardation factor by AWIA
R = 1+Rs+Raw                                # retardation factor

dz = 1                                      # Spatial resolution [cm]
zs = np.linspace(dz/2.0,L-dz/2.0,int(L/dz)) # Spatial location of interest. zs needs to be an array. If a single location z0 is of interest, then set zs = np.array([z0])                        
ts = np.array([5*year,20*year])             # Time of interest. ts needs to be an array. If a single time point t0 is of interest, then set ts = np.array([t0])
pfas_tot = C0/M*0.55*2/24*t0/(10*day)       # total (single-component) PFAS released during t0 [micro mol per unit area in cm^2]                                   


spaflag = 0 # 0 - SPA is equilibrium 1 - SPA is kinetic
cflag = 0 # 0 - volume-averaged 1 - flux-averaged
Ci = np.zeros(len(zs))                      # initial PFAS aqueous concentration. This can be nonzero and arbitrary.

# Employ analytical solution
# C1                    aqueous concentration in space [micro mol/cm^3]
#                       NB: C1 here is the C in Guo et al (2022) AWR.         
# C2                    kinetic solid-phase adsorption [micro mol/g]
#                       NB: C2 here is the Cs,2 in Guo et al (2022) AWR.  
# C_tot                 total concentration per bulk volume in space [micro mol/cm^3] 

# NB: 	When either ts or zs is a vector (i.e., length > 1), the output parameters C1, C2, and Ctot will be vectors. 
#		When both ts and zs are vectors, the output parameters C1, C2, and Ctot will be matrices.

C1, C2, Ctot = analytical_soln(ts,zs,t0,pfas_tot,Ci,L,v,theta,rhob,D,Kd,alphas,Fs,R,Rs,Raw,spaflag,cflag)
           
# plot the results
fig, ax = plt.subplots()
ax.plot(Ctot[:,0]*M,zs,label='5 yrs');
ax.plot(Ctot[:,1]*M,zs,label='20 yrs');
ax.invert_yaxis()
ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
ax.set_xlabel('Total PFOA concentration (mg / dm^3)')
ax.set_ylabel('Depth (cm)')
ax.legend()