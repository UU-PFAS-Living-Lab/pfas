import numpy as np
from scipy.special import iv
from scipy.special import erfc

def ABfunc(Z,T,ws,betas,beta,P,R,Rs,m,cflag):
    # Compute the A and B functions in Eqs (16-17)
    
    # output parameters
    # A         Eq (18) of Guo et al (2022) AWR
    # B         Eq (19) of Guo et al (2022) AWR
    
    # input parameters
    # Z         dimensionless length (Z = z/L)
    # T         dimensionless time (T = v*t/L)
    # ws        ws = alphas*(1-betas)*(1+Rs)*L/v (Damköhler number)
    # betas     betas = (1+Fs*Rs)/(1+Rs)
    # beta      beta = (betas*(1+Rs)+Raw)/(1+Rs+Raw)
    # P         P = v*L/D (Peclect number)
    # R         total retardation factor (-)
    # Rs        retardation factor associated with SPA
    # m         number of modified bessel function terms used
    # cflag     A flag to denote volume-averaged or flux-averaged concentration
    #           NB: 0 - concentration is volume-averaged, 1 - concentration is flux-averaged
    
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


    # number of cells for the numerical integration
    n = 1001       
    # tau cannot be zero, so start from a very small number to avoid the boundaries
    tau = np.linspace(1E-6,T-1E-6,n) 
    Jab = np.zeros(len(tau))
    Jba = np.zeros(len(tau))
    
    if cflag == 0:
        # volume-averaged concentration
        g = np.sqrt(P/(np.pi*beta*R*tau))*np.exp(-P*(beta*R*Z-tau)**2/(4*beta*R*tau)) -1/2*(P/(beta*R))*np.exp(P*Z)*erfc(np.sqrt(P/(4*beta*R*tau))*(beta*R*Z+tau))
    elif cflag == 1:
        # flux-averaged concentration
        g = Z/tau * np.sqrt(P*beta*R/(4*np.pi*tau)) * np.exp(-P*((beta*R*Z-tau)**2) /(4*beta*R*tau))
    
    # Approximating the Goldstein's J function
    if betas == 1:
        Jab = Jab*0 + 1
        Jba = Jba*0 + 1
    else:
        a = ws*tau/(beta*R)
        b = ws*(T-tau)/((1-betas)*(Rs+1))
        for i in range(n):
            if a[i] + b[i] > 10:
               Jab[i] = 1/2*erfc(np.sqrt(a[i]) - np.sqrt(b[i]) - 1/8/np.sqrt(a[i]) - 1/8/np.sqrt(b[i]))
               Jba[i] = 1/2*erfc(np.sqrt(b[i]) - np.sqrt(a[i]) - 1/8/np.sqrt(b[i]) - 1/8/np.sqrt(a[i])) 
            else:
               Iab_sum = 0
               Iba_sum = 0
               if a[i] >= b[i]:
                   for j in range(m):
                       Iab_sum = Iab_sum + (b[i]/a[i])**(j/2.0)*iv(j,2*np.sqrt(a[i]*b[i]))
                   for j in range(1,m+1):
                       Iba_sum = Iba_sum + (b[i]/a[i])**(j/2.0)*iv(j,2*np.sqrt(a[i]*b[i]))
                   Jab[i] = np.exp(-a[i]-b[i])*Iab_sum
                   Jba[i] = 1 - np.exp(-a[i]-b[i])*Iba_sum
               else:
                   for j in range(1,m+1):
                       Iab_sum = Iab_sum + (a[i]/b[i])**(j/2.0)*iv(j,2*np.sqrt(a[i]*b[i]))
                   for j in range(m):
                       Iba_sum = Iba_sum + (a[i]/b[i])**(j/2.0)*iv(j,2*np.sqrt(a[i]*b[i]))
                   Jab[i] = 1 - np.exp(-a[i]-b[i])*Iab_sum
                   Jba[i] = np.exp(-a[i]-b[i])*Iba_sum
    A = np.trapz(g*Jab,tau)
    B = np.trapz(g*(1-Jba),tau)
    return A, B

def Aaw_func_thermo(sigma0,poro,alpha,n,th,thr,ths,sf):
    #Computing air-water interfacial area using the thermodynamic approach
    
    # Aaw       air-water interfacial area (cm^2/cm^3)
    # sigma0    surface tension (dyn/cm)
    # poro      porosity (-)
    # alphal    V-G parameter (cm^-1)
    # n         V-G parameter (-)
    # th        water content (-)
    # thr       residual water content (-)
    # ths       saturated water content (-)
    
    # rhow      water density (kg/m^3)
    # g         gravity acceleration (m/s^2)
    # Pc        capillary pressure
    # sf        scaling factor to correct the thermodynamic-based Aaw (-)
    
    rhow = 1000
    g = 9.81
    m = 1 - 1/n
    Sr = thr/ths
    Sw = np.linspace(th/ths,1,1000)
    Pc = lambda Sw: (((1-Sr)/(Sw-Sr))**(1/m) - 1)**(1/n)/alpha/100*rhow*g
    
    Aaw = 10*np.trapz(poro/sigma0*Pc(Sw),Sw)
    
    Aaw = Aaw*sf
    
    return Aaw  
def Aaw_func_tracer(sw,x2,x1,x0):
    #Computing air-water interfacial area using the thermodynamic approach
    
    # Aaw           air-water interfacial area (cm^2/cm^3)
    # sw            water saturation
    # x2, x1, x0    polynomial fitting coefficients
    Aaw = x2*sw**2 + x1*sw + x0
    return Aaw