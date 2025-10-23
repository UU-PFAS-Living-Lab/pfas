# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 14:26:22 2022

@author: boguo
"""
function [A, B] = ABfunc(Z,T,ws,betas,beta,P,R,Rs,m,cflag)
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

#{
Copyright 2021-2022 Bo Guo (University of Arizona, Email: boguo@arizona.edu).

This file is part of the implementation for the analytical solver presented 
in the article 

Guo, B., Zeng, J., Brusseau, M.L. and Zhang, Y., 2022. 
A screening model for quantifying PFAS leaching in the vadose zone and 
mass discharge to groundwater. Advances in Water Resources, 160, p.104102.

The analytical solver is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details, <http://www.gnu.org/licenses/>.
#}

# number of cells for the numerical integration
n = 1001;       
# tau cannot be zero, so start from a very small number to avoid the boundaries
tau = linspace(1E-6,T-1E-6,n)'; 
Jab = zeros(length(tau),1);
Jba = zeros(length(tau),1);

if cflag == 0
    # volume-averaged concentration
    g = sqrt(P./(pi*beta*R*tau)).*exp(-P*(beta*R*Z-tau).^2./(4*beta*R*tau)) ...
        -1/2*(P/(beta*R)).*exp(P*Z).*erfc(sqrt(P./(4*beta*R*tau)).*(beta*R*Z+tau));
elseif cflag == 1
    # flux-averaged concentration
    g = Z./tau .* sqrt(P*beta*R./(4*pi*tau)) .* exp(-P*((beta*R*Z-tau).^2) ./(4*beta*R*tau));
end

# Approximating the Goldstein's J function
if betas == 1
    Jab = Jab*0 + 1;
    Jba = Jba*0 + 1;
else
    a = ws*tau/(beta*R);
    b = ws*(T-tau)/((1-betas)*(Rs+1));
    for i = 1:n
        if a(i) + b(i) > 10
           Jab(i) = 1/2*erfc(sqrt(a(i)) - sqrt(b(i)) - 1/8/sqrt(a(i)) - 1/8/sqrt(b(i)));
           Jba(i) = 1/2*erfc(sqrt(b(i)) - sqrt(a(i)) - 1/8/sqrt(b(i)) - 1/8/sqrt(a(i))); 
        else
           Iab_sum = 0;
           Iba_sum = 0;
            if a(i) >= b(i)
                for j = 0:m-1
                    Iab_sum = Iab_sum + (b(i)/a(i))^(j/2)*besseli(j,2*sqrt(a(i)*b(i)));
                end
                for j = 1:m
                    Iba_sum = Iba_sum + (b(i)/a(i))^(j/2)*besseli(j,2*sqrt(a(i)*b(i)));
                end
                Jab(i) = exp(-a(i)-b(i))*Iab_sum;
                Jba(i) = 1 - exp(-a(i)-b(i))*Iba_sum;
            else
                for j = 1:m
                    Iab_sum = Iab_sum + (a(i)/b(i))^(j/2)*besseli(j,2*sqrt(a(i)*b(i)));
                end
                for j = 0:m-1
                    Iba_sum = Iba_sum + (a(i)/b(i))^(j/2)*besseli(j,2*sqrt(a(i)*b(i)));
                end
                Jab(i) = 1 - exp(-a(i)-b(i))*Iab_sum;
                Jba(i) = exp(-a(i)-b(i))*Iba_sum;
            end
        end
    end
end
A = trapz(tau,g.*Jab);
B = trapz(tau,g.*(1-Jba));

end
  