def kinetic_solver():
  """
  Kinetic solver and parameters.
  """
  # Solution for the boundary value problem
  # Define the solution for a constant boundary condition as a function
  eqbvpfunc = lambda T: 0.5 * erfc((R*Z-T)/(2*(T*R/P)**(1/2))) + ((T*P)/(np.pi*R))**(1/2) * np.exp(-(R*Z-T)**2/(4*T*R/P)) - 1/2 * (1 + P*Z + P*T/R) * np.exp(P*Z) * erfc((R*Z + T)/(2*(T*R/P)**(1/2)))
  for i in range(len(T)):
    if T[i] <= T0:
      C1_bvp[:,i] = C10 * eqbvpfunc(T[i])
    else:
      C1_bvp[:,i] = C10 * eqbvpfunc(T[i]) - C10 * eqbvpfunc(T[i]-T0)
    if max(Ci) != 0:
    # Solution for the initial value problem
      for i in range(len(T)):
        for j in range(len(Z)):
          kesi = np.linspace(0,1,len(Ci))       
          eqivpfunc = lambda Z, T: (np.exp(-(R*Z-R*kesi-T)**2/(4*T*R/P)) + np.exp(-P*kesi - (R*Z+R*kesi-T)**2/(4*T*R/P)))/(2*np.sqrt(np.pi*T/P/R)) - P/2*np.exp(P*Z)*erfc((R*Z+R*kesi+T)/(2*np.sqrt(T*R/P)))            
          C1_ivp[j,i] = np.trapz(eqivpfunc(Z[j],T[i])*Ci,kesi)
  return

def equilibrium_solver():
  """
  Equilibrium solver and its parameters.
  """
  print("TODO")
