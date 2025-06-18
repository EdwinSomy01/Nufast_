import numpy as np
import matplotlib.pyplot as plt



eVsqkm_to_GeV_over4 = 1e-9 / 1.97327e-7 * 1e3 / 4;
YerhoE2a = 1.52e-4;

def Probability_Matter_LBL(s12sq,  s13sq,  s23sq, delta, Dmsq21,Dmsq31,  L, E,  rho, Ye, N_Newton, *probs_returned):
    # --------------------------------------------------------------------- #
    # First calculate useful simple functions of the oscillation parameters #
    # --------------------------------------------------------------------- #
    c13sq = 1 - s13sq 
    
    # Ueisq's
    Ue2sq = c13sq * s12sq 
    Ue3sq = s13sq 
    
    # Umisq's, Utisq's and Jvac	 
    Um3sq = c13sq * s23sq 
    # Um2sq and Ut2sq are used here as temporary variables, will be properly defined later	 
    Ut2sq = s13sq * s12sq * s23sq 
    Um2sq = (1 - s12sq) * (1 - s23sq) 
    
    Jrr = np.sqrt(Um2sq * Ut2sq) 
    sind = np.sin(delta) 
    cosd = np.cos(delta) 
    
    Um2sq = Um2sq + Ut2sq - 2 * Jrr * cosd 
    Jmatter = 8 * Jrr * c13sq * sind 
    Amatter = Ye * rho * E * YerhoE2a 
    Dmsqee = Dmsq31 - s12sq * Dmsq21 
    
    # calculate A, B, C, See, Tee, and part of Tmm
    A = Dmsq21 + Dmsq31  # temporary variable
    See = A - Dmsq21 * Ue2sq - Dmsq31 * Ue3sq 
    Tmm = Dmsq21 * Dmsq31  # unp.sing Tmm as a temporary variable	  
    Tee = Tmm * (1 - Ue3sq - Ue2sq) 
    C = Amatter * Tee 
    A = A + Amatter 
    
    # ---------------------------------- #
    # Get lambda3 from lambda+ of MP/DMP #
    # ---------------------------------- #
    xmat = Amatter / Dmsqee 
    tmp = 1 - xmat 
    lambda3 = Dmsq31 + 0.5 * Dmsqee * (xmat - 1 + np.sqrt(tmp * tmp + 4 * s13sq * xmat)) 
    
    # ---------------------------------------------------------------------------- #
    # Newton iterations to improve lambda3 arbitrarily, if needed, (B needed here) #
    # ---------------------------------------------------------------------------- #
    B = Tmm + Amatter * See  # B is only needed for N_Newton >= 1
    for i in range( N_Newton):
            lambda3 = (lambda3 * lambda3 * (lambda3 + lambda3 - A) + C) / (lambda3 * (2 * (lambda3 - A) + lambda3) + B)  # this strange form prefers additions to multiplications
    
    # ------------------- #
    # Get  Delta lambda's #
    # ------------------- #
    tmp = A - lambda3 
    Dlambda21 = np.sqrt(tmp * tmp - 4 * C / lambda3) 
    lambda2 = 0.5 * (A - lambda3 + Dlambda21) 
    Dlambda32 = lambda3 - lambda2 
    Dlambda31 = Dlambda32 + Dlambda21 
    
    # ----------------------- #
    # Use Rosetta for Veisq's #
    # ----------------------- #
    # denominators	  
    PiDlambdaInv = 1 / (Dlambda31 * Dlambda32 * Dlambda21) 
    Xp3 = PiDlambdaInv * Dlambda21 
    Xp2 = -PiDlambdaInv * Dlambda31 
    
    # numerators
    Ue3sq = (lambda3 * (lambda3 - See) + Tee) * Xp3 
    Ue2sq = (lambda2 * (lambda2 - See) + Tee) * Xp2 
    
    Smm = A - Dmsq21 * Um2sq - Dmsq31 * Um3sq 
    Tmm = Tmm * (1 - Um3sq - Um2sq) + Amatter * (See + Smm - A) 
    
    Um3sq = (lambda3 * (lambda3 - Smm) + Tmm) * Xp3 
    Um2sq = (lambda2 * (lambda2 - Smm) + Tmm) * Xp2 
    
    # ------------- #
    # Use NHS for J #
    # ------------- #
    Jmatter = Jmatter * Dmsq21 * Dmsq31 * (Dmsq31 - Dmsq21) * PiDlambdaInv 
    
    # ----------------------- #
    # Get all elements of Usq #
    # ----------------------- #
    Ue1sq = 1 - Ue3sq - Ue2sq 
    Um1sq = 1 - Um3sq - Um2sq 
    
    Ut3sq = 1 - Um3sq - Ue3sq 
    Ut2sq = 1 - Um2sq - Ue2sq 
    Ut1sq = 1 - Um1sq - Ue1sq 
    
    # ----------------------- #
    # Get the kinematic terms #
    # ----------------------- #
    Lover4E = eVsqkm_to_GeV_over4 * L / E 
    
    D21 = Dlambda21 * Lover4E 
    D32 = Dlambda32 * Lover4E 
      
    sinD21 = np.sin(D21) 
    sinD31 = np.sin(D32 + D21) 
    sinD32 = np.sin(D32) 
    
    tp_sin = sinD21 * sinD31 * sinD32 
    
    sinsqD21_2 = 2 * sinD21 * sinD21 
    sinsqD31_2 = 2 * sinD31 * sinD31 
    sinsqD32_2 = 2 * sinD32 * sinD32 
    
    # ------------------------------------------------------------------- #
    # Calculate the three necessary probabilities, separating CPC and CPV #
    # ------------------------------------------------------------------- #
    Pme_CPC = (Ut3sq - Um2sq * Ue1sq - Um1sq * Ue2sq) * sinsqD21_2 + (Ut2sq - Um3sq * Ue1sq - Um1sq * Ue3sq) * sinsqD31_2 + (Ut1sq - Um3sq * Ue2sq - Um2sq * Ue3sq)*sinsqD32_2 
    Pme_CPV = -Jmatter * tp_sin 
    
    Pmm = 1 - 2 * (Um2sq * Um1sq *sinsqD21_2 + Um3sq * Um1sq * sinsqD31_2 + Um3sq * Um2sq * sinsqD32_2) 
    
    Pee = 1 - 2 * (Ue2sq * Ue1sq * sinsqD21_2 + Ue3sq * Ue1sq * sinsqD31_2 + Ue3sq * Ue2sq * sinsqD32_2) 
    
    
    # probs_returned[0][0] = Pee 														# Pee
    # probs_returned[0][1] = Pme_CPC - Pme_CPV 										# Pem
    # probs_returned[0][2] = 1 - Pee - probs_returned[0][1]   						# Pet
    
    # probs_returned[1][0] = Pme_CPC + Pme_CPV 										# Pme
    # probs_returned[1][1] = Pmm 														# Pmm
    # probs_returned[1][2] = 1 - probs_returned[1][0] - Pmm 						# Pmt
    
    # probs_returned[2][0] = 1 - Pee - probs_returned[1][0] 						# Pte
    # probs_returned[2][1] = 1 - probs_returned[0][1] - Pmm 						# Ptm
    # probs_returned[2][2] = 1 - probs_returned[0][2] - probs_returned[1][2] 	# Ptt


    Pee= Pee 														# Pee
    Pem = Pme_CPC - Pme_CPV 										# Pem
    Pet = 1 - Pee - Pem  						# Pet
    
    Pme = Pme_CPC + Pme_CPV 										
    Pmm = Pmm 												
    Pmt= 1 - Pme - Pmm 						
    
    Pte = 1 - Pee - Pme					
    Ptm = 1 - Pem - Pmm 						
    Ptt= 1 - Pet - Pmt 
      
    return {
        ('e', 'e'): Pee,   ('e', 'm'): Pem,   ('e', 't'): Pet,
        ('m', 'e'): Pme,  ('m', 'm'): Pmm,  ('m', 't'): Pmt,
        ('t', 'e'): Pte, ('t', 'm'): Ptm, ('t', 't'): Ptt
            }
    
