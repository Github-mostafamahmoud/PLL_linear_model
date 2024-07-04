import matplotlib.pyplot as plt
import numpy as np
import control as ct 
s = ct.TransferFunction.s
#################################################
wc=2*np.pi*2*10**6       
start_frequency=2*np.pi*12E3
end_frequency=2*np.pi*20E6
#####################################
# variables
I_p= 1*10**-3                 
x=7.595
Kvco= 2*np.pi*90*10**6       
N= 21
K = (Kvco*I_p)/(2*np.pi*N)
P=6
R=2
fo = (320*10**6)*N/(R*P)
R_1 = wc/(K*(1-1/(x**2)))                         
wz=wc/x 
wp=wc*x

c_2=((wz * I_p * Kvco )/( wp * 2 * np.pi * wc**2 * N ))*np.sqrt((1+(wc/wz)**2)/(1+(wc/wp)**2))            
c_1= c_2*( (x**2 )- 1)              
            
c_eq = (c_1*c_2)/(c_1+c_2)

taw_1 = (R_1 * c_1)
taw_2 = taw_1 * (c_2 / (c_1 + c_2) )
###################################################
#Loop gain & open loop gain
TS = (I_p / (2 * np.pi)) * ((R_1 * c_1 * s + 1) / (R_1 * c_eq * s + 1)) * (1 / (s * (c_1 + c_2))) * ((Kvco) / (N * s))
if 1:print('TS(S)= ',TS)
A = TS 
if 1:print('A(S)= ',A)
if 1:plt.figure(1)
if 1:(mag_TS, phase ,w) = ct.bode(TS,Hz=True,dB=True,plot=True,margins=True )
if 0:      #Change to 1 to plot
    plt.figure(2)
    plt.semilogx(w/(2*np.pi), 20*np.log10(mag_TS), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Open Loop Transfer Function (dB)")
##################################3
#Closed loop response
B=(N) * (TS / (1 + TS))
if 1:print('B(S)= ',B)
if 1:plt.figure()
if 1:(magnitudeB, phase ,w) = ct.bode(B,Hz=True,dB=True,plot=True)
if 0:      #Change to 1 to plot
    plt.figure(3)
    plt.semilogx(w/(2*np.pi), 20*np.log10(magnitudeB), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Closed Loop Transfer Function (dB)")

################################################################################

        #VCO noise 
#Transfer Function
Tn_vco=(1 / ((1 + TS)*P))     
if 0:print('Tn_vco(S)= ',Tn_vco)
if 1:plt.figure()  
if 1:(mag_vco, phase ,w) = ct.bode(Tn_vco,Hz=True,dB=True,plot= False,omega_limits=(start_frequency, end_frequency))
f=w/(2*np.pi)
if 1:plt.figure() 
if 1:      #Change to 1 to plot
    plt.semilogx(f, 20*np.log10(mag_vco), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("VCO Phase-Noise Transfer Function")

#  VCO Phase Noise:
def vco_noise(f):
    vco_flicker_noise = -30*np.log10(f) + 52     #offset(52)=(-30*-3)-38 . 1KHz = 3 decade ,-30 --> slope 
    vco_white_noise = -20 *np.log10(f) + 5       #offset(5)=(-20*-6)-115 . 1MHz = 6 decade, -20 --> slope
    vco_noise_floor = -145                       #Noise floor of vco
    vco_pn=10**(vco_flicker_noise/10) + 10**(vco_noise_floor/10) + 10**(vco_white_noise/10)
    return vco_pn

if 1:      #Change to 1 to plot
    plt.figure()
    plt.semilogx(f,  10*np.log10(vco_noise(f)), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("VCO Phase-Noise")

# VCO output-referred Phase noise:
def vco_op_noise(f):
    return vco_noise(f)* (mag_vco**2)

if 1:      #Change to 1 to plot   
    plt.figure()
    plt.semilogx(f,  10*np.log10(vco_op_noise(f)), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("VCO OUT Phase-Noise")
##############################################################################     
# Referance Noise Transfer Function

Tn_ref =(1/(R*P))*B

if 0:print('Tn_ref(S)= ',Tn_ref) 
if 1:mag_ref, phase ,w = ct.bode(Tn_ref,Hz=True,dB=True,plot= False,omega_limits=(start_frequency, end_frequency))
f=w/(2*np.pi)
if 0:plt.figure()
if 1:      #Change to 1 to plot
    plt.figure()
    plt.semilogx(f, 20*np.log10(mag_ref), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Referance Transfer Function")

#  Referance Phase noise:
def Referance_noise(f):
    Referance_noise_flicker = -30 * np.log10(f)         #offset(0)=(-30*-3)-90 . 1KHz = 3 decade
    Referance_white_noise = -20 *np.log10(f) - 39       #offset(-39)=(-20*-5)-139 . 100KHz = 5 decade
    Referance_noise_floor = -150                        
    return 10 ** (Referance_noise_flicker / 10) + 10 ** (Referance_noise_floor / 10) + 10 ** ( Referance_white_noise/ 10)

if 1:      #Change to 1 to plot
    plt.figure()
    plt.semilogx(f,  10*np.log10(Referance_noise(f)), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Referance Phase-Noise")

#Referance Output Phase Noise:
def Referance_op_noise(f):
    return Referance_noise(f)* (mag_ref**2)

if 1:      #Change to 1 to plot   
        plt.figure()
        plt.semilogx(f,10*np.log10(Referance_op_noise(f)), 'black',linewidth=1)
        plt.grid(which="both")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Referance OUT Phase-Noise")
##########################
# Loop Filter Noise Transfer Function

Tn_lf= ((Kvco/s) / ((1 + TS)*P))

if 1:print('Tn_lf(S)= ',Tn_lf)  
if 1:mag_lf, phase ,w = ct.bode(Tn_lf,Hz=True,dB=True,plot= False,omega_limits=(start_frequency, end_frequency))
f=w/(2*np.pi)
if 1:      #Change to 1 to plot
    plt.figure()
    plt.semilogx(f, 20*np.log10(mag_lf), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("LF Transfer Function")

#  Loop Filter Phase noise:
def lf_noise(f):
    Ko = 1.38E-23    #boltzmann constant
    return (4*Ko*300*R_1*f)/f   

if 1:      #Change to 1 to plot
    plt.figure()
    plt.semilogx(f,  10*np.log10(lf_noise(f)), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("LF Phase-Noise")
    
#  lf Output Phase Noise:
def lf_op_noise(f):
    return lf_noise(f)* (mag_lf**2)
    
if 1:      #Change to 1 to plot   
        plt.figure()
        plt.semilogx(f,  10*np.log10(lf_op_noise(f)), 'black',linewidth=1)
        plt.grid(which="both")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("LF OUT Phase-Noise")

##########################
#Charge-Pump Noise Transfer Function

Tn_cp = (2* np.pi /(I_p*P)) * B

if 0:print('Tn_cp(S)= ',Tn_cp)
if 1:mag_cp, phase ,w = ct.bode(Tn_cp,Hz=True,dB=True,plot= False,omega_limits=(start_frequency, end_frequency))
f=w/(2*np.pi)
if 0:plt.figure()
if 1:      #Change to 1 to plot
    plt.figure(19)
    plt.semilogx(f,20*np.log10(mag_cp), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Charge_Pump Transfer Function")

#  Charge Pump Phase noise:
def cp_noise(f):
      cp_noise_floor = -231+10*np.log10(I_p/300e-6)#-231  # dBc            
      cp_flicker_noise =  -10 *np.log10(f) + 10 * np.log10(2*10**6) -231+10*np.log10(I_p/300e-6)                     
      return (10**(cp_noise_floor/10) + 10**(cp_flicker_noise/10))     # assuming noise of charge pump is for the two transistor currents

if 1:      #Change to 1 to plot
    plt.figure()
    plt.semilogx(f,10*np.log10(cp_noise(f)), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Charge_Pump Phase-Noise ")

#Charge-Pump Output Phase Noise:
def cp_op_noise(f):
    return cp_noise(f)* (mag_cp**2)

if 1:      #Change to 1 to plot   
        plt.figure()
        plt.semilogx(f,10*np.log10(cp_op_noise(f)), 'black',linewidth=1)
        plt.grid(which="both")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Charge_Pump OUT Phase-Noise ") 
###########################################################
#Feedback Divider Noise Transfer Function

Tn_divN = B *(1/P)

if 0:print('Tn_divN(S)= ',Tn_divN)
if 1:mag_divN, phase ,w = ct.bode(Tn_divN,Hz=True,dB=True,plot= False,omega_limits=(start_frequency, end_frequency))
f=w/(2*np.pi)
if 0:plt.figure()
if 1:      #Change to 1 to plot
    plt.figure(15)
    plt.semilogx(f, 20*np.log10(mag_divN), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Divider N Transfer Function")

#Divider N Phase Noise for 
def divN_noise(f):
    divN_noise_floor = -160
    divN_flicker_noise = -10 *np.log10(f) + 10 * np.log10(2*10**6) - 160    #flicker Corner 2 MHz
    return 10 ** (divN_flicker_noise / 10) + 10 ** (divN_noise_floor / 10)

#Divider N Phase noise:
if 1:      #Change to 1 to plot
    plt.figure()
    plt.semilogx(f,  10*np.log10(divN_noise(f)), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Divider N Phase-Noise")

#Divider N Output Phase Noise:
def divN_op_noise(f):
    return divN_noise(f)* (mag_divN**2)

if 1:      #Change to 1 to plot   
        plt.figure()
        plt.semilogx(f,10*np.log10(divN_op_noise(f)), 'black',linewidth=1)
        plt.grid(which="both")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Divider N OUT Phase-Noise") 
#############################################################
#R Divider Noise Transfer Function

Tn_divR = (1/(P))* B

if 0:print('Tn_divR(S)= ',Tn_divR)
if 1:mag_divR, phase ,w = ct.bode(Tn_divR,Hz=True,dB=True,plot= False,omega_limits=(start_frequency, end_frequency))
f=w/(2*np.pi)
if 0:plt.figure()
if 1:      #Change to 1 to plot
    plt.figure()
    plt.semilogx(f,20*np.log10(mag_divR), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Divider R Transfer Function")

#Divider R Phase Noise for 
def divR_noise(f):
    divR_noise_floor = -160
    divR_flicker_noise = -10 *np.log10(f) + 10 * np.log10(2*10**6) - 160    #flicker Corner 2 MHz
    return 10 ** (divR_flicker_noise / 10) + 10 ** (divR_noise_floor / 10)

#Divider R Phase noise:
if 1:      #Change to 1 to plot
    plt.figure()
    plt.semilogx(f,  10*np.log10(divR_noise(f)), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Divider R Phase-Noise")

#Divider R Output Phase Noise:
def divR_op_noise(f):
    return divR_noise(f)*(mag_divR**2)
if 1:      #Change to 1 to plot   
        plt.figure()
        plt.semilogx(f,10*np.log10(divR_op_noise(f)), 'black',linewidth=1)
        plt.grid(which="both")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Divider R OUT Phase-Noise ") 
###################################
#Divider P Noise Transfer Function

Tn_divP = 1

mag_divP=Tn_divP
if 0:print('Tn_divP(S)= ',Tn_divP)

'''
#if 1: mag_divP, phase ,f = ct.bode(Tn_divP,f,Hz=True,dB=True,plot= False,omega_limits=(start_frequency, end_frequency))
#f=w/(2*np.pi)
'''
#if 1:plt.figure()
if 1:      #Change to 1 to plot
    plt.figure()
    plt.semilogx(f,20*np.log10(Tn_divP*f/f), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Divider P Transfer Function")

#Divider P Phase Noise for 
def divP_noise(f):
    divP_noise_floor = -160
    divP_flicker_noise = -10 *np.log10(f) + 10 * np.log10(2*10**6) - 160    #flicker Corner 2 MHz
    return 10 ** (divP_flicker_noise / 10) + 10 ** (divP_noise_floor / 10)

#Divider P Phase noise:
if 1:      #Change to 1 to plot
    plt.figure()
    plt.semilogx(f, 10*np.log10(divP_noise(f)), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Divider P Phase-Noise")
#Divider P Output Phase Noise:
def divP_op_noise(f):
    return divP_noise(f)*(mag_divP**2)

if 1:      #Change to 1 to plot   
        plt.figure()
        plt.semilogx(f,10*np.log10(divP_op_noise(f)), 'black',linewidth=1)
        plt.grid(which="both")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Divider P OUT Phase-Noise ") 
##########################################################
#Buffer Noise Transfer Function

Tn_buf = 1


'''
if 0:print('Tn_buf(S)= ',Tn_buf)

if 0:mag_buf, phase ,f = ct.bode(Tn_buf,f,Hz=True,dB=True)
if 0:plt.figure()
'''
if 1:      #Change to 1 to plot
    plt.figure()
    plt.semilogx(f,20*np.log10(Tn_buf*f/f), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Buffer Transfer Function")


#Buffer Phase Noise for 
def buf_noise(f):
    buf_noise_floor = -160
    buf_flicker_noise = -10 *np.log10(f) + 10 * np.log10(2*10**6) - 160    #flicker Corner 2 MHz
    return 10 ** (buf_flicker_noise / 10) + 10 ** (buf_noise_floor / 10)

#Buffer Phase noise:
if 1:      #Change to 1 to plot
    plt.figure()
    plt.semilogx(f, 10*np.log10(buf_noise(f)), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Buffer Phase-Noise")
#Buffer Output Phase noise:
def buf_op_noise(f):
    return buf_noise(f)*(Tn_buf**2)
if 1:      #Change to 1 to plot
    plt.figure()
    plt.semilogx(f,  10*np.log10(buf_noise(f)*Tn_buf**2), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Buffer Output Phase-Noise")

######################################
# Overall output-referred Phase noise:
      
def overall_out_phase_noise(f):
   return  vco_op_noise(f) + lf_op_noise(f) + Referance_op_noise(f) + cp_op_noise(f) + divN_op_noise(f) + divP_op_noise(f) + divR_op_noise(f) + buf_op_noise(f)
if 1:#Change to 1 to plot
    plt.figure()
    plt.semilogx(f, 10*np.log10(overall_out_phase_noise(f)), 'black', label='Total',linewidth=2)
    plt.legend()
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase noise in dBc/Hz")
    plt.title("PLL Output Phase noise")
######################################
# Overall Phase noise:
    
def overall_phase_noise(f):
   return  vco_noise(f) + lf_noise(f) + Referance_noise(f) + cp_noise(f) + divN_noise(f) + divP_noise(f) + divR_noise(f) + buf_noise(f)
if 1:#Change to 1 to plot
    plt.figure()
    plt.semilogx(f,10*np.log10(overall_phase_noise(f)), 'black', label='Total',linewidth=2)
    plt.legend()
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase noise in dBc/Hz")
    plt.title("PLL Phase noise")

##############################/((I_p/2*np.pi)**2) 
def Close_in(f):
   return (Referance_noise(f) + divN_noise(f) + cp_noise(f)/((I_p/2*np.pi)**2) + divR_noise(f))*N**2
if 1:      #Change to 1 to plot
    plt.figure()
    plt.semilogx(f,  10*np.log10(vco_noise(f)), 'green', label='VCO',linewidth=1)
    plt.semilogx(f,  10*np.log10(Close_in(f)), 'cyan', label='Close_in',linewidth=1)
    plt.legend()
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase noise in dBc/Hz")
    
#All blocks & Overall output-referred Phase noise:
    
if 1:      #Change to 1 to plot
    plt.figure( figsize =(12 , 9))       
    plt.semilogx(f,  10*np.log10(vco_op_noise(f)), 'green', label='VCO',linewidth=1)
    plt.semilogx(f,  10*np.log10(lf_op_noise(f)), 'cyan', label='L.Filter',linewidth=1)
    plt.semilogx(f,  10*np.log10(Referance_op_noise(f)), 'red', label='Ref',linewidth=1)
    plt.semilogx(f,  10*np.log10(cp_op_noise(f)), 'magenta', label='Charge_Pump',linewidth=1)
    plt.semilogx(f,  10*np.log10(divN_op_noise(f)), 'blue', label='div_N',linewidth=1)  
    plt.semilogx(f,  10*np.log10(divR_op_noise(f)), 'Purple', label='div_R', linewidth=1)
    plt.semilogx(f,  10*np.log10(divP_op_noise(f)), 'grey', label='div_P', linewidth=1)
    plt.semilogx(f,  10*np.log10(buf_op_noise(f)), 'yellow', label='buffer', linewidth=1)     
    plt.semilogx(f,  10*np.log10(overall_out_phase_noise(f)), 'black', label='Total',linewidth=2)
    plt.legend()
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase noise in dBc/Hz")
    plt.title("PLL Phase Output noise & Blocks @ PM=75")
##############################################
#Integrated RMS phase jitter
def integrate_and_calculate_rms(phase_noise, frequency, ref_frequency):
    # Make sure phase_noise and frequency are NumPy arrays
    phase_noise = np.array(phase_noise)
    frequency = np.array(frequency)
 
    # Define the constant frequency range for integration (12 kHz to 20 MHz)
    lower_limit = 12e3  # 12 kHz
    upper_limit = 20e6  # 20 MHz
 
    # Find the indices that correspond to the specified frequency range
    indices = (frequency >= lower_limit) & (frequency <= upper_limit)
 
    # Integrate the phase noise using numpy.trapz
    integrated_value = np.trapz(phase_noise[indices], frequency[indices])
 
    # Calculate RMS output jitter
    rms_output_jitter = np.sqrt(2 * integrated_value) / (2 * np.pi * ref_frequency)
 
    return integrated_value, rms_output_jitter
##########################
overall_out_phase_noise = overall_out_phase_noise(f)
Total_Integral,Jitter = integrate_and_calculate_rms(overall_out_phase_noise,f, fo)

vco_op_noise = vco_op_noise(f)
vco_Integral,Jitter_vco = integrate_and_calculate_rms(vco_op_noise,f, fo)

lf_op_noise = lf_op_noise(f)
lf_Integral,Jitter_lf = integrate_and_calculate_rms(lf_op_noise,f, fo)

Referance_op_noise = Referance_op_noise(f)
Referance_Integral,Jitter_ref = integrate_and_calculate_rms(Referance_op_noise,f, fo)

cp_op_noise = cp_op_noise(f)
cp_Integral,Jitter_cp = integrate_and_calculate_rms(cp_op_noise,f, fo)

divN_op_noise = divN_op_noise(f)
divn_Integral,Jitter_divn = integrate_and_calculate_rms(divN_op_noise,f, fo)

divP_op_noise = divP_op_noise(f)
divP_Integral,Jitter_divP = integrate_and_calculate_rms(divP_op_noise,f, fo)

divR_op_noise = divR_op_noise(f)
divR_Integral,Jitter_divR = integrate_and_calculate_rms(divR_op_noise,f, fo)

buf_op_noise = buf_op_noise(f)
buf_Integral,Jitter_buf = integrate_and_calculate_rms(buf_op_noise,f, fo)





  

''' 
    min_Jitter = min(Jitter)
    if (min_Jitter==Jitter)&((c_1+c_2)<=10**-9):
                            I_P_minj=I_p 
                            wc_miniJ=wc
                            R_1minj=R_1
                            c_1minj=c_1
                            c_2minj=c_2
'''
print("RMS Jitter = ", Jitter*1E15,"fsec")
print("RMS Jitter_vco = ", Jitter_vco*1E15,"fsec")
print("RMS Jitter_lf = ", Jitter_lf*1E15,"fsec")
print("RMS Jitter_ref = ", Jitter_ref*1E15,"fsec")
print("RMS Jitter_cp = ", Jitter_cp*1E15,"fsec")
print("RMS Jitter_divn = ", Jitter_divn*1E15,"fsec")
print("RMS Jitter_divP = ", Jitter_divP*1E15,"fsec")
print("RMS Jitter_divR = ", Jitter_divR*1E15,"fsec")
print("RMS Jitter_buf = ", Jitter_buf*1E15,"fsec")
print("BW (Hz)= ", wc/(2*np.pi))
print("I_CP = ", I_p)     
print("R1 = ", R_1)
print("C1 = ", c_1)
print("C2 = ", c_2)
########################################################3
###########contributions

#REF Contribution
REF_Contribution_Precentage =(Referance_Integral/Total_Integral)*100
print("Reference Contribution Precentage= ",REF_Contribution_Precentage,"%")


#VCO Contribution
VCO_Contribution_Precentage =(vco_Integral/Total_Integral)*100
print("VCO Contribution Precentage= ",VCO_Contribution_Precentage,"%")

#CP Contribution
CP_Contribution_Precentage =(cp_Integral/Total_Integral)*100
print("Charge Pump (CP) Contribution Precentage= ",CP_Contribution_Precentage,"%")


#LF Contribution
LF_Contribution_Precentage =(lf_Integral/Total_Integral)*100
print("Loop Filter (LF) Contribution Precentage= ",LF_Contribution_Precentage,"%")



#N-Divider Contribution
N_Divider_Contribution_Precentage=(divn_Integral/Total_Integral)*100
print("N-Divider Contribution Precentage= ",N_Divider_Contribution_Precentage,"%")

#R-Divider Contribution
R_Divider_Contribution_Precentage=(divR_Integral/Total_Integral)*100
print("R-Divider Contribution Precentage= ",R_Divider_Contribution_Precentage,"%")


#P-Divider Contribution
P_Divider_Contribution_Precentage=(divP_Integral/Total_Integral)*100
print("P-Divider Contribution Precentage= ",P_Divider_Contribution_Precentage,"%")

#Buffer Contribution
Buffer_Contribution_Precentage=(buf_Integral/Total_Integral)*100
print("Buffer Contribution Precentage= ",Buffer_Contribution_Precentage,"%")


Total_Contribution_Precentage =Buffer_Contribution_Precentage + P_Divider_Contribution_Precentage + R_Divider_Contribution_Precentage + N_Divider_Contribution_Precentage + LF_Contribution_Precentage + CP_Contribution_Precentage + VCO_Contribution_Precentage + REF_Contribution_Precentage 
print("Overall Contribution Precentage= ",Total_Contribution_Precentage,"%")

