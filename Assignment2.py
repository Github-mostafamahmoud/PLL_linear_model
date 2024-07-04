import matplotlib.pyplot as plt
import numpy as np
import control as ct 
#################################################
#I_ps=np.arange(50*10**-6,1*10**-3,50*10**-6)
#wcs=np.arange(2*np.pi*10**1,2*np.pi*16*10**6,2*np.pi*10*10**2)
#I_p= 300*10**-6
while True:
    G = int(input('Enter M (allowed values {0,105,200})\n'))
    if G not in (0,105,200):
        print("Invalid input")
    else:
        break
G=G *10**-6
wc=2*np.pi*7.8*10**5
#for I_p in I_ps:
#for wc in wcs:
   
#Definition of omega & frequency 
wi = 2*np.pi*1E3
wf = 2*np.pi*40E6
dw = 200
nw = int((wf - wi) / dw) + 1  # Number of points of angular freq
w = np.linspace(wi, wf, nw)        
f = w/(2*np.pi)

# variables
I_p= 1*10**-3                 
x=6
Kvco= 2*np.pi*90*10**6        
P=24
R=1
fo = 156.25*10**6       #(320*10**6)*N/(R*P)
N= (fo*P)/(160*10**6)
K = (Kvco*I_p)/(2*np.pi*N)
R_1 =wc/(K*(1-1/(x**2)))
wz=wc/x 
wp=wc*x
c_1=1/(R_1*wz)
c_2=c_1/((x**2)-1)
c_eq = (c_1*c_2)/(c_1+c_2)

taw_1 = (R_1 * c_1)
taw_2 = taw_1 * (c_2 / (c_1 + c_2) )

# impednce (2nd_order_passive_filter)

num1 =np.array([taw_1,1])
den1 = np.convolve(np.array([(c_1 + c_2),0]) ,np.array([taw_2,1]))
Z =ct.tf(num1,den1)
if 0:print('Z(S)= ',Z)

if 1:magnitudeZ, phase ,f = ct.bode(Z,f,Hz=True,plot=False)
if 0:plt.figure(0)



# open loop response

num2 = [K]
den2 = [1,0]
D = ct.tf(num2,den2)
#if 0:print('D(S)= ',D)
A = ct.series(D, Z)
#print('A(S)= ',A)

if 1:(magnitudeA, phase ,f) = ct.bode(A,f ,Hz=True ,margins=True ,plot=True)
if 1:plt.figure()

########################################
#Closed loop response

B=N*ct.feedback(A,1)   # N*(A(S)/(1+A(S)))
if 0:print('B(S)= ',B)
if 1:(magnitudeB, phase ,f) = ct.bode(B, f ,Hz=True,dB=True,plot=False)
if 0:plt.figure()
if 0:      #Change to 1 to plot
    plt.figure()
    plt.semilogx(f, 20*np.log10(magnitudeB), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Closed Loop Transfer Function (dB)")


################################################################################
                        ### Noise Transfer Function ####
    
# loop Gain 
TN = np.convolve(np.array([Kvco*I_p]) ,np.array([R_1*c_1,1]))
TD = np.convolve(np.array([R_1*c_eq,1]) ,np.array([2*np.pi*(c_1+c_2)*N,0,0]))

T=ct.tf(TN,TD)
T1=ct.feedback(1,T)  ## 1/(1+T(S))
if 0:print('T(S)= ',T) 
if 0:print('T1(S)= ',T1) 
############################
        #VCO noise 

#Transfer Function
Tn_vco=(1/P)*T1     
if 0:print('Tn_vco(S)= ',Tn_vco)  
if 1:(mag_vco, phase ,f) = ct.bode(Tn_vco,f,Hz=True,dB=True,plot= False)
if 0:plt.figure()
if 0:      #Change to 1 to plot
    plt.semilogx(f, 20*np.log10(mag_vco), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("VCO Phase-Noise Transfer Function")

###############
#  VCO Phase Noise:
def vco_noise(f):
    vco_flicker_noise = -30*np.log10(f) + 52     #offset(52)=(-30*-3)-38 . 1KHz = 3 decade ,-30 --> slope 
    vco_white_noise = -20 *np.log10(f) + 5       #offset(5)=(-20*-6)-115 . 1MHz = 6 decade, -20 --> slope
    vco_noise_floor = -145                       #Noise floor of vco
    return 10**(vco_flicker_noise/10) + 10**(vco_noise_floor/10) + 10**(vco_white_noise/10)

if 0:      #Change to 1 to plot
    plt.figure()
    plt.semilogx(f,  10*np.log10(vco_noise(f)), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("VCO Phase-Noise")

# VCO output-referred Phase noise:
def vco_op_noise(f):
    return vco_noise(f)* (mag_vco**2)

if 0:      #Change to 1 to plot   
    plt.figure()
    plt.semilogx(f,  10*np.log10(vco_op_noise(f)), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("VCO OUT Phase-Noise")
      
##########################
# Loop Filter Noise Transfer Function

num3=[Kvco]
den3=[P,0]
F =ct.tf(num3,den3)
if 0:print('F(S)= ',F )

Tn_lf=ct.series(F,T1)
if 0:print('Tn_lf(S)= ',Tn_lf) 

 
if 1:mag_lf, phase ,f = ct.bode(Tn_lf,f,Hz=True,dB=True,plot= False)
if 0:      #Change to 1 to plot
    plt.figure()
    plt.semilogx(f, 20*np.log10(mag_lf), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("LF Transfer Function")

#  Loop Filter Phase noise:
def lf_noise(f):
    Ko = 1.38E-23    #boltzmann constant
    T = 293         #room temperature (Kelvin)
    return (4*Ko*T*R_1*f)/f   

if 0:      #Change to 1 to plot
    plt.figure(9)
    plt.semilogx(f,  10*np.log10(lf_noise(f)), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("LF Phase-Noise")
    
#  lf Output Phase Noise:
def lf_op_noise(f):
    return lf_noise(f)* (mag_lf**2)
    
if 0:      #Change to 1 to plot   
        plt.figure(10)
        plt.semilogx(f,  10*np.log10(lf_op_noise(f)), 'black',linewidth=1)
        plt.grid(which="both")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("LF OUT Phase-Noise")
    
##########################
# crystal Noise Transfer Function

Tn_ref =(1/(P))*B

#crystal Noise Transfer Function  

Tn_crystal =(1/(P))*B

if 0:print('Tn_crystal(S)= ',Tn_crystal) 
if 1:mag_crystal, phase ,f = ct.bode(Tn_crystal,f,Hz=True,dB=True,plot= False)
if 0:plt.figure()
if 0:      #Change to 1 to plot
    plt.figure(11)
    plt.semilogx(f, 20*np.log10(mag_crystal), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Crystal Transfer Function")

#  Crystal Phase noise:
def crystal_noise(f):
    crystal_noise_flicker1 = -30 * np.log10(f) - 40    #offset(-40)=(-30*-3)-130 , 1KHz = 3 decade
    crystal_noise_flicker2 = -20 * np.log10(f) - 70    #offset(-70)=(-20*-3)-130 , 1Hz = 1 decade
    buf_flicker_noise = -10 * np.log10(f) - 100
    buf_white_noise = -160 
    return 10 ** ( crystal_noise_flicker1 / 10) + 10 ** ( crystal_noise_flicker2 / 10)  + 10 ** ( buf_flicker_noise / 10) + 10 ** ( buf_white_noise/ 10)
if 1:      #Change to 1 to plot
    plt.figure(12)
    plt.semilogx(f,  10*np.log10(crystal_noise(f)), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Crystal Phase-Noise")

#Referance Output Phase Noise:
def crystal_op_noise(f):
    return crystal_noise(f)* (mag_crystal**2)

if 0:      #Change to 1 to plot   
        plt.figure(13)
        plt.semilogx(f,10*np.log10(crystal_op_noise(f)), 'black',linewidth=1)
        plt.grid(which="both")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Crystal OUT Phase-Noise")
  
##########################
#Charge-Pump Noise Transfer Function
Tn_cp = (2* np.pi /(I_p*P)) * B


if 0:print('Tn_cp(S)= ',Tn_cp)
if 1:mag_cp, phase ,f = ct.bode(Tn_cp,f,Hz=True,dB=True,plot= False)
if 0:plt.figure(18)
if 0:      #Change to 1 to plot
    plt.figure(19)
    plt.semilogx(f,20*np.log10(mag_cp), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Charge_Pump Transfer Function")

#  Charge Pump Phase noise:
def cp_noise(f):
      cp_noise_floor = -231-20*np.log10(I_p/300e-6)#-231  # dBc            
      cp_flicker_noise =  -10 *np.log10(f) + 10 * np.log10(2*10**6) -231-20*np.log10(I_p/300e-6)                     
      return (10**(cp_noise_floor/10) + 10**(cp_flicker_noise/10))     # assuming noise of charge pump is for the two transistor currents

if 0:      #Change to 1 to plot
    plt.figure(20)
    plt.semilogx(f,10*np.log10(cp_noise(f)), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Charge_Pump Phase-Noise ")

#Charge-Pump Output Phase Noise:
def cp_op_noise(f):
    return cp_noise(f)* (mag_cp**2)

if 0:      #Change to 1 to plot   
        plt.figure(21)
        plt.semilogx(f,10*np.log10(cp_op_noise(f)), 'black',linewidth=1)
        plt.grid(which="both")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Charge_Pump OUT Phase-Noise ")   
##########################
#Feedback Divider Noise Transfer Function

Tn_divN = B *(1/P)

if 0:print('Tn_divN(S)= ',Tn_divN)
if 1:mag_divN, phase ,f = ct.bode(Tn_divN,f,Hz=True,dB=True,plot= False)
if 0:plt.figure(14)
if 0:      #Change to 1 to plot
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
if 0:      #Change to 1 to plot
    plt.figure(16)
    plt.semilogx(f,  10*np.log10(divN_noise(f)), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Divider N Phase-Noise")

#Divider N Output Phase Noise:
def divN_op_noise(f):
    return divN_noise(f)* (mag_divN**2)

if 0:      #Change to 1 to plot   
        plt.figure(17)
        plt.semilogx(f,10*np.log10(divN_op_noise(f)), 'black',linewidth=1)
        plt.grid(which="both")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Divider N OUT Phase-Noise") 
###########################################################
# Delta Sigma phase noise
Tn_delta_sigma = B *(1/(P*N))

if 0:print('Tn_delta_sigma(S)= ',Tn_delta_sigma)
if 1:mag_delta_sigma, phase ,f = ct.bode(Tn_delta_sigma,f,Hz=True,dB=True,plot= False)
if 0:plt.figure(33)
if 0:      #Change to 1 to plot
    plt.figure(34)
    plt.semilogx(f, 20*np.log10(mag_delta_sigma), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Delta sigma Transfer Function")

def delta_sigma_noise(f):
    f_ref = 160E6
    Sq=1/(12*f_ref)
    delta_sigma_noise = (1/(N**2)) * ((f_ref/f)**2) * (np.abs((2*np.sin(np.pi*f/(f_ref))))**4) * Sq   # 4 = 2 * n
    return delta_sigma_noise

if 1:      #Change to 1 to plot
    plt.figure(31)
    plt.semilogx(f,  10*np.log10(delta_sigma_noise(f)), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Delta sigma Phase-Noise")   

def delta_sigma_OP_noise(f):
    delta_sigma_OP_noise = delta_sigma_noise(f)* (mag_delta_sigma**2)
    return delta_sigma_OP_noise

if 1:      #Change to 1 to plot
    plt.figure(32)
    plt.semilogx(f,  10*np.log10(delta_sigma_OP_noise(f)), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Delta sigma Out Phase-Noise")         
#############################################################3
#Input Referred
f_ref = 160E6
num4=[1]
den4=[(1/(2*np.pi*10**4)),1]
F2 =ct.tf(num4,den4)
if 0:print('F2(S)= ',F2 )
 
Tn_input_refferd1 = B *(1/(P*N))*G
if 0:print('Tn_input_refferd1(S)= ',Tn_input_refferd1) 

Tn_input_refferd2 =ct.series(F2,Tn_input_refferd1)

if 0:print('Tn_input_refferd2(S)= ',Tn_input_refferd2)
if 1:mag_input_refferd2, phase ,f = ct.bode(Tn_input_refferd2,f,Hz=True,dB=True,plot= False)
mag_input_refferd = mag_input_refferd2 * (f_ref/f)
if 0:plt.figure(35)
if 0:      #Change to 1 to plot
    plt.figure(36)
    plt.semilogx(f, 20*np.log10(mag_input_refferd), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Input_refferd Transfer Function")

def input_refferd_noise(f):
    input_refferd_noise_floor = -135
    input_refferd_flicker_noise = -10 *np.log10(f) + 10 * np.log10(10*10**3) -135    #flicker Corner 10 KHz
    return 10 ** (input_refferd_flicker_noise / 10) + 10 ** (input_refferd_noise_floor / 10)


if 0:      #Change to 1 to plot
    plt.figure(37)
    plt.semilogx(f,  10*np.log10(input_refferd_noise(f)), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Input_refferd Phase-Noise")   

def input_refferd_OP_noise(f):
    input_refferd_OP_noise = input_refferd_noise(f)* (mag_input_refferd**2)
    return input_refferd_OP_noise

if 0:      #Change to 1 to plot
    plt.figure(38)
    plt.semilogx(f,  10*np.log10(input_refferd_OP_noise(f)), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Input_refferd Out Phase-Noise")  
###################################
#Divider P Noise Transfer Function

Tn_divP = 1/P
mag_divP=Tn_divP
if 0:print('Tn_divP(S)= ',Tn_divP)
#if 1: mag_divP, phase ,f = ct.bode(Tn_divP,f,Hz=True,dB=True)
#if 1:plt.figure()
'''
if 0:      #Change to 1 to plot
    plt.figure()
    plt.semilogx(f,20*np.log10(mag_divP), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Divider P Transfer Function")
'''
#Divider P Phase Noise for 
def divP_noise(f):
    divP_noise_floor = -160
    divP_flicker_noise = -10 *np.log10(f) + 10 * np.log10(2*10**6) - 160    #flicker Corner 2 MHz
    return 10 ** (divP_flicker_noise / 10) + 10 ** (divP_noise_floor / 10)

#Divider P Phase noise:
if 0:      #Change to 1 to plot
    plt.figure(22)
    plt.semilogx(f, 10*np.log10(divP_noise(f)), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Divider P Phase-Noise")
#Divider P Output Phase Noise:
def divP_op_noise(f):
    return divP_noise(f)*(mag_divP**2)

if 0:      #Change to 1 to plot   
        plt.figure()
        plt.semilogx(f,10*np.log10(divP_op_noise(f)), 'black',linewidth=1)
        plt.grid(which="both")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Divider P OUT Phase-Noise ")   
############################################################
'''
#R Divider Noise Transfer Function

Tn_divR = (1/(P))* B

if 0:print('Tn_divR(S)= ',Tn_divR)
if 1:mag_divR, phase ,f = ct.bode(Tn_divR,f,Hz=True,dB=True,plot= False)
if 0:plt.figure(23)
if 0:      #Change to 1 to plot
    plt.figure(24)
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
if 0:      #Change to 1 to plot
    plt.figure(25)
    plt.semilogx(f,  10*np.log10(divR_noise(f)), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Divider R Phase-Noise")

#Divider R Output Phase Noise:
def divR_op_noise(f):
    return divR_noise(f)*(mag_divR**2)
if 0:      #Change to 1 to plot   
        plt.figure(26)
        plt.semilogx(f,10*np.log10(divR_op_noise(f)), 'black',linewidth=1)
        plt.grid(which="both")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Divider R OUT Phase-Noise ")  
        '''
##############################################################################
#Buffer Noise Transfer Function

Tn_buf = 1



if 0:print('Tn_buf(S)= ',Tn_buf)
'''
if 0:mag_buf, phase ,f = ct.bode(Tn_buf,f,Hz=True,dB=True)
if 0:plt.figure()
if 0:      #Change to 1 to plot
    plt.figure()
    plt.semilogx(f,20*np.log10(mag_buf), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Buffer Transfer Function")
'''

#Buffer Phase Noise for 
def buf_noise(f):
    buf_noise_floor = -160
    buf_flicker_noise = -10 *np.log10(f) + 10 * np.log10(2*10**6) - 160    #flicker Corner 2 MHz
    return 10 ** (buf_flicker_noise / 10) + 10 ** (buf_noise_floor / 10)

#Buffer Output Phase noise:
def buf_op_noise(f):
    return buf_noise(f)*(Tn_buf**2)
if 0:      #Change to 1 to plot
    plt.figure(27)
    plt.semilogx(f,  10*np.log10(buf_noise(f)*Tn_buf**2), 'black',linewidth=1)
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Buffer Output Phase-Noise")

######################################
# Overall output-referred Phase noise:
      
def overall_out_phase_noise(f):
   return  vco_op_noise(f) + lf_op_noise(f) + crystal_op_noise(f) + cp_op_noise(f) + divN_op_noise(f) + divP_op_noise(f) + delta_sigma_OP_noise(f) + buf_op_noise(f) + input_refferd_OP_noise(f)
if 1:#Change to 1 to plot
    plt.figure(28)
    plt.semilogx(f,10*np.log10(overall_out_phase_noise(f)), 'blue', label='Total',linewidth=2)
    plt.legend()
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase noise in dBc/Hz")
    plt.title("PLL Out Phase noise @ PM= 71 deg")
    
##############################
def Close_in(f):
   return (crystal_noise(f) + divN_noise(f) + cp_noise(f)/((I_p/2*np.pi)**2) + delta_sigma_noise(f) + input_refferd_noise(f))*(N**2)
if 0:      #Change to 1 to plot
    plt.figure(29)
    plt.semilogx(f,  10*np.log10(vco_noise(f)), 'green', label='VCO',linewidth=1)
    plt.semilogx(f,  10*np.log10(Close_in(f)), 'cyan', label='Close_in',linewidth=1)
    plt.legend()
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase noise in dBc/Hz")
    
#All blocks & Overall output-referred Phase noise:
    
if 0:      #Change to 1 to plot
    plt.figure(30 , figsize =(12 , 9))       
    plt.semilogx(f,  10*np.log10(vco_op_noise(f)), 'green', label='VCO',linewidth=1)
    plt.semilogx(f,  10*np.log10(lf_op_noise(f)), 'cyan', label='L.Filter',linewidth=1)
    plt.semilogx(f,  10*np.log10(crystal_op_noise(f)), 'red', label='Ref',linewidth=1)
    plt.semilogx(f,  10*np.log10(cp_op_noise(f)), 'magenta', label='Charge_Pump',linewidth=1)
    plt.semilogx(f,  10*np.log10(divN_op_noise(f)), 'blue', label='div_N',linewidth=1)  
    plt.semilogx(f,  10*np.log10(divP_op_noise(f)), 'deeppink', label='div_P', linewidth=1)
    plt.semilogx(f,  10*np.log10(delta_sigma_OP_noise(f)), 'Purple', label='Delta_sigma', linewidth=1) 
    plt.semilogx(f,  10*np.log10(buf_op_noise(f)), 'yellow', label='buffer', linewidth=1)     
    plt.semilogx(f,  10*np.log10(overall_out_phase_noise(f)), 'black', label='Total',linewidth=2)
    plt.legend()
    plt.grid(which="both")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase noise in dBc/Hz")
    plt.title("PLL Phase noise & Blocks")
##############################################
#Integrated RMS phase jitter function
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
########################################################################
overall_out_phase_noise = overall_out_phase_noise(f)
Total_Integral,Jitter = integrate_and_calculate_rms(overall_out_phase_noise,f, fo)

vco_op_noise = vco_op_noise(f)
vco_Integral,Jitter_vco = integrate_and_calculate_rms(vco_op_noise,f, fo)

lf_op_noise = lf_op_noise(f)
lf_Integral,Jitter_lf = integrate_and_calculate_rms(lf_op_noise,f, fo)

crystal_op_noise = crystal_op_noise(f)
crystal_Integral,Jitter_crystal = integrate_and_calculate_rms(crystal_op_noise,f, fo)

cp_op_noise = cp_op_noise(f)
cp_Integral,Jitter_cp = integrate_and_calculate_rms(cp_op_noise,f, fo)

divN_op_noise = divN_op_noise(f)
divn_Integral,Jitter_divn = integrate_and_calculate_rms(divN_op_noise,f, fo)

divP_op_noise = divP_op_noise(f)
divP_Integral,Jitter_divP = integrate_and_calculate_rms(divP_op_noise,f, fo)

delta_sigma_OP_noise = delta_sigma_OP_noise(f)
delta_sigma_Integral,Jitter_delta_sigma = integrate_and_calculate_rms(delta_sigma_OP_noise,f, fo)

buf_op_noise = buf_op_noise(f)
buf_Integral,Jitter_buf = integrate_and_calculate_rms(buf_op_noise,f, fo)

input_refferd_OP_noise = input_refferd_OP_noise(f)
input_refferd_Integral,Jitter_input_refferd = integrate_and_calculate_rms(input_refferd_OP_noise,f, fo)




  

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
print("RMS Jitter_crystal = ", Jitter_crystal*1E15,"fsec")
print("RMS Jitter_cp = ", Jitter_cp*1E15,"fsec")
print("RMS Jitter_divn = ", Jitter_divn*1E15,"fsec")
print("RMS Jitter_divP = ", Jitter_divP*1E15,"fsec")
print("RMS Jitter_delta_sigma = ", Jitter_delta_sigma*1E15,"fsec")
print("RMS Jitter_buf = ", Jitter_buf*1E15,"fsec")
print("fu = ", wc/(2*np.pi),"Hz")
print("I_CP = ", I_p)     
print("R1 = ", R_1)
print("C1 = ", c_1)
print("C2 = ", c_2)
########################################################3
###########contributions

#REF Contribution
crystal_Contribution_Precentage =(crystal_Integral/Total_Integral)*100
print("Reference Contribution Precentage= ",crystal_Contribution_Precentage,"%")


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
delta_sigma_Contribution_Precentage=(delta_sigma_Integral/Total_Integral)*100
print("Delta_sigma Contribution Precentage= ",delta_sigma_Contribution_Precentage,"%")


#P-Divider Contribution
P_Divider_Contribution_Precentage=(divP_Integral/Total_Integral)*100
print("P-Divider Contribution Precentage= ",P_Divider_Contribution_Precentage,"%")

#Buffer Contribution
Buffer_Contribution_Precentage=(buf_Integral/Total_Integral)*100
print("Buffer Contribution Precentage= ",Buffer_Contribution_Precentage,"%")

#Buffer Contribution
input_refferd_Contribution_Precentage=(input_refferd_Integral/Total_Integral)*100
print("Input_refferd Contribution Precentage= ",input_refferd_Contribution_Precentage,"%")

#Total Contribution
Total_Contribution_Precentage =Buffer_Contribution_Precentage + P_Divider_Contribution_Precentage + delta_sigma_Contribution_Precentage + N_Divider_Contribution_Precentage + LF_Contribution_Precentage + CP_Contribution_Precentage + VCO_Contribution_Precentage + crystal_Contribution_Precentage 
print("Overall Contribution Precentage= ",Total_Contribution_Precentage,"%")










