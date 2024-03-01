""" IMPORT STATEMENTS"""
import numpy as np
import glob
import pandas as pd
from decimal import Decimal
import matplotlib.pyplot as plt

""" 
Variables that were used in Abaqus data creation 
and empty lists for data access and classification

"""
CFlim = 163892
lambdas = np.linspace(0.5, 1.5, 151)
imperfections = [0.01, 0.05, 0.1, 0.2, 0.5]
fs = []
fs1 = []
labels = []
features = []
length = []
CFMAX = []
CFFINAL = []
divide = []
freqs = []
kk = 0
"""
Fill the lists with the files that were stored during ABAQUS runs
""""
for i in range(len(lambdas)):
        for j in range(len(imperfections)):
            for k in range(len(imperfections)): 
                dlt1 = '%.0E' % Decimal(imperfections[j])
                dlt2 = '%.0E' % Decimal(imperfections[k])
                fr=str(round(lambdas[i], 3))+'_'+str(imperfections[j])+'_'+str(imperfections[k])
                nm1 = "WF6X6_RIKS_"+str(i)+"_"+dlt1+'_'+dlt2+".txt"
                nm2 = "WF6X6_RIKS_"+str(i)+"_"+dlt1+'_'+dlt2+"_wave_by_frame*.txt"

                fs.extend(glob.glob("150/"+fr+"_wave_center/"+nm1))
                fs1.extend(glob.glob("150/"+fr+"_wave_center/"+nm2))

"""
Classification Statement
For each case, if the peak load is less than the constant load that we put in the script (bifurcation load),
and the final load occurs at a higher displacement U3 and such load is even less,
then it is imperfection sensitive (IS).
"""
for j, f in enumerate(fs):
    f1 = open(f)
    f2 = f1.read()
    num1 = f.split('_RIKS_')[1].split('_')[0]
    CFmax = float(f2.split('CFmax=')[1].split('U3')[0])
    CFMAX.append(CFmax)
    CFfinal = float(f2.split('CF=')[1].split('PF')[0])
    CFFINAL.append(CFfinal)
    U3 = float(f2.split('U3=')[1].split('CF')[0])
    U3max = float(f2.split('U3max=')[1].split('CFmax')[0])
    if CFmax < CFlim and CFfinal <= 0.85 * CFmax:  # and float(num1)<=80:
        labels.append(1)
    else:
        labels.append(0)
""" 
End of Classification
"""

    """
    Hacky way to access length of column from files made during ABAQUS runs
    """
    f3 = open('150/WF6X6_' + num1 + '_buckle.txt')
    f4 = (f3.readlines())
    lengths = f4[0]
    a = lengths.split('=')[1]
    length.append(float(a))

    """
    Problem is periodic so we extend to full column length
    To do that we use Pandas to read the wave undulation files
    created during ABAQUS runs for our load. 
    Since it is periodic it's easy to extend length and deformation
    """
    df1 = pd.read_csv(fs1[j], delimiter=' ', header=None)
    kk += 1
    hd = ['Node', 'X_undeformed', 'Y_undeformed', 'Z_undeformed', 'U1', 'U2', 'U3']
    df1.columns = hd
    df1 = df1.sort_values(by=['Z_undeformed'])

    x1 = list(df1['Z_undeformed'])
    x2 = [-x for x in x1]
    x_wave = x2[::-1] + x1
    y1 = list(df1['U2'])
    y2 = [x for x in y1]
    y_wave = y2[::-1] + y1

    """
    Obtain the FFT Frequencies for our undulations
    We define the domain using the length of the column.
    We transform an interpolated version of the deformation values 
    over to the FFT domain 
    """

    """
    FFT Domain Definition
    """
    T_w = max(x_wave) + abs(min(x_wave))
    n_w = 50  #number of sampling points, must be large
    sr_w = 1 / n_w  #sample rate
    timeStep_w = T_w / n_w  #sample spacing
    fftFreq_w = np.fft.fftfreq(n_w, d=timeStep_w)  # see definition in website
    shiftFreq_w = np.fft.fftshift(fftFreq_w)
    sr_w = 1 / n_w  #sample rate
    timeStep_w = T_w / n_w  #sample spacing
    fftFreq_w = np.fft.fftfreq(n_w, d=timeStep_w)  # see definition in website
    shiftFreq_w = np.fft.fftshift(fftFreq_w)

    """"
    Transform undulations (signal) to the FFT domain
    """
    xvals = np.linspace(min(x_wave), max(x_wave), 150)
    yinterp = np.interp(xvals, x_wave, y_wave)
    transform_w = np.fft.fft(yinterp)
    #plt.plot(xvals,yinterp,label='original')
    shiftTransform_w = np.fft.fftshift(transform_w)  #look up on website

    maxval = 0
    n = n_w + (n_w - 1) % 2 #make sure n is odd
    n0 = int((n + 1) / 2 - 1) #where the zero freq. is


    """
    Find the LEADING contributing frequency 
    of the wave undulation to shift back to the domain
    and keep it as a feature for ML
    """
    for k in range(0, 10):  # modes
        oneTerm = np.zeros(n, dtype=complex)  #empty array
        oneTerm[n0 + k] = shiftTransform_w[n0 +
                                           k]  #put single DFT in its place
        oneTerm[n0 - k] = shiftTransform_w[n0 - k]  #and use its conjudate also
        inverse = np.fft.ifft(np.fft.ifftshift(oneTerm))  # note (1)
        if max((inverse)) > maxval:
            maxval = max(inverse)
            tokeep = inverse
    freqs.append(tokeep)

"""
Save features and labels files 
"""

features1 = np.zeros((len(freqs),150)) #Empty array for inverse FFT term 
for i in range(len(freqs)):
    for j,item in enumerate(freqs[i]):
        features1[i,j]=item  #add the value from FFT to the array

features = np.column_stack([length,features1]) # Stack Length of column to Features1 array
np.savetxt('features.txt',features)
np.savetxt('labels.txt', labels)
