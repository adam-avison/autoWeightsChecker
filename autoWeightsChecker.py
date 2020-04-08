""" autoWeightChecker.py
- Created by A.Avison (adam.avison@manchester.ac.uk)
- April 2020

Description:
Script to check relative weights of 7M and 12M ALMA data prior to joint deconvolution imaging.
To be run in CASA.

Usage:
User interaction is minimal, user defined input are listed under comment #=== USER INPUT ===#
and are:

do_outfile_print = True      #--- Allow printing of results to text file. [Boolean (True/False)]
do_commandline_print = False #--- Just for testing, print some results to command line.  [Boolean (True/False)]
do_plot = False              #--- Just for testing, plot some diagnostic plots (not very useful).  [Boolean (True/False)]
SPWS = ['0','1','2','3']     #--- Array of requested SPWs to check. [Array of strings]
fileType = 'contsub'         #--- Suffix of MS to use, e.g. 'ms', 'split.cal'. [String]
sourcePath = '/raid1/scratch/aavison/ALMAGAL/TEST_SAMPLE/sources/'  #--- Path to where the data is.
                                                                    #--- Expected directory structure is below. [String]
outfileName = 'CheckWeights_Source_SPW'                             #--- If do_outfile_print = True this will be the prefix of .txt files
                                                    #--- printed the SPW number will be added to this name. [String]

Once these are set the user can execute this in CASA with execfile('autoWeightChecker.py')

Expected directory structure:
The expected directory structure is,

<sourcePath>/sources/SOURCE_NAME/ARRAY_TYPE/MEASUREMENT_SETS


"""


import matplotlib.pyplot as plt
import numpy as np
import re
import os
from datetime import datetime

cwd = os.getcwd()
np.set_printoptions(threshold='nan')

#==== FUNCTION DEFINITIONS ====#
#=== GET WEIGHT INFORMATION FROM THE CALIBRATED DATASET TABLE ===#
def weightsStats(source,SPW,fileType,array,do_plot):

    sourceNom = re.split('/',source)[-1]
    fileName = sourceNom+'_'+array+'_spw'+SPW+'.ms.split.cal.'+fileType

    #---- OPEN MS AND GET WEIGHTS, UV DATA AND FLAGS ---#
    tb.open(source+'/'+array+'/'+fileName, nomodify=True)
    weights = tb.getcol('WEIGHT') #--get weights column
    uvw = tb.getcol('UVW')        #--get uv values for uvdist
    flags = tb.getcol('FLAG')     #--get flag data
    tb.close()

    #--- OPEN MS AND GET NUMBER OF CHANNELS IN TARGET SPW ---#
    #-- (spw always = 0 here as it has been split out from a larger MS and renumbered)
    msmd.open(source+'/'+array+'/'+fileName)
    numChan = msmd.nchan(0)
    msmd.close()

    uu = uvw[0] #--- u data (in metres)
    vv = uvw[1] #--- v data (in metres)
    uvdist = np.sqrt((uu*uu)+(vv*vv)) #--- uv distance (in metres)

    weight_xx = weights[0]
    weight_yy = weights[1]

    flags_xx = np.sum(flags,axis=1)[0]
    flags_yy = np.sum(flags,axis=1)[1]

    #--- scrap auto correlations (u&v == 0)
    uvdist_noauto = uvdist[np.where(np.logical_and(uu!=0,vv!=0))]
    weight_xx_noauto = weight_xx[np.where(np.logical_and(uu!=0,vv!=0))]
    weight_yy_noauto = weight_yy[np.where(np.logical_and(uu!=0,vv!=0))]

    #--- Keep only data with some unflagged channels ---#
    uvdist_unflagged = uvdist[np.where(np.logical_and(flags_xx!=numChan,flags_yy!=numChan))]
    weight_xx_unflagged = weight_xx[np.where(np.logical_and(flags_xx!=numChan,flags_yy!=numChan))]
    weight_yy_unflagged = weight_yy[np.where(np.logical_and(flags_xx!=numChan,flags_yy!=numChan))]

    #--- Mean and stdev Wts ---#
    meanWt_xx = np.mean(weight_xx_unflagged)
    stdWt_xx = np.std(weight_xx_unflagged)
    meanWt_yy = np.mean(weight_yy_unflagged)
    stdWt_yy = np.std(weight_xx_unflagged)

    #--- some diagnostic plots... not that useful except for testing code ---#
    if do_plot:
        if array =='7M':
            markerType = 'x'
        elif array =='TM2':
            markerType = 's'
        fig = plt.figure(float(SPW),figsize=[10,7])
        ax = fig.add_subplot(111)
        ax.plot(uvdist_unflagged,weight_xx_unflagged,marker=markerType,mec='darkmagenta',mfc='None',linestyle='None',label='weights_xx_'+array)
        ax.plot(uvdist_unflagged,weight_yy_unflagged,marker=markerType,mec='goldenrod',mfc='None',linestyle='None',label='weights_yy_'+array)
        ax.hlines(meanWt_xx,0.0,np.max(uvdist_unflagged),color='darkmagenta',linestyle="--",alpha=0.5)
        ax.hlines(meanWt_yy,0.0,np.max(uvdist_unflagged),color='goldenrod',linestyle="--",alpha=0.5)

        plt.show()


    return meanWt_xx, meanWt_yy, stdWt_xx, stdWt_yy,np.max(uvdist_unflagged)

#=== CALCULATE THEORETICAL WT VALUE FROM TSYS AND OTHER META DATA ===#
def calcTheoWT(source,SPW,fileType,array,do_plot,maxuvdist):
    sourceNom = re.split('/',source)[-1]
    fileName = sourceNom+'_'+array+'_spw'+SPW+'.ms.split.cal.'+fileType #--- the file we want

    """
    WTs are \propto 1/sigma^2

    sigma_ij (Jy) = (2k/(eta_q*eta_c*Aeff))*sqrt((Tsys_i*Tsys_j)/(2*chanWid*t_int))*10^26
    Subscript i & j are the ith and jth antenna in the array,

    For simplicity in this calc we will take an median Tsys and square it.
    """

    k = 1.38e-23 #--- Boltzmann constant
    eta_c = 0.88 #--- Correlator efficiency, ALMA tech handbook
    eta_q = 0.96 #--- quantisation efficiency, ALMA tech handbook

    #--- Affective antenna area & integration time per visibility are array dependent
    if array =='7M':
        Aeff = 0.69*np.pi*(np.power(7.0/2.0,2.0)) #--- Geometric area of dish * aperture efficiency from tech handbook (at 230GHz)
        t_int = 10.1#--- In seconds
    elif array =='TM2':
        Aeff = 0.68*np.pi*(np.power(12.0/2.0,2.0))
        t_int = 6.05#--- In seconds

    #--- Channel width & channel freqs & baseband
    tmpSPW = '0'        #--- Because the data have been split out SPW in the data is 0
    msmd.open(source+'/'+array+'/'+fileName)
    chanWid = msmd.chanwidths(int(tmpSPW),unit="Hz")[0]
    #chanFreqs = msmd.chanfreqs(int(SPW),unit="Hz")
    baseband  = msmd.baseband(int(tmpSPW))
    msmd.close()

    baseband_str = 'BB_'+str(baseband) #--- get SPW baseband into basebandName format
    #--- ensure positive value for channel width
    if chanWid < 0:
        chanWid = chanWid*-1.0

    #--- OPEN DATA TABLE AND GET TSYS & FREQ RANGES VALUES ---#
    tb.open(source+'/'+array+'/'+fileName+'/ASDM_CALATMOSPHERE', nomodify=True)
    tsys = tb.getcol('tSys')                #--- ALL Tsys vales
    BB = tb.getcol('basebandName')          #--- Baseband Name
    tb.close()

    #--- CALCULATE SIGMA AND WTS ---#

    wantTsys = tsys[0][np.where(BB==baseband_str)[0]]
    wantTsys = np.append(wantTsys,tsys[1][np.where(BB==baseband_str)[0]])
    medianTsys = np.median(wantTsys)
    stdTsys = np.std(wantTsys)


    sigSqrt = np.sqrt((medianTsys*medianTsys)/(2.0*chanWid*t_int))
    sigFront = (2.0*k)/(eta_c*eta_q*Aeff)

    sigSqrtpstd = np.sqrt(((medianTsys+stdTsys)*(medianTsys+stdTsys))/(2.0*chanWid*t_int))
    sigSqrtmstd = np.sqrt(((medianTsys-stdTsys)*(medianTsys-stdTsys))/(2.0*chanWid*t_int))

    sigma = sigFront*sigSqrt*1.0e26
    sigmapstd = sigFront*sigSqrtpstd*1.0e26
    sigmamstd = sigFront*sigSqrtmstd*1.0e26

    wt = 1.0/np.power(sigma,2.0)
    wtpstd = 1.0/np.power(sigmapstd,2.0)
    wtmstd = 1.0/np.power(sigmamstd,2.0)

    if do_plot:
        fig = plt.figure(float(SPW),figsize=[10,7])
        ax = fig.add_subplot(111)
        ax.hlines(wt,0.0,maxuvdist,color='forestgreen',linestyle="-",alpha=0.5,linewidth=2.0)
        plt.show()

    return wt, np.max(wtmstd-wt, wt-wtpstd), medianTsys

def stdTolCheck(val1,std1,val2,std2):
    #--- CHECK THE TWO
    overlap = False#--- assume the ranges don't overlap to start with
    check1 = 0
    check2 = 0

    min1=val1-std1
    max1=val1+std1

    min2=val2-std2
    max2=val2+std2

    if min1 >= min2 and min1 <= max2:
        check1 = 1
    elif max1 >= min2 and max1 <= max2:
        check1 = 1

    if min2 >= min1 and min2 <= max1:
        check2 = 1
    elif max2 >= min1 and max2 <= max1:
        check2 = 1

    if check1+check2 ==2:
        overlap = True

    return overlap


#===============================================================================#
#======================= CODE PROPER ===========================================#
#===============================================================================#


#=== USER INPUT ===#
do_outfile_print = True      #--- Print results to file
do_commandline_print = False #--- Just for testing
do_plot = False              #--- Just for testing
SPWS = ['0','1','2','3']
fileType = 'contsub' #--- Suffix of MS to use
sourcePath = '/<your>/<path>/<to sources>/'
outfileName = 'CheckWeights_SPW' #----If do_outfile_print = True this will be the prefix of .txt files printed
#==================#

#--- list all
sourceDirs = os.listdir(sourcePath)

#--- Print start time
now = datetime.now()
print "STARTED: ",now.strftime('%H:%M:%S')

#=== Loop through Sources and SPW generate an output file for each ===#
for SPW in SPWS:
    if do_outfile_print:
        outF = open(outfileName+str(SPW)+'.txt','w')
        #--- OUTFILE HEADER
        print >> outF, "#Source\tSPW\tTsys_7M\tTsys_TM2\tWt7M\tstdWt7M\tWtTM2\tstdWtTM2\tWttheo7M\tstdWttheo7M\tWttheoTM2\tstdWttheoTM2\tratio7MTM2\terrRatio7MTM2\tratio7MTM2_theo\terrRatio7MTM2_theo\twarning"

    for sourceDir in sourceDirs:
        #--- get caluclated mean and std dev WTs and max uvdistace for plotting
        SM_meanWtXX,SM_meanWtYY,SM_stdWtXX,SM_stdWtYY,SM_maxUVdist=weightsStats(sourcePath+sourceDir,SPW,fileType,'7M',do_plot)
        TM2_meanWtXX,TM2_meanWtYY,TM2_stdWtXX,TM2_stdWtYY,TM2_maxUVdist = weightsStats(sourcePath+sourceDir,SPW,fileType,'TM2',do_plot)

        #--- average XX and YY
        SM_meanWtXXYY = (SM_meanWtXX+SM_meanWtYY)/2.0
        SM_stdWtXXYY = np.sqrt((SM_stdWtXX*SM_stdWtXX)+(SM_stdWtYY*SM_stdWtYY))

        TM2_meanWtXXYY = (TM2_meanWtXX+TM2_meanWtYY)/2.0
        TM2_stdWtXXYY = np.sqrt((TM2_stdWtXX*TM2_stdWtXX)+(TM2_stdWtYY*TM2_stdWtYY))

        #--- Get theoretical WT and median Tsys
        SM_theoWT,SM_theoStdWt,SM_medTsys = calcTheoWT(sourcePath+sourceDir,SPW,fileType,'7M',do_plot,TM2_maxUVdist)
        TM2_theoWT,TM2_theoStdWt,TM2_medTsys = calcTheoWT(sourcePath+sourceDir,SPW,fileType,'TM2',do_plot,TM2_maxUVdist)

        #--- calculate 7M/TM2 WTs ratio and associated 'error' based on the std. dev
        SMoverTM_XX = SM_meanWtXX/TM2_meanWtXX
        SMoverTM_YY = SM_meanWtYY/TM2_meanWtYY
        SMoverTM_XXYY = SM_meanWtXXYY/TM2_meanWtXXYY
        SMoverTM_theo = SM_theoWT/TM2_theoWT

        error7over12_XX = SMoverTM_XX*(np.sqrt(((SM_stdWtXX/SM_meanWtXX)**2.0)+((TM2_stdWtXX/TM2_meanWtXX)**2.0)))
        error7over12_YY = SMoverTM_YY*(np.sqrt(((SM_stdWtYY/SM_meanWtYY)**2.0)+((TM2_stdWtYY/TM2_meanWtYY)**2.0)))
        error7over12_XXYY = SMoverTM_XXYY*(np.sqrt(((SM_stdWtXXYY/SM_meanWtXXYY)**2.0)+((TM2_stdWtXXYY/TM2_meanWtXXYY)**2.0)))
        error7over12_theo = SMoverTM_theo*(np.sqrt(((SM_theoStdWt/SM_theoWT)**2.0)+((TM2_theoStdWt/TM2_theoWT)**2.0)))

        #--- ADD A WARNING FLAG IF ANY OF THE REAL/THEO RATIOS +/-1stddev ranges do not overlap.
        warning = ""

        overlap_xx = stdTolCheck(SMoverTM_XX,error7over12_XX,SMoverTM_theo,error7over12_theo)#--- returns True/False
        overlap_yy = stdTolCheck(SMoverTM_YY,error7over12_YY,SMoverTM_theo,error7over12_theo)#--- returns True/False
        overlap_xxyy = stdTolCheck(SMoverTM_XXYY,error7over12_XXYY,SMoverTM_theo,error7over12_theo)#--- returns True/False

        if not overlap_xx:
            warning+="WT_XX!"
        if not overlap_yy:
            warning+="WT_YY!"
        if not overlap_xxyy:
            warning+="WT_XXYY!"

        #--- OUTFILE PRINTING
        if do_outfile_print:
            #--- OUTFILE CONTENT
            print >> outF, sourceDir+"\t"+str(SPW)+"\t"+str(np.around(SM_medTsys,1))+"\t"+str(np.around(TM2_medTsys,1))+"\t"+str(np.around(SM_meanWtXXYY,3))+"\t"+str(np.around(SM_stdWtXXYY,3))+"\t"+str(np.around(TM2_meanWtXXYY,3))+"\t"+str(np.around(TM2_stdWtXXYY,3))+"\t"+str(np.around(SM_theoWT,3))+"\t"+str(np.around(SM_theoStdWt,3))+"\t"+str(np.around(TM2_theoWT,3))+"\t"+str(np.around(TM2_theoStdWt,3))+"\t"+str(np.around(SMoverTM_XXYY,3))+"\t"+str(np.around(error7over12_XXYY,3))+"\t"+str(np.around(SMoverTM_theo,3))+"\t"+str(np.around(error7over12_theo,3))+"\t"+warning


        #--- COMMAND LINE PRINTING
        #--- just for testing prints some values to terminal
        if do_commandline_print:
            print "\nSPW"+str(SPW)+":"
            print "====================================================================="
            print "Array\tmean Wtxx\t\tmean Wtyy\t\ttheoWT"
            print "====================================================================="
            print '7M:\t '+str(np.around(SM_meanWtXX,4))+"+/-"+str(np.around(SM_stdWtXX,2))+"\t\t"+str(np.around(SM_meanWtYY,4))+"+/-"+str(np.around(SM_stdWtYY,2))+"\t\t"+str(np.around(SM_theoWT,4))
            print '12M:\t '+str(np.around(TM2_meanWtXX,4))+"+/-"+str(np.around(TM2_stdWtXX,2))+"\t\t"+str(np.around(TM2_meanWtYY,4))+"+/-"+str(np.around(TM2_stdWtYY,2))+"\t\t"+str(np.around(TM2_theoWT,4))
            print "---------------------------------------------------------------------"
            print '7/12: \t'+str(np.around(SMoverTM_XX,4))+"+/-"+str(np.around(error7over12_XX,2))+"\t\t"+str(np.around(SMoverTM_YY,4))+"+/-"+str(np.around(error7over12_YY,2))+"\t\t"+str(np.around(SMoverTM_theo,4))+"+/-"+str(np.around(error7over12_theo,2))
            print "---------------------------------------------------------------------"

    if do_outfile_print:
        outF.close() #--- close output file

now2 = datetime.now()
print "FINISHED: ",now2.strftime('%H:%M:%S')
