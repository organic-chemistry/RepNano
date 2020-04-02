# -*- coding: utf-8 -*-
##
####### usage : $ python ForkPrediction-CNN-TM.py FOLDERwithRepNanoOutputs DetectionFOLDER prefix

import sys
import os
import numpy as np
import Utilities as mu

folder = sys.argv[1]
files = os.listdir(folder)
output = sys.argv[2]
#sample = folder.split('/')[-2].split('-')[0]
sample = sys.argv[3]


############## Parameters for the detection #####################


MinReadLength = 5000 # reads smaller than this limit won't be used for the track detection
Stdev_TM = 0.3 # min standard deviation before rescalling (TM)
Stdev_CNN = 0.115 # min standard deviation before rescalling (CNN)
smoothing = 32 # smoothing for TM
smoothingCNN = 30 # smoothing for CNN
MinAmplitude_TM = 0.5 # min signal amplitude before rescalling (TM)
MinAmplitude_CNN =0.4 # min signal amplitude before rescalling (CNN)
Sparam = 0.25 # parameter of the  Ramer–Douglas–Peucker segmentation
MinJump_TM = 0.44 # Minimum amplitude of signal jump to detect track (TM)
MinJump_CNN = 0.44 # Minimum amplitude of signal jump to detect track (CNN)
LowPlateau_TM = 0.12 # Maximum value of the plateau between diverging forks to call an initiation event (TM)
LowPlateau_CNN = 0.14 # Maximum value of the plateau between diverging forks to call an initiation event (CNN)
MinDist = 1000 # minimum distance between diverging forks to call an initiation event
score = 2 # minimum asymetry score to call termination and initiation events
jumpscore = 1 # minimum jump score to call termination and initiation events

############## Output files #####################
outputTM_F = open(output+'/'+sample+'_S'+str(smoothing)+'SD'+str(Stdev_TM)+'Fit'+str(Sparam)+'MinJump'+str(MinJump_TM)+'MinAmpli'+str(MinAmplitude_TM)+'score'+str(score)+'JS'+str(jumpscore)+'_TM.forks5','w')
outputTM_FnoF = open(output+'/'+sample+'_S'+str(smoothing)+'SD'+str(Stdev_TM)+'Fit'+str(Sparam)+'MinJump'+str(MinJump_TM)+'MinAmpli'+str(MinAmplitude_TM)+'_TM.forksNoF5','w')
outputCNN_F = open(output+'/'+sample+'_S'+str(smoothingCNN)+'SD'+str(Stdev_CNN)+'Fit'+str(Sparam)+'MinJump'+str(MinJump_CNN)+'MinAmpli'+str(MinAmplitude_CNN)+'score'+str(score)+'JS'+str(jumpscore)+'_CNN.forks5','w')
outputCNN_FnoF = open(output+'/'+sample+'_S'+str(smoothingCNN)+'SD'+str(Stdev_CNN)+'Fit'+str(Sparam)+'MinJump'+str(MinJump_CNN)+'MinAmpli'+str(MinAmplitude_CNN)+'_CNN.forksNoF5','w')
outputTM_I = open(output+'/'+sample+'_S'+str(smoothing)+'SD'+str(Stdev_TM)+'Fit'+str(Sparam)+'MinJump'+str(MinJump_TM)+'LowPlateau'+str(LowPlateau_TM)+'MinAmpli'+str(MinAmplitude_TM)+'score'+str(score)+'JS'+str(jumpscore)+'_TM.inits5','w')
outputCNN_I = open(output+'/'+sample+'_S'+str(smoothingCNN)+'SD'+str(Stdev_CNN)+'Fit'+str(Sparam)+'MinJump'+str(MinJump_CNN)+'LowPlateau'+str(LowPlateau_CNN)+'MinAmpli'+str(MinAmplitude_CNN)+'score'+str(score)+'JS'+str(jumpscore)+'_CNN.inits5','w')
outputTM_T = open(output+'/'+sample+'_S'+str(smoothing)+'SD'+str(Stdev_TM)+'Fit'+str(Sparam)+'MinJump'+str(MinJump_TM)+'MinAmpli'+str(MinAmplitude_TM)+'score'+str(score)+'JS'+str(jumpscore)+'_TM.term5','w')
outputCNN_T = open(output+'/'+sample+'_S'+str(smoothingCNN)+'SD'+str(Stdev_CNN)+'Fit'+str(Sparam)+'MinJump'+str(MinJump_CNN)+'MinAmpli'+str(MinAmplitude_CNN)+'score'+str(score)+'JS'+str(jumpscore)+'_CNN.term5','w')


############## Detection #####################

TractNumberCNN = {}
TractNumberTM = {}
TractNumberCNN[0] = 0
TractNumberTM[0] = 0

for Seqs in files :
    if Seqs.split('.')[-1] == 'fa' : 
        print(Seqs)
        g = open(folder+'/'+Seqs,'r')
        try : h = open(folder+'/'+Seqs+'_ratio', 'r')
        except: 
            try : 
                h = open(folder+'/'+Seqs+'_ratio_B', 'r')
            except:
                 print("CNN ratio file not found")
        s = g.readline() 
        l = h.readline()
        while s != '' :
            seq = g.readline()
            if len(seq) > MinReadLength :
                readname = s.split('/')[-1].split(' ')[0].split('\n')[0]
                start, end, strand,chrom = mu.Attributes(s)
                x,y = mu.give_ratio_index2(list(seq))
                if strand == '-' :   
                    X = end - np.array(x)
                else : 
                    X = start+np.array(x)   
                Xs,Ys,Yr = mu.Simplify(X,y, Stdev_TM,smoothing, MinAmplitude_TM,Sparam)
                if Xs!=[]:
                        TractsTM = mu.Detection(Xs,Ys,MinJump_TM)
                        try : 
                            TractNumberTM[len(TractsTM)]+=1
                        except : 
                            TractNumberTM[len(TractsTM)] = 1
                        InitsTM, TractsTM = mu.DetectInits(TractsTM, Xs, X, Yr, LowPlateau_TM,MinDist,score,jumpscore)
                        mu.ExportBedForksNoFilter(TractsTM,outputTM_FnoF, chrom, Seqs, readname, strand)
                        mu.ExportBedForks(TractsTM,outputTM_F, chrom, Seqs, readname, strand, jumpscore, score)
                        mu.ExportInits(InitsTM,outputTM_I,chrom, Seqs, readname, strand)
                        TermTM = mu.DetectTermsFilter3(TractsTM, Xs, X, Yr, LowPlateau_TM, jumpscore, score)
                        mu.ExportInits(TermTM, outputTM_T, chrom, Seqs, readname, strand)
                else : 
                    TractNumberTM[0]+=1

                    ########Ratio CNN#######
                read = l.split('/')[-1].split(' ')[0].split('\n')[0]
                if read != readname :
                    print('CNN and TM files not phased')
                l = h.readline()
                prob = l.split('\n')[0].split(' ')
                y_CNN =[]
                X_CNN = []
                k=0
                for i in x :
                    try : 
                        if prob[i] !='nan' : 
                            y_CNN.append(float(prob[i]))
                            X_CNN.append(X[k])
                            k+=1
                    except : pass
                Xs_CNN,Ys_CNN,Yr_CNN = mu.Simplify(X_CNN,y_CNN, Stdev_CNN,smoothingCNN, MinAmplitude_CNN,Sparam)
                if Xs_CNN!=[]:   
                    TractsCNN = mu.Detection(Xs_CNN,Ys_CNN,MinJump_CNN)
                    try : 
                        TractNumberCNN[len(TractsCNN)]+=1
                    except : 
                        TractNumberCNN[len(TractsCNN)] = 1
                    InitsCNN,  TractsCNN= mu.DetectInits(TractsCNN, Xs_CNN, X_CNN, Yr_CNN, LowPlateau_CNN,MinDist,score, jumpscore) # include correction during initiation events detection 
                    mu.ExportBedForksNoFilter(TractsCNN,outputCNN_FnoF, chrom, Seqs, readname, strand)
                    mu.ExportBedForks(TractsCNN,outputCNN_F, chrom, Seqs, readname, strand,jumpscore, score)
                    mu.ExportInits(InitsCNN,outputCNN_I,chrom, Seqs, readname, strand)
                    TermCNN = mu.DetectTermsFilter3(TractsCNN, Xs_CNN, X_CNN, Yr_CNN, LowPlateau_CNN, jumpscore, score)
                    mu.ExportInits(TermCNN, outputCNN_T, chrom, Seqs, readname, strand)
                    
                else : 
                    TractNumberCNN[0]+=1
            else: 
                l = h.readline()
            s = g.readline()
            l = h.readline()
            
        g.close()
        h.close()
        
        print(Seqs, 'CNN', TractNumberCNN, 'TM', TractNumberTM)

outputTM_FnoF.close()
outputCNN_FnoF.close()
outputTM_F.close()
outputTM_I.close()
outputTM_T.close()
outputCNN_F.close()
outputCNN_I.close()
outputCNN_T.close()

