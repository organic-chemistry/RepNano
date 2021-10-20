# -*- coding: utf-8 -*-
## Usage ##
####### python ForkPredictionPlot.py FOLDERwithRepNanoOutputs OutputName

import sys
import os
import numpy as np
import Utilities as mu
import matplotlib.pyplot as plt


folder = sys.argv[1]
files = os.listdir(folder)
sample = folder.split('/')[-2].split('-')[0]
output = sys.argv[2]

############## Parameters for the detection #####################


MinReadLength = 20000 # reads smaller than this limit won't be used for the track detection
Stdev_TM = 0.3 # min standard deviation before rescalling (TM)
Stdev_CNN = 0.115 # min standard deviation before rescalling (CNN)
smoothing_TM = 32 # smoothing for TM
smoothing_CNN = 30 # smoothing for CNN
MinAmplitude_TM = 0.5 # min signal amplitude before rescalling (TM)
MinAmplitude_CNN =0.4 # min signal amplitude before rescalling (CNN)
Sparam = 0.25 # parameter of the  Ramer–Douglas–Peucker segmentation
MinJump_TM = 0.44 # Minimum amplitude of signal jump to detect track (TM)
MinJump_CNN = 0.44 # Minimum amplitude of signal jump to detect track (CNN)
LowPlateau_TM = 0.12 # Maximum value of the plateau between diverging forks to call an initiation (TM)
LowPlateau_CNN = 0.14 # Maximum value of the plateau between diverging forks to call an initiation (CNN)
MinDist = 1000 # minimum distance between diverging forks to call an initiation
score = 1.7 # minimum asymetry score to call termination and initiation events, usually 2
jumpscore = 1 # minimum jump score to call termination and initiation events


fig = plt.figure(figsize=(8,15))
w = 0.05
j=1

############## Detection #####################

TractNumberCNN = {}
TractNumberTM = {}
TractNumberCNN[0] = 0
TractNumberTM[0] = 0

for Seqs in files :
    if Seqs.split('.')[-1] == 'fa': 
        #print(Seqs,j)
        Sample = Seqs.split('_')[0]
        Number = Seqs.split('.')[0].split('_')[-1]
        g = open(folder+'/'+Seqs,'r')
        try : h = open(folder+'/'+Seqs+'_ratio', 'r')
        except: 
            try : 
                h = open(folder+'/'+Seqs+'_ratio_B', 'r')
            except:
                 print("CNN ratio file not found")
        s = g.readline() 
        l = h.readline()
        while s != '' and j<18 :
            seq = g.readline()
            readname = s.split('/')[-1].split(' ')[0].split('\n')[0]
            if len(seq) > MinReadLength:
                start, end, strand,chrom = mu.Attributes(s)
                x,y = mu.give_ratio_index2(list(seq))
                Y = mu.runningMean(y,smoothing_TM)
                if strand == '-' :   
                    X = end - np.array(x)
                else : 
                    X = start+np.array(x)   

                    
                    ######## TM #########"
                    
                Xs,Ys,Yr = mu.Simplify(X,y, Stdev_TM,smoothing_TM, MinAmplitude_TM,Sparam)   
                if Xs != []:
                                            
                    TractsTM = mu.Detection(Xs,Ys,MinJump_TM)
                    try : 
                        TractNumberTM[len(TractsTM)]+=1
                    except : 
                        TractNumberTM[len(TractsTM)] = 1
                    InitsTM, TractsTM = mu.DetectInits(TractsTM, Xs, X, Yr, LowPlateau_TM,MinDist,score,jumpscore)
                    #mu.ExportBedForksNoFilter(TractsTM,outputTM_FnoF, chrom, Seqs, readname, strand)
                    #mu.ExportBedForks(TractsTM,outputTM_F, chrom, Seqs, readname, strand, jumpscore, score)
                    #mu.ExportInits(InitsTM,outputTM_I,chrom, Seqs, readname, strand)
                    TermTM = mu.DetectTermsFilter3(TractsTM, Xs, X, Yr, LowPlateau_TM, jumpscore, score)
                    #mu.ExportInits(TermTM, outputTM_T, chrom, Seqs, readname, strand)
                        
                else : 
                    TractNumberTM[0]+=1

                        ######## Ratio CNN #######
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
                Xs_CNN,Ys_CNN,Yr_CNN = mu.Simplify(X_CNN,y_CNN, Stdev_CNN,smoothing_CNN, MinAmplitude_CNN,Sparam)
                    
                if Xs_CNN!=[]:   
                    TractsCNN = mu.Detection(Xs_CNN,Ys_CNN,MinJump_CNN)
                    try : 
                        TractNumberCNN[len(TractsCNN)]+=1
                    except : 
                        TractNumberCNN[len(TractsCNN)] = 1
                    InitsCNN,  TractsCNN= mu.DetectInits(TractsCNN, Xs_CNN, X_CNN, Yr_CNN, LowPlateau_CNN,MinDist,score, jumpscore) # include correction during initiation events detection 
                    #print(TractsCNN)
                    #mu.ExportBedForksNoFilter(TractsCNN,outputCNN_FnoF, chrom, Seqs, readname, strand)
                    #mu.ExportBedForks(TractsCNN,outputCNN_F, chrom, Seqs, readname, strand,jumpscore, score)
                    #mu.ExportInits(InitsCNN,outputCNN_I,chrom, Seqs, readname, strand)
                    TermCNN = mu.DetectTermsFilter3(TractsCNN, Xs_CNN, X_CNN, Yr_CNN, LowPlateau_CNN, jumpscore, score)
                    #mu.ExportInits(TermCNN, outputCNN_T, chrom, Seqs, readname, strand)
                    
                else : 
                        TractNumberCNN[0]+=1

           ########### make the figure  ###########
           
                #### plot only the reads with tracks
                
                if (Xs_CNN!=[] and TractsCNN != {}) or (Xs != [] and TractsTM != {}): 
           
                    ax = fig.add_axes([0.05, .99-(w+0.003)*j, .85, w], xticks=[], yticks=[])
                        #fig.text(0.8, 1.0-(w+0.002)*(j), ID)
                    j+=1
                    plt.ylim(0,1)
                    plt.xlim(start-5000, start+75000)
                    
                    plt.plot(X_CNN,y_CNN, color='m', label='CNN', lw = 1)
                    if Xs_CNN!=[]:
                        mu.PlotTracts(TractsCNN, 'm','k','k',0.8)
                        mu.PlotInits(InitsCNN, 'm',0.8)
                        mu.PlotTerms(TermCNN, 'm',0.8)
                        
                    plt.plot(X,Y, color='g', label='TM', lw = 1)
                    if Xs != []:
                        mu.PlotTracts(TractsTM,'g','k','k',0.9)
                        mu.PlotInits(InitsTM, 'g',0.9)
                        mu.PlotTerms(TermTM, 'g',0.9)
                    if j == 2 : plt.legend()

                    
                
                                                    
            else: 
                l = h.readline()
            s = g.readline()
            l = h.readline()
            
        g.close()
        h.close()
        
        print(Seqs, 'CNN', TractNumberCNN, 'TM', TractNumberTM)

###### Add x labels #####
ax = fig.add_axes([0.05, 1.03-(w+0.003)*(j), .85,0.005],yticks=[])
plt.xlim(0,1)
plt.ylim(0,1)
ticks = np.array(range(-1,17))/16
plt.xticks(ticks=ticks, labels=['-5','0','5','10','15','20','25','30','35','40','45','50','55','60','65','70','75','80'])
plt.xlabel('read size (Kb)')

####### Save figure #######
plt.savefig(output+'.svg')
plt.savefig(output+'.png', dpi=200)
#plt.show()

