import numpy as np
from simplification.cutil import simplify_coords
import matplotlib.pyplot as plt

def give_ratio_index2(seq):
    val = np.zeros(len(seq))*np.nan
    val[np.array(seq) == "T"] = 0
    val[np.array(seq) == "B"] = 1
    index = [ind for ind, v in enumerate(val) if v in [0, 1]]
    return index, val[~np.isnan(val)]

def Tsites(seq) : 
    val = np.zeros(len(seq))*np.nan
    val[np.array(seq) == "T"] = 1
    val[np.array(seq) == "B"] = 1
    val[np.array(seq) == "X"] = 1
    index = [ind for ind, v in enumerate(val) if v in [1]]
    return index
    

def runningMean(x, N):
    filt = np.convolve(x, np.ones((N,))/N)[(N-1):]
    filt1 = np.convolve(x, np.ones((N,))/N)[:-N+1]
    filt = 0.5*(filt+filt1)
    return filt

def Simplify(x,y, stdev,smoothing, MinAmplitude,Sparam):
    Xs = []
    Ys =[]
    Yr=[]
    if np.std(y)> stdev :
        Y = runningMean(y,smoothing)
        if max(Y)-min(Y) > MinAmplitude :
            Yr = (np.array(Y)-min(Y))/(max(Y)-min(Y)) # rescale between 0 and 1 
            coords = list(zip(x,Yr))
            simplified = np.array(simplify_coords(coords, Sparam))
            simplified = simplified[simplified[:,0].argsort()]
            Xs = simplified[:,0]
            Ys = simplified[:,1]
            for i in range(0,len(Xs)-2) : 
                if abs(Xs[i+2]-Xs[i]) < 1000 and abs(Ys[i+2]-Ys[i]) < 0.3 and (abs(Ys[i+2]-Ys[i+1])>0.6 or abs(Ys[i]-Ys[i+1])>0.6):
                    Ys[i+1] = (Ys[i+2]+Ys[i])/2 
    return Xs,Ys,Yr

def Detection(Xs,Ys,MinJump):
        slope = {}
        jump = {}
        Tracts = {}
        ID = 0
        Counted = []
        for a in range(0, len(Xs)-1) : 
            slope[a] = (Ys[a+1]-Ys[a])/(Xs[a+1]-Xs[a])*100000
            jump[a] = (Ys[a+1]-Ys[a])
        for i in range(0, len(Xs)-2) :
            if jump[i] > MinJump or (jump[i]+jump[i+1]>MinJump and jump[i] <= MinJump and jump[i+1] <= MinJump): 
                startTract = Xs[i]
                rest = 0  
                for k in range(i+1, len(Xs)-1) :
                    if jump[k]+rest < -MinJump*0.6: 
                        try : 
                            if jump[k+1]<0 and slope[k+1] < -50 : 
                                k+=1 # problem when decrease is splitted into 2 tracts
                        except : pass
                        endTract = Xs[k+1]
                        Counted.append(k)
                        Tracts[ID] = {}
                        Tracts[ID]['start'] = startTract
                        Tracts[ID]['end'] = endTract
                        Tracts[ID]['JumpScore'] = max((jump[i]-MinJump)*10, (jump[i]+jump[i+1]-MinJump)*10, (-jump[k]-rest-MinJump)*10)
                        Oriented = False
                        try : 
                            if endTract == Tracts[ID-1]['end'] and Tracts[ID-1]['dir'] == '?': 
                                ID -=1
                                Tracts[ID]['start'] = startTract
                                del Tracts[ID+1]
                        except : pass
                        maxi = Ys[i:k+1].argmax()
                        slope1 = (Ys[i+maxi]-Ys[i])/(Xs[i+maxi]-Xs[i])*100000
                        slope2 = (Ys[k+1]-Ys[i+maxi])/(Xs[k+1]-Xs[i+maxi])*100000
                        
                        if -slope2*1.5 < slope1:                                 
                            Tracts[ID]['dir'] = 'R'
                            S = slope1/-slope2
                            Y = Ys[k+1]-Ys[i]
                            Tracts[ID]['score'] = max(S,20*Y)
                            Oriented = True 
                            
                        if Ys[k+1]-Ys[i] > 0.075 and k+1 != len(Xs)-1: 
                            Tracts[ID]['dir'] = 'R'
                            S = slope1/-slope2
                            Y = Ys[k+1]-Ys[i]
                            Tracts[ID]['score'] = max(S,20*Y)
                            Oriented = True 
                        
                        if Ys[i]-Ys[k+1] > 0.075 and i !=0 :
                            if Oriented == False : 
                                Tracts[ID]['dir'] = 'L'
                                S = slope2/slope1
                                Y = Ys[k+1]-Ys[i]
                                Tracts[ID]['score'] = min(S,20*Y)
                            else  : 
                                Tracts[ID]['dir'] = '?'
                                Tracts[ID]['score'] = 0
                        
                        if slope1*1.5< -slope2: 
                            if Oriented == False : 
                                Tracts[ID]['dir'] = 'L'
                                S = slope2/slope1
                                Y = Ys[k+1]-Ys[i]
                                Tracts[ID]['score'] = min(S,20*Y)
                            else  : 
                                Tracts[ID]['dir'] = '?'
                                Tracts[ID]['score'] = 0
                        
                        
                        if 'dir' not in Tracts[ID].keys() :
                            Tracts[ID]['dir'] = '?'
                            Tracts[ID]['score'] = 0
                        ID+=1
                        break
                    rest+=jump[k]

        for k in range(0, len(Xs)-2) :
            if k not in Counted : 
                if jump[k] < -MinJump or (jump[k]+jump[k+1]<-MinJump and jump[k]>=-MinJump and jump[k+1]>=-MinJump ): 
                    endTract = Xs[k+1]
                    rest = 0    
                    for i in range(0, k)[::-1] :
                        if jump[i]+rest > MinJump*0.6:
                            startTract = Xs[i]
                            conflict = False
                            maxi = Ys[i:k+1].argmax()
                            slope1 = (Ys[i+maxi]-Ys[i])/(Xs[i+maxi]-Xs[i])*100000
                            slope2 = (Ys[k+1]-Ys[i+maxi])/(Xs[k+1]-Xs[i+maxi])*100000
                            S = slope2/slope1
                            if k+1 == len(Xs)-1 or i == 0 :
                                Y = 0
                            else : Y = Ys[k+1]-Ys[i] 
                            score = min(S,20*Y)                  
                             
                            for item in Tracts : 
                                if startTract == Tracts[item]['start'] : 
                                    ID = item
                                    if  Tracts[item]['dir'] =='R' : 
                                        if -score*2 > Tracts[item]['score'] :
                                            if -score < 2*Tracts[item]['score'] :
                                                conflict = True
                                            else : conflict = False
                                        if -score*2 <= Tracts[item]['score'] : conflict = 'Solved'
                                    if  Tracts[item]['dir'] =='?' :
                                        conflict = True
                                    if  Tracts[item]['dir'] =='L' :
                                        if Tracts[item]['score'] < score : conflict = 'Solved'
                                        else :
                                            del Tracts[item]
                                    break
                                
                                if startTract < Tracts[item]['start']:
                                    ID = item
                                    if endTract <= Tracts[item]['start']:
                                        for it2 in range(item, max(Tracts.keys())+1)[::-1] : 
                                            Tracts[it2+1] = Tracts[it2]
                                        break
                                    else : 
                                        if  Tracts[item]['dir'] =='R' : 
                                            if -score*2 > Tracts[item]['score'] :
                                                if -score < 2*Tracts[item]['score'] :conflict = True
                                                else : conflict = False
                                            if -score*2 <= Tracts[item]['score'] : conflict = 'Solved'
                            if Tracts!={} and startTract > Tracts[max(Tracts.keys())]['start']:
                                ID=max(Tracts.keys())+1

                            if conflict == False: 
                                Tracts[ID] = {}
                                Tracts[ID]['start'] = startTract
                                Tracts[ID]['end'] = endTract
                                if Ys[i]-Ys[k+1] > 0.075 or slope1*1.5< -slope2:
                                    Tracts[ID]['dir'] = 'L'
                                    Tracts[ID]['score'] = min(S,20*Y)
                                else: 
                                    Tracts[ID]['dir'] = '?'
                                    Tracts[ID]['score'] = 0
                                Tracts[ID]['JumpScore'] = max((-jump[k]-MinJump)*10, (-jump[k]-jump[k+1]-MinJump)*10, (jump[i]+rest-MinJump)*10)

                            if conflict == True: 
                                Tracts[ID] = {}
                                Tracts[ID]['start'] = startTract
                                Tracts[ID]['end'] = endTract
                                Tracts[ID]['dir'] = '?'
                                Tracts[ID]['score'] = 0
                                Tracts[ID]['JumpScore'] = max((-jump[k]-MinJump)*10, (-jump[k]-jump[k+1]-MinJump)*10, (jump[i]+rest-MinJump)*10)

                            ID+=1
                            break
                        rest+=jump[i]
        
        Finish = False
        while Finish == False: # removing duplicates
            TractCopy = Tracts.copy()    
            Finish = True
            for item in TractCopy: 
                for item2 in TractCopy:
                    if item != item2 and item in Tracts: 
                        if  TractCopy[item]['start'] >= TractCopy[item2]['start'] and TractCopy[item]['start'] < TractCopy[item2]['end'] :
                            Finish = False
                            if abs(TractCopy[item]['score']) > abs(TractCopy[item2]['score']) :
                                del Tracts[item2]
                                for it in range(item2,len(Tracts)):
                                    Tracts[it] = Tracts[it+1]
                            else : 
                                del Tracts[item]
                                for it in range(item,len(Tracts)):
                                    Tracts[it] = Tracts[it+1]
        return Tracts  

def DetectInits(Tracts, Xs, X, Yr, LowPlateau,MinDist, score, jumpscore): 
    Initiations = []
    if Tracts !={} :            
        for ID in range(min(Tracts.keys()), max(Tracts.keys())):
            if abs(Tracts[ID+1]['JumpScore']) > jumpscore/3.0 and abs(Tracts[ID]['JumpScore']) > jumpscore/3.0 :
                if (Tracts[ID]['dir'] == 'L' and Tracts[ID]['score']<-score  and (Tracts[ID+1]['dir'] == 'R' or (Tracts[ID+1]['dir'] != 'R'and abs(Tracts[ID+1]['score'])<score and abs(Tracts[ID+1]['JumpScore']) > jumpscore and abs(Tracts[ID]['JumpScore']) > jumpscore))) or (Tracts[ID+1]['dir'] == 'R' and Tracts[ID+1]['score']>score and (Tracts[ID]['dir'] == 'L' or (Tracts[ID]['dir'] != 'L'and abs(Tracts[ID]['score'])>-score and abs(Tracts[ID+1]['JumpScore']) > jumpscore and abs(Tracts[ID]['JumpScore']) > jumpscore))):
                        INIT = (Tracts[ID+1]['start'] + Tracts[ID]['end'])/2
                        I1 = (np.abs(X-Tracts[ID]['end'])).argmin()
                        I2 = (np.abs(X-Tracts[ID+1]['start'])).argmin()
                        M = []
                        for x in range(min(I1, I2), max(I1,I2)) :
                            M.append(Yr[x])
                        if  M != [] and np.array(M).mean()< LowPlateau and Tracts[ID+1]['start'] - Tracts[ID]['end'] > MinDist:
                            Initiations.append([INIT,Tracts[ID]['start'], Tracts[ID]['end'],Tracts[ID+1]['start'], Tracts[ID+1]['end']])
                            if Tracts[ID]['dir'] == '?' or Tracts[ID]['dir']=='R':  # correct tract direction when initiation is detected
                                Tracts[ID]['dir'] = 'L' 
                                Tracts[ID]['score'] = 5
                                Tracts[ID]['JumpScore'] = 5
                            if Tracts[ID+1]['dir']== '?' or Tracts[ID+1]['dir']=='L':
                                Tracts[ID+1]['dir'] = 'R'   
                                Tracts[ID+1]['score'] = 5
                                Tracts[ID+1]['JumpScore'] = 5  
    return Initiations, Tracts


def DetectTermsFilter3(Tracts, Xs, X, Yr, LowPlateau, jumpscore, score):
    Terminaisons = []
    if Tracts !={} :  
        for ID in range(min(Tracts.keys()), max(Tracts.keys())):
            if abs(Tracts[ID+1]['JumpScore']) > jumpscore/3.0 and abs(Tracts[ID]['JumpScore']) > jumpscore/3.0 :
                if (Tracts[ID+1]['dir'] == 'L' and Tracts[ID+1]['score']<-score  and (Tracts[ID]['dir'] == 'R' or (Tracts[ID]['dir'] != 'R'and abs(Tracts[ID]['score'])<score and abs(Tracts[ID]['JumpScore']) > jumpscore and abs(Tracts[ID+1]['JumpScore']) > jumpscore))) or (Tracts[ID]['dir'] == 'R' and Tracts[ID]['score']>score and (Tracts[ID+1]['dir'] == 'L' or (Tracts[ID+1]['dir'] != 'L'and abs(Tracts[ID+1]['score'])>-score and abs(Tracts[ID]['JumpScore']) > jumpscore and abs(Tracts[ID+1]['JumpScore']) > jumpscore))):
                    TERM = (Tracts[ID]['start'] + Tracts[ID+1]['end'])/2
                    B1 = (np.abs(X-Tracts[ID]['start'])).argmin()
                    B2 = (np.abs(X-(Tracts[ID]['start']-500))).argmin()
                    A1 = (np.abs(X-Tracts[ID+1]['end'])).argmin()
                    A2 = (np.abs(X-(Tracts[ID+1]['end']+500))).argmin()
                    if (Yr[B1]< LowPlateau or Yr[B2]< LowPlateau) and (Yr[A1]< LowPlateau or Yr[A2]< LowPlateau) : 
                        Terminaisons.append([TERM, Tracts[ID]['start'],Tracts[ID]['end'],Tracts[ID+1]['start'], Tracts[ID+1]['end']]) 
    return Terminaisons


def ExportBedForks(Tracts, output,chrom, Fasta, readname,strand, jumpscore, score):
    for ID in Tracts:
        if Tracts[ID]['dir']!='?' and Tracts[ID]['JumpScore'] >= jumpscore and Tracts[ID]['score']>=score :
            output.write( "%s\t%d\t%d\t%s\t%s\t%s\t%s\n"%(chrom,Tracts[ID]['start'], Tracts[ID]['end'], Tracts[ID]['dir'], Fasta, readname,strand)) 

def ExportBedForksNoFilter(Tracts, output,chrom, Fasta, readname,strand):
    for ID in Tracts:
        if Tracts[ID]['dir']!='?' :
            output.write( "%s\t%d\t%d\t%s\t%s\t%s\t%s\t%.2f\t%.2f\n"%(chrom,Tracts[ID]['start'], Tracts[ID]['end'], Tracts[ID]['dir'], Fasta, readname,strand,  Tracts[ID]['JumpScore'], Tracts[ID]['score']))         

def ExportInits(InitsTable, output,chrom, Fasta, readname,strand):
    for I in InitsTable :
        output.write("%s\t%d\t%d\t%d\t%d\t%d\t%s\t%s\t%s\n" % (chrom, I[0], I[1], I[2], I[3], I[4], Fasta, readname,strand)) 

def Attributes(l):
    attribute = l.split('{')[1].split(',')
    Attri = {}
    for i in range(0,len(attribute)) : 
        Attri[attribute[i].split(':')[0].split('\'')[1]] = attribute[i].split(':')[1].split('}')[0]
    strand = Attri['mapped_strand'].split('\'')[1]
    start = float(Attri['mapped_start'])
    end = float(Attri['mapped_end'])
    chrom = Attri['mapped_chrom'].split('\'')[1]
    return start, end, strand, chrom

def PlotTracts(Tracts, color1,color2,color3,y, head=500):
    if Tracts !={} :  
        for ID in Tracts  : 
            if abs(Tracts[ID]['score']) >= 2 :
                if Tracts[ID]['dir'] =='R' :
                    plt.arrow(Tracts[ID]['start'],y, Tracts[ID]['end']-Tracts[ID]['start']-head, 0, fc=color1, ec=color1,head_width=0.1, head_length=head, lw=1.5)
                if  Tracts[ID]['dir'] =='L' :
                    #print('LL1', Tracts[ID]['start'],Tracts[ID]['end'])
                    plt.arrow(Tracts[ID]['end'],y, -Tracts[ID]['end']+Tracts[ID]['start']+head, 0, fc=color1, ec=color1,head_width=0.1, head_length=head, lw=1.5)
            else :
                #print 'score', Tracts[ID]['score']
                if Tracts[ID]['dir'] =='R' :
                    plt.arrow(Tracts[ID]['start'],y, Tracts[ID]['end']-Tracts[ID]['start']-head, 0, fc=color2, ec=color2,head_width=0.1, head_length=head, lw=1.5) #, linestyle = 'dashed')
                    #print('RR', Tracts[ID]['start'],Tracts[ID]['end'])
                if  Tracts[ID]['dir'] =='L' :
                    plt.arrow(Tracts[ID]['end'],y, -Tracts[ID]['end']+Tracts[ID]['start']+head, 0, fc=color2, ec=color2,head_width=0.1, head_length=head, lw=1.5) #, linestyle = 'dashed')
                if Tracts[ID]['dir'] =='?' :
                    plt.arrow(Tracts[ID]['end'],y, -Tracts[ID]['end']+Tracts[ID]['start'], 0, fc=color2, ec=color2,head_width=0.0, head_length=0, lw=1.5) #, linestyle = 'dashed')

def PlotInits(Inits, color,y):
    if Inits!= None:
        for i in Inits:
            plt.plot(i[0], y, color,marker = '^', markersize=8, alpha = 1)
            
def PlotTerms(Inits, color,y):
    if Inits!= None:
        for i in Inits:
            plt.plot(i[0], y, color,marker = 'v', markersize=8, alpha = 1)
