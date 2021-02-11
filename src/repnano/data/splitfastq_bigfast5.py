#USAGE : python3 splitfastq_bigfast5.py PATHtoFAST5folder PATHtoFASTQfile.fastq 


import os
import h5py
import sys
import pickle

def extractReads(name):
    # name : nom du fichier fast5
    Reads = []
    h5 = h5py.File(name, "r")
    for ik, k in enumerate(h5.keys()):
        # print(k)
        h5p = h5[k]
        ch = int(str(h5p["channel_id"].attrs["channel_number"])[2:-1])
        readn = int(h5p["Raw"].attrs["read_number"])
        ID = "ch%iread%i" % (ch, readn)
        #print('1', ID)
        Reads.append(ID)
    h5.close()
    return Reads

folder = sys.argv[1]
fastq = sys.argv[2]

Fast5Files=[]

for root, dirs, files in os.walk(folder) : 
    for FILE in files : 
        if FILE.split('.')[-1]=='fast5':
            Fast5Files.append(root+'/'+FILE)

            
######### Read big fast5 to get the name of the reads, it takes time, to do only once! 

if os.path.isfile(fastq.split('.')[0]+'.PathTo') : 
    with open(fastq.split('.')[0]+'.PathTo', 'rb') as f1:
        PathTo = pickle.load(f1)
        
else:       
    PathTo = {}
    for FILE in Fast5Files:
        #print('extracting reads from '+FILE)
        g = open(FILE.split('.')[0]+'.list', 'w')
        Reads = extractReads(FILE)
        print('writing list of reads in '+FILE.split('.')[0]+'.list')
        for r in Reads : 
            PathTo[r] = FILE
            g.write(r+'\n')
        g.close()  
    # save PathTo
    with open(fastq.split('.')[0]+'.PathTo', 'wb') as f1:
        pickle.dump(PathTo, f1)

# split fastq [and write index for DNAscent]

for FILE in Fast5Files :
    Number=FILE.split('.')[0].split('_')[-1]
    #print(FILE, Number)
    open(fastq.split('.')[0]+'_'+Number+'.fastq', 'w')
    open(fastq.split('.')[0]+'_'+Number+'.index', 'w')

f = open(fastq,'r')
s = f.readline()
t=0
while s != '':
    if s[0:3] == '@ch': #@ch405_read46_template_pass_FAK90124
        ID = s.split('@')[1].split('_')[0]+s.split('@')[1].split('_')[1]
        if ID in PathTo :
            FileN =  PathTo[ID].split('.')[0].split('_')[-1]
            g = open(fastq.split('.')[0]+'_'+FileN+'.fastq', 'a')
            g.write(s)
            h= open(fastq.split('.')[0]+'_'+FileN+'.index', 'a')
            t+=1
            #print(str(t)+' '+fastq.split('.')[0]+'_'+FileN+'.fastq')
            ss = s.split()
            h.write(ss[0].split('@')[1]+'\t'+PathTo[ID]+'\n')
            s = f.readline()
            while s[0:3] != '@ch' : 
                g.write(s)
                s = f.readline()
            g.close()
            h.close()
        else : 
            s = f.readline()
    else :
        s = f.readline()
f.close()

