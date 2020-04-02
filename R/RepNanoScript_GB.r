### Script use for RepNanoPaper
### 20200124

path0 <- "~/work/Ori/RepNanoPaper/"
# path where the data are stored
pathD <- paste0(path0,"Data_GB/")
# path where to store the results and figures
pathR <- paste0(path0,"Results_GB/")

# Home made functions used for the script
source(paste0(path0,"RepNanoFunction_GB.r"))

# required library
library(GenomicRanges)
library(GenomicAlignments)
library(rtracklayer)
library(ggpubr)
library(ggcorrplot)
library(Hmisc)
suppressMessages(library(tidyverse))
library(colorRamps)
library(DECIPHER)
library(seqLogo)
library(caTools)
theme_set(theme_bw(base_size = 16))

# Import Yeast Genome and removing mitochondrial DNA
library("BSgenome.Scerevisiae.UCSC.sacCer1")
genome <- BSgenome.Scerevisiae.UCSC.sacCer1
seqinfsc1 <- seqinfo(genome)
seqlevels(seqinfsc1) <- seqlevels(seqinfsc1)[1:16]
library("BSgenome.Scerevisiae.UCSC.sacCer3")
genome <- BSgenome.Scerevisiae.UCSC.sacCer3
seqinf <- seqinfo(genome)
seqlevels(seqinf) <- seqlevels(seqinf)[1:16]

# Importing ARS
# beware, I had to remove ' in the 2 Y'-ARS to avoid error in the import
# file copied form oriDB webpage
newARS <- read.table(paste0(pathD,"OriDB20190614.txt"),sep="\t",as.is=T)
newARS <- newARS[-1,]
newARS.RO <- sapply(newARS[,1], function(x) paste0('chr',as.character(as.roman(as.numeric(x)))))
ARS <- GRanges(seqnames=newARS.RO,ranges=IRanges(start=as.numeric(newARS[,2]),end=as.numeric(newARS[,3])),strand="*",name=newARS[,4],altname=newARS[,5],status=newARS[,6],seqinfo=seqinf)
ARS[ARS$name==""]$name <- ARS[ARS$name==""]$status

export(ARS,con=paste0(pathR,"ARSfromOriDB.bed"))
ConfARS <- ARS[!ARS$name %in% c('Likely','Dubious')]
LikARS <- ARS[ARS$name %in% c('Likely')]
DubARS <-  ARS[ARS$name %in% c('Dubious')]
ARSc <- resize(ARS,fix="center",width=1)
# 829 ARS, 410 Conf, 216 Likely, 203 Dubious

# Importing Oriented BrdU segments
toread <- dir(paste0(pathD,"forks_TM"))
seg.tm <- lapply(toread, function(i) read.table(paste0(pathD,"forks_TM/",i),sep="\t",as.is=T))
seg.tm2 <- lapply(seg.tm, function(x) {x[x[,4]=="L",4]="+";x[x[,4]=="R",4]="-";x[,1]=paste0('chr',as.character(as.roman(as.numeric(sapply(as.vector(x[,1]),function(z) strsplit(z,'chr')[[1]][2])))));return(x)})
seg_TM <- lapply(seg.tm2, function(x) GRanges(seqnames=x[,1],ranges=IRanges(start=x[,2],end=x[,3]),strand=x[,4],inf1=x[,5],inf2=x[,6],str.map=x[,7],seqinfo=seqinf))
segBC_TM <- do.call(c,seg_TM[1:3])
segBD_TM <- do.call(c,seg_TM[4:6])
segBCD_TM <- do.call(c,seg_TM)
#44412 forks

toread <- dir(paste0(pathD,"forks_CNN"))
seg.cnn <- lapply(toread, function(i) read.table(paste0(pathD,"forks_CNN/",i),sep="\t",as.is=T))
seg.cnn2 <- lapply(seg.cnn, function(x) {x[x[,4]=="L",4]="+";x[x[,4]=="R",4]="-";x[,1]=paste0('chr',as.character(as.roman(as.numeric(sapply(as.vector(x[,1]),function(z) strsplit(z,'chr')[[1]][2])))));return(x)})
seg_cnn <- lapply(seg.cnn2, function(x) GRanges(seqnames=x[,1],ranges=IRanges(start=x[,2],end=x[,3]),strand=x[,4],inf1=x[,5],inf2=x[,6],str.map=x[,7],seqinfo=seqinf))
segBC_cnn <- do.call(c,seg_cnn[1:3])
segBD_cnn <- do.call(c,seg_cnn[4:6])
segBCD_cnn <- do.call(c,seg_cnn)
#44517 forks


# Importing our OK seq data
#path2bam <- "/Volumes/rce/Results/Xia seq/yeast/data/work/NGS/runs/170220_170518_nextseq/Step_11_bwa/"
#bf <- BamFile(paste0(path2bam,"P11-Yeast2-Recut_S5_all_R1_001_cutadapt_yeast_sorted.bam"))
#gr <- bam2gr(bf)
#seqlevels(gr, pruning.mode="coarse") <- seqlevels(seqinf)
#seqinfo(gr) <- seqinf
#save(gr,file=paste0(pathR,'P11PErawdata.RData'))
# to run only once...
load(paste0(pathR,'P11PErawdata.RData'))
grs <- resize(gr,fix='start', width=1)
# 20723112 reads
# only start are kept to remove read length bias between OH and IW experiments


# Importing Whitehouse OK seq data
# downloaded from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE33786
# we decided to work with the file containing more reads
RawOF2 <- read.table(paste0(pathD,'GSM835651_wt_replicate.bed.gz'),header=F,sep='\t',as.is=T)
RawOF2 <- RawOF2[RawOF2[,1]<17,]
wgr <- GRanges(seqnames=seqlevels(genome)[RawOF2[,1]],
ranges=IRanges(start=RawOF2[,2],end=RawOF2[,3]),strand=Rle(RawOF2[,6]),seqinfo=seqinf)
wgrs <- resize(wgr,fix="start",width=1)
# 42294012 reads


### compute RFD and coverage at nucleotide level
rfd.BCnt.tm <- simpleRFD(segBC_TM,lr=0,na2zero=F,expor=F,OKcheck=F)
rfd.BDnt.tm <- simpleRFD(segBD_TM,lr=0,na2zero=F,expor=F,OKcheck=F)
rfd.BCnt.cnn <- simpleRFD(segBC_cnn,lr=0,na2zero=F,expor=F,OKcheck=F)
rfd.BDnt.cnn <- simpleRFD(segBD_cnn,lr=0,na2zero=F,expor=F,OKcheck=F)
rfd.OHnt <- simpleRFD(grs,lr=0,na2zero=F,expor=T,OKcheck=T,outname=paste0(pathR,'OH'))
rfd.IWnt <- simpleRFD(wgrs,lr=0,na2zero=F,expor=T,OKcheck=T,outname=paste0(pathR,'IW'))
rfd.BCDnt.tm <- simpleRFD(segBCD_TM,lr=0,na2zero=F,expor=T,OKcheck=F,outname=paste0(pathR,'segBCD_tm'))
rfd.BCDnt.cnn <- simpleRFD(segBCD_cnn,lr=0,na2zero=F,expor=T,OKcheck=F,outname=paste0(pathR,'segBCD_cnn'))


# correlation between replicate (BC/BD)
# without 1kb smoothing
cor.rfd(rfd.BCnt.cnn$RFD2, rfd.BDnt.cnn$RFD2)
# 0.7498
cor.rfd(rfd.BCnt.tm$RFD2,rfd.BDnt.tm$RFD2)
# 0.6533
# with 1kb smoothing
cor.rfd( endoapply(rfd.BCnt.cnn$RFD2,runmean,k=1000,align="center",endrule="NA"), endoapply(rfd.BDnt.cnn$RFD2,runmean,k=1000,align="center",endrule="NA"))
# 0.7520
cor.rfd( endoapply(rfd.BCnt.tm$RFD2,runmean,k=1000,align="center",endrule="NA"), endoapply(rfd.BDnt.tm$RFD2,runmean,k=1000,align="center",endrule="NA"))
# 0.6624

# apply runmean on RFD before export and correlation
rfd.listnt <- list(rfd.IWnt$RFD2,rfd.OHnt$RFD2,rfd.BCDnt.cnn$RFD2,rfd.BCDnt.tm$RFD2)
rfd.listnt.rm <- lapply(rfd.listnt, function(x) endoapply(x,runmean,k=1000,align="center",endrule="NA"))
export.file.names <- c("IWrfdntrm","OHrfdntrm","CNNrfdntrm","TMrfdntrm")
dum <- lapply(1:4, function(i) export(rfd.listnt.rm[[i]],con=paste0(pathR,export.file.names[i],".bw")))
mcornt.rm <- sapply(rfd.listnt.rm, function(x) sapply(rfd.listnt.rm, function(y) cor.rfd(x,y)))
colnames(mcornt.rm) <- rownames(mcornt.rm) <- c("IWnt","OHnt","BCDnt.cnn","BCDnt.tm")
ggcorrplot(mcornt.rm,lab=T,lab_size=8)+
	scale_fill_gradient2(limit = c(0,1), low = "blue", high =  "red", mid = "white", midpoint = 0.5)
ggsave(paste0(pathR,"fig2e.PDF"),width=5,heigh=5)

## adding the merged forks
### all forks merged
seg.merge <- read.table(paste0(pathD,"CNN.forks5-TM.forks5_merge"),sep="\t",as.is=T)
x <- seg.merge
x[x[,4]=="L",4]="+";x[x[,4]=="R",4]="-";x[,1]=paste0('chr',as.character(as.roman(as.numeric(sapply(as.vector(x[,1]),function(z) strsplit(z,'chr')[[1]][2])))))
seg_merge <- GRanges(seqnames=x[,1],ranges=IRanges(start=x[,2],end=x[,3]),strand=x[,4],inf1=x[,5],inf2=x[,6],str.map=x[,7],seqinfo=seqinf)
# 58651 forks
rfd.BCDnt.merge <- simpleRFD(seg_merge,lr=0,na2zero=T,expor=F,OKcheck=F,outname=paste0(pathR,'segBCD_merge'))

rfd.BCDnt.merge.rm <- endoapply(rfd.BCDnt.merge$RFD2,runmean,k=1000,align="center",endrule="NA")
export(rfd.BCDnt.merge.rm,con=paste0(pathR,"FORKseq_rfdntrm.bw"))


# Initiation results
ini.merge <- read.table(paste0(pathD,"TM.init5-CNN.init5merge"),sep="\t",as.is=T)
ini.merge.RO <- paste0('chr',as.character(as.roman(as.numeric(sapply(as.vector(ini.merge[,1]),function(z) strsplit(z,'chr')[[1]][2])))))
ini.merge[,1] <- ini.merge.RO
gr.ini <- GRanges(seqnames=ini.merge[,1],ranges=IRanges(start=ini.merge[,2],end=ini.merge[,2]),strand="*",inf1=ini.merge[,7],inf2=ini.merge[,8],str.map=ini.merge[,9],seqinfo=seqinf)
export(gr.ini,con=paste0(pathR,"FS_ini.bed"))
# 4964 initiation events

# Initiation inter-event distances compared to shuffled
## ECDF without clustering
ini.all <- gr.ini
mcols(ini.all) <- NULL
ini.all$dtn <- data.frame(distanceToNearest(ini.all))[,3]
# randomize by shuffling
#set.seed(42)
chnb=16
# find gaps of unassigned bases (N) in the genome
listGaps <- lapply(1:chnb, function(i) {ra= findNgaps(genome[[i]]);if(length(ra)>0) {resu=GRanges(seqnames=seqnames(genome)[i],ranges=ra,seqinfo=seqinfo(genome))}else{resu=GRanges(seqnames=seqnames(genome)[i],ranges=IRanges(start=1,width=1),seqinfo=seqinfo(genome))};return(resu)})
Ngaps=do.call(c,listGaps)
Ngaps2=Ngaps[width(Ngaps)>=20]
# randomize 10 times the initiation
# set.seed(73) not done for the figure generated in the manuscript. small variation can thus occur in median(dtn) and clusters
rd.ini.l <- mclapply(1:10, function(i) trim(suppressWarnings(shuffleGRgen(i,gen=genome,inputGR=ini.all,gap2=Ngaps2,chrlist=1:16))),mc.cores=5L)
# select one of the random
ini.rd <- rd.ini.l[[2]]
mcols(ini.rd) <- NULL
ini.rd$dtn <- data.frame(distanceToNearest(ini.rd))[,3]
median(ini.rd$dtn)
### Importing termination events
termi <- read.table(paste0(pathD,"BC-BD_TM.term6-BC-BD_CNN.term6MD2000merge"),sep="\t",as.is=T)
termi.RO <- paste0('chr',as.character(as.roman(as.numeric(sapply(as.vector(termi[,1]),function(z) strsplit(z,'chr')[[1]][2])))))
termi[,1] <- termi.RO
gr.ter <- GRanges(seqnames=termi[,1],ranges=IRanges(start=termi[,2],end=termi[,2]),strand="*",inf1=termi[,7],inf2=termi[,8],str.map=termi[,9],seqinfo=seqinf)
export(gr.ter,con=paste0(pathR,"FS_ter.bed"))
ter.all <- gr.ter
mcols(ter.all) <- NULL
ter.all$dtn <- data.frame(distanceToNearest(ter.all))[,3]
pdf(paste0(pathR,"figS6a.pdf"),width=7,heigh=5)
Ecdf(ini.all$dtn,col="red",q=0.5,xlim=c(0,3500),xlab="Distance to nearest",ylab="ECDF",lwd=2)
Ecdf(ter.all$dtn,col="blue",add=T,q=0.5,lwd=2)
Ecdf(ini.rd$dtn,col="black",add=T,q=0.5,lwd=2)
legend("bottomright",legend=c(paste0("FORK-seq_ini (n=4964, med=",median(ini.all$dtn),")"),paste0("shuffled init (n=4964, med=",median(ini.rd$dtn),")"),paste0("FORK-seq_ter (n=",length(ter.all),", med=",median(ter.all$dtn),")")), text.col=c("red","black","blue"),bty='n')
dev.off()

## Clustering
#clust.list <- c(1,5,10,20,50,100,150,200,300,500,1000,1500,2000,2500,3000,4000,5000,7000,10000,20000,50000,100000)
#ini.all.clust <- function.cluster(ini.all, clust.list0=clust.list,mc=6)
#saveRDS(ini.all.clust,file=paste0(pathR,"Alliniclust.rds"))
#ini.rd.all.clust <- function.cluster(ini.rd, clust.list0=clust.list,mc=6)
#saveRDS(ini.rd.all.clust,file=paste0(pathR,"Allinirdclust.rds"))
ini.all.clust <- readRDS(paste0(pathR,"Alliniclust.rds"))
ini.rd.all.clust <- readRDS(paste0(pathR,"Allinirdclust.rds"))
# this saved version of the cluster of the shuffled results is the one used to generate the figures of the manuscript

#ter.all.clust <- function.cluster(gr.ter,mc=6)
#saveRDS(ter.all.clust,file=paste0(pathR,"Allterclust.rds"))
ter.all.clust <- readRDS(paste0(pathR,"Allterclust.rds"))

#clsutering analysis
clust1 <- sapply(ini.all.clust,function(x) length(x[x$eff==1]))
clust.dim.l <- lapply(c(1,2,3,5,10), function(y) sapply(ini.all.clust,function(x) length(x[x$eff>y])))
dt.clust <- data.frame(clust.list, do.call(cbind,clust.dim.l))
colnames(dt.clust) <- c('clust.dist',paste0("dim_",c(1,2,3,5,10)))
mypal <- rainbow(10)
plot(x=clust.list,clust1,log="xy",ylab="number of clusters",xlab="interevent distance(bp)",type='b',xlim=c(1,20000),ylim=c(1,5000))
tt <- lapply(2:6, function(i) lines(clust.list,dt.clust[,i],col=mypal[i]))
legend("bottomleft",legend=c("clust.dim=1",paste0("clust.dim>",c(1,2,3,5,10))), text.col=c('black',mypal[2:9]),bty='n')
bp <- boxplot(lapply(ini.all.clust,function(x) width(x[x$eff>1])),outline=F,range=0,log="y",names=clust.list,varwidth=T,ylim=c(1,1e6),ylab=paste0("width(clust,eff>",1,")"));mtext(text=bp$n,1,2,at=1:22,font=4,cex=0.5)

pdf(paste0(pathR,"fig4ab.pdf"),width=10,heigh=7)
plot(x=clust.list,clust1,log="xy",ylab="number of clusters",xlab="interevent distance(bp)",type='b',xlim=c(1,20000),ylim=c(1,5000))
tt <- lapply(2:6, function(i) lines(clust.list,dt.clust[,i],col=mypal[i]))
legend("bottomleft",legend=c("clust.dim=1",paste0("clust.dim>",c(1,2,3,5,10))), text.col=c('black',mypal[2:9]),bty='n')
bp <- boxplot(lapply(ini.all.clust,function(x) width(x[x$eff>1])),outline=F,range=0,log="y",names=clust.list,varwidth=T,ylim=c(1,1e6),ylab=paste0("width(clust,eff>",1,")"));mtext(text=bp$n,1,2,at=1:22,font=4,cex=0.5)
abline(h=1500,lty=3)
dev.off()

clust1rd <- sapply(ini.rd.all.clust,function(x) length(x[x$eff==1]))
clust.dim.lrd <- lapply(c(1,2,3,5,10), function(y) sapply(ini.rd.all.clust,function(x) length(x[x$eff>y])))
dt.clustrd <- data.frame(clust.list, do.call(cbind,clust.dim.lrd))
colnames(dt.clustrd) <- c('clust.dist',paste0("dim_",c(1,2,3,5,10)))
mypal <- rainbow(10)
plot(x=clust.list,clust1rd,log="xy",ylab="number of clusters",xlab="interevent distance(bp)",type='b',xlim=c(1,20000),ylim=c(1,5000))
tt <- lapply(2:6, function(i) lines(clust.list,dt.clustrd[,i],col=mypal[i]))
legend("bottomleft",legend=c("clust.dim=1",paste0("clust.dim>",c(1,2,3,5,10))), text.col=c('black',mypal[2:9]),bty='n')
bp <- boxplot(lapply(ini.rd.all.clust,function(x) width(x[x$eff>1])),outline=F,range=0,log="y",names=clust.list,varwidth=T,ylim=c(1,1e6),ylab=paste0("width(clust,eff>",1,")"));mtext(text=bp$n,1,2,at=1:22,font=4,cex=0.5)

pdf(paste0(pathR,"fig4cd.pdf"),width=10,heigh=7)
plot(x=clust.list,clust1rd,log="xy",ylab="number of clusters",xlab="interevent distance(bp)",type='b',xlim=c(1,20000),ylim=c(1,5000))
tt <- lapply(2:6, function(i) lines(clust.list,dt.clustrd[,i],col=mypal[i]))
legend("bottomleft",legend=c("clust.dim=1",paste0("clust.dim>",c(1,2,3,5,10))), text.col=c('black',mypal[2:9]),bty='n')
bp <- boxplot(lapply(ini.rd.all.clust,function(x) width(x[x$eff>1])),outline=F,range=0,log="y",names=clust.list,varwidth=T,ylim=c(1,1e6),ylab=paste0("width(clust,eff>",1,")"));mtext(text=bp$n,1,2,at=1:22,font=4,cex=0.5)
abline(h=1500,lty=3)
dev.off()

#Export clusters as bed for figure S6b
dum <- lapply(1:length(clust.list), function (i) export(ini.all.clust[[i]],con=paste0(pathR,"bed_cluster/clust_",clust.list[i],".bed")))
dum <- lapply(1:length(clust.list), function (i) export(ini.rd.all.clust[[i]],con=paste0(pathR,"bed_cluster/clustrd_",clust.list[i],".bed")))
#add a 5O nt margin to look better for fig sup clustering
dum <- lapply(1:length(clust.list), function (i) export(ini.all.clust[[i]]+50,con=paste0(pathR,"bed_cluster/clust50_",clust.list[i],".bed")))
dum <- lapply(1:length(clust.list), function (i) export(ini.rd.all.clust[[i]]+50,con=paste0(pathR,"bed_cluster/clustrd50_",clust.list[i],".bed")))
#

### Selecting the 1.5kb clustering results
ini.clust <- ini.all.clust[[12]]
ini.clust$score <- ini.clust$eff
#export(coverage(ini.clust, weight=ini.clust$eff), con=paste0(pathR,"FS_ini_clust.bw"))
#export(ini.clust,con=paste0(pathR,"FS_ini_clust.bed"))
summary(width(ini.clust[ini.clust$eff>1]))
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#     34     944    1579    1729    2355    7701 
table(ini.clust$eff)
#  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23 
#362  73  41  26  28  23  20  10  12  13  12  16  15  11   4   7   6   8   9  11  11   6  12 
# 24  25  26  27  28  29  30  31  32  33  37  39  42  45 140 
#  4   7   7   3   2   2   1   5   3   2   3   1   1   1   1 
summary(ini.clust[ini.clust$eff>1 & ini.clust$eff<80]$eff)
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#   2.00    3.00    7.00   10.73   17.00   45.00 

ini.cl <- GRanges(seqnames=seqnames(ini.clust), ranges=IRanges(start=ini.clust$med, width=1),seqinfo=seqinf,eff=ini.clust$eff)
ini.cl$dtac <- data.frame(distanceToNearest(ini.cl,ARSc))[,3]
ini.cl$eff.cl <- cut(ini.cl$eff,c(-Inf,1,4,12,+Inf),include.lowest = T, labels=c("spo","2-4","5-12","13+"))
ini.cl$eff.spo <-cut(ini.cl$eff,c(-Inf,1,+Inf),include.lowest = T, labels=c("spo","clust")) 

ini.clustrd <- ini.rd.all.clust[[12]]
ini.clustrd$score <- ini.clustrd$eff
#export(coverage(ini.clustrd, weight=ini.clustrd$eff), con=paste0(pathR,"FS_ini_clustrd.bw"))
#export(ini.clustrd,con=paste0(pathR,"FS_ini_clustrd.bed"))

ini.clrd <- GRanges(seqnames=seqnames(ini.clustrd), ranges=IRanges(start=ini.clustrd$med, width=1),seqinfo=seqinf,eff=ini.clustrd$eff)
ini.clrd$dtac <- data.frame(distanceToNearest(ini.clrd,ARSc))[,3]
ini.clrd$eff.cl <- cut(ini.clrd$eff,c(-Inf,1,4,12,+Inf),include.lowest = T, labels=c("spo","2-4","5-12","13+"))
ini.clrd$eff.spo <-cut(ini.clrd$eff,c(-Inf,1,+Inf),include.lowest = T, labels=c("spo","clust")) 
table(ini.cl$eff.cl)
# spo  2-4  5-12  13+ 
# 362  140  134   143 
table(ini.clrd$eff.cl)
# spo  2-4   5-12   13+ 
# 1420 1123  110    0
ini.rd$dtac <- data.frame(distanceToNearest(ini.rd,ARSc))[,3]


Ecdf(ini.cl[ini.cl$eff==1,]$dtac,weight=ini.cl[ini.cl$eff==1,]$eff,xlim=c(0,20000),col=mypal[1],q=0.5,main="clust.dim=1,2-4,5-12,13+ and random",ylab="ECDF",xlab="distance to the nearest ARS center",lwd=2)
eff.cl.list <- levels(ini.cl$eff.cl)
dum <- lapply(2:4, function(i) { 
xdf <- ini.cl[ini.cl$eff.cl==eff.cl.list[i],]
Ecdf(xdf$dtac,weight=xdf$eff,col=mypal[i],q=0.5,add=T,lwd=2)
})
Ecdf(ini.rd$dtac,add=T,q=0.5,col="blue",lwd=2)
#Ecdf(ini.clrd$dtac,weight=ini.clrd$eff,add=T,q=0.5,col="darkblue",lty=2,lwd=2)

pdf(paste0(pathR,"fig5a.pdf"),width=10)
Ecdf(ini.cl[ini.cl$eff==1,]$dtac,weight=ini.cl[ini.cl$eff==1,]$eff,xlim=c(0,20000),col=mypal[1],q=0.5,main="clust.dim=1,2-4,5-12,13+ and random",ylab="ECDF",xlab="distance to the nearest ARS center",lwd=2)
eff.cl.list <- levels(ini.cl$eff.cl)
dum <- lapply(2:4, function(i) { 
xdf <- ini.cl[ini.cl$eff.cl==eff.cl.list[i],]
Ecdf(xdf$dtac,weight=xdf$eff,col=mypal[i],q=0.5,add=T,lwd=2)
})
Ecdf(ini.rd$dtac,add=T,q=0.5,col="blue",lwd=2)
dev.off()

### Figure 5b (RFD at ini)
fov1 <- findOverlaps(ini.clust,ini.all)
ini.all$eff <- sapply(seq_along(ini.all), function(x) ini.cl[queryHits(fov1)[subjectHits(fov1)==x]]$eff)
ini.all$eff.cl <- cut(ini.all$eff,c(-Inf,1,4,12,+Inf),include.lowest = T, labels=c("spo","2-4","5-12","13+"))

cv <- rfd.BCDnt.merge$RFD2

feat <- ini.cl
feat <- feat[overlapsAny(feat,as(seqinfo(genome),'GRanges')-10000)]
# let also exclude rDNA region
feat <- feat[!overlapsAny(feat,GRanges("chrXII",ranges=IRanges(start=451000,end=469000),strand="*"))]
feat2 <- resize(feat,fix="center",width=20001)
eff.cl.list <- levels(ini.all$eff.cl)
feat3 <- feat2[which(feat2$eff.cl==eff.cl.list[1])]
profeat <- cv[feat3]
profeat2 <- RleList2matrix(profeat)
toplot <- colMeans(profeat2,na.rm=T)
mypal <- rainbow(10)
xlarg <-10000
ylableg <- 'mean RFD'
pdf(file=paste0(pathR,"fig5b.pdf"),width=10)
plot(toplot[(10001L-xlarg):(10001L+xlarg)],x=((-xlarg):xlarg),lwd=2,type='l',xlab="Distance init cluster center",col='red',yaxs='i',xaxs='i',ylim=c(-0.8,0.8),cex.axis=1.3,cex.lab=1.4,font=2,font.lab=2,ylab=ylableg)
abline(v=0,lty=2)
abline(h=0,lty=2)
dum <- lapply(2:4, function(i)
	{
feat3 <- feat2[which(feat2$eff.cl==eff.cl.list[i])]
profeat <- cv[feat3]
profeat2 <- RleList2matrix(profeat)
toplot <- colMeans(profeat2,na.rm=T)
lines(toplot[(10001L-xlarg):(10001L+xlarg)],x=((-xlarg):xlarg),lwd=2,type='l',col=mypal[2*(i-1)+3])
	})
legend("topleft",legend=eff.cl.list,text.col=mypal[c(1,5,7,9)],bty="n",cex=1.4)
dev.off()

### Computing OEM like in McGuffee et al on 10kb windows
cvL <- rfd.BCDnt.merge$cv_L
cvR <- rfd.BCDnt.merge$cv_R
cvT <- rfd.BCDnt.merge$cv
cvLn <- cvL/cvT
cvRn <- cvR/cvT
cvLn[is.na(cvLn)] <- 0
cvRn[is.na(cvRn)] <- 0
win=10000
cvLs2 <- endoapply(cvLn, function(x) Rle(caTools::runmean(x,win,align="left",endrule="NA")))
oem2 <- endoapply(cvLs2, function(cv) c(Rle(rep(NA,(win))),cv)-c(cv,Rle(rep(NA,(win)))))
oem_FS <- endoapply(oem2, function(x) x[1:(length(x)-win)])
export(oem_FS, con=paste0(pathR,"FS_oem10k.bw"))


ini.all$oem <- as.numeric(oem_FS[ini.all])
tbOEM.ini <- enframe(ini.all$oem,value="OEM",name=NULL)
tbOEM.ini$type <- "f_ini"
tbOEM.ini$eff <- ini.all$eff
tbOEM.ini$eff.cl <- ini.all$eff.cl
tbOEM.ini$type <- "eff"
tbOEM.ini[tbOEM.ini$eff.cl=="spo",]$type <- "spo"
ggplot(tbOEM.ini)+geom_histogram(aes(x=OEM,y=..count..,fill=type),position="dodge",binwidth=0.1,color="black")+xlim(c(-1,1))
ggsave(paste0(pathR,"supfig_S7c.PDF"),width=5,heigh=5)

sum(ini.all$oem>0,na.rm=T)/length(ini.all)
# 89%
sum(ini.all[ini.all$eff==1]$oem>0,na.rm=T)/length(ini.all[ini.all$eff==1])
# 40%
sum(ini.all[ini.all$eff>1]$oem>0,na.rm=T)/length(ini.all[ini.all$eff>1])
# 92%
### control rd
ini.rd$oem <- as.numeric(oem_FS[ini.rd])
sum(ini.rd$oem>0,na.rm=T)/length(ini.rd)
# 42%


### timing distribution

t1 <- import(paste0(pathD,"GSM1180749_T9475_Illumina_normalised.wig"))
t1.new <- GRanges(seqnames=seqnames(t1), ranges=IRanges(start=start(t1),end=start(t1)),strand=strand(t1),score=t1$score,seqinfo=seqinfsc1)
t1.new$timing <- 1-(t1.new$score-min(min(t1.new$score,na.rm=TRUE)))/(max(max(t1.new$score,na.rm=TRUE))-min(min(t1.new$score,na.rm=TRUE)))
t1sc3 <- unlist(liftOver(t1.new,import.chain("~/work/Ori/sacCer1ToSacCer3.over.chain")))
seqinfo(t1sc3) <- seqinf

bs <- 1000
bingen <- tileGenome(seqinf,tilewidth=bs, cut.last.tile.in.chrom=T)
ol1 <- findOverlaps(t1sc3,bingen)
bingen$score <- sapply(1:length(bingen), function(x) mean(t1sc3[queryHits(ol1)[subjectHits(ol1)==x]]$timing,na.rm=T))
RTsc3 <- bingen
RTsc3$timing <- RTsc3$score
export(coverage(RTsc3,weight=RTsc3$timing), con=paste0(pathR,"RTsc3.bw"))

ini.all$timing <- sapply(as(ini.all,"GRangesList"), function(x) RTsc3[overlapsAny(RTsc3,x)]$timing)
ini.spo <- ini.all[ini.all$eff==1]
ini.all.clust <- ini.all[!overlapsAny(ini.all,ini.spo)]
ini.all.clust$group <- "all.clust"
df.timing <- data.frame(mcols(RTsc3))
df.timing$group <- "all.genome"
df.timing <- df.timing[,-1]
ini.all$group <- "all.init"
ini.spo$group <- "sporadic"
ini.all.clust <- ini.all[!overlapsAny(ini.all,ini.spo)]
ini.all.clust$group <- "all.clust"
df.timing <- rbind.data.frame(df.timing,data.frame(mcols(ini.all.clust))[,c("timing","group")],data.frame(mcols(ini.spo))[,c("timing","group")])
densityplot(df.timing$timing,plot.point=F,ref=T,groups=df.timing$group,from=0,to=1,ylab="Distribution Density",xlab="Replication Timing",lwd=2)
densityplot(df.timing$timing,plot.point=F,ref=T,groups=df.timing$group,from=0,to=1,bw=0.07,ylab="Distribution Density (bw=0.07)",xlab="Replication Timing",lwd=2)
densityplot(df.timing$timing,plot.point=F,ref=T,groups=df.timing$group,from=0,to=1,bw=0.1,ylab="Distribution Density (bw=0.1)",xlab="Replication Timing",lwd=2)

pdf(file=paste0(pathR,'fig5c.pdf'),width=10)
densityplot(df.timing$timing,plot.point=F,ref=T,groups=df.timing$group,from=0,to=1,bw=0.07,ylab="Distribution Density (bw=0.07)",xlab="Replication Timing",lwd=2)
dev.off()

#### ACS enrichment
acs <- read.table(paste0(pathD,"GSM424494_acs_locations.bed.gz"),sep="\t",as.is=T)
acs.ch <- Rle(paste0('chr',as.character(as.roman(as.numeric(acs[,1])))))
acs.st <- apply(acs[,c(2,3)],1,min)
acs.en <- apply(acs[,c(2,3)],1,max)
acs.new <- GRanges(seqnames=acs.ch, ranges=IRanges(start=acs.st,end=acs.en),strand=acs[,6],score=acs[,5],seqinfo=seqinf)
acs.v <- Views(genome,acs.new)
acs.al <- suppressMessages(AlignSeqs(DNAStringSet(acs.v),verbose=F))
acs.cm <- consensusMatrix(acs.al)[1:4,14:46]
acs.norm <- t(apply(acs.cm,1, function(x) x/colSums(acs.cm)))
acs.pwm <- makePWM(acs.norm)
mPWMgenome <- matchPWM(acs.norm,genome,min.score = "80%",with.score = T)
length(mPWMgenome)
# 5858 putative ACS in the genome

ARS4mot <- suppressWarnings(resize(ARS,fix="center",width=2001))
ConfARS4mot <- suppressWarnings(resize(ConfARS,fix="center",width=2001))
LikARS4mot <- suppressWarnings(resize(LikARS,fix="center",width=2001))
DubARS4mot <- suppressWarnings(resize(DubARS,fix="center",width=2001))

ini.cl4mot <- resize(ini.cl,fix="center",width=2001)

ARS4mot.motif <- sum(overlapsAny(ARS4mot,mPWMgenome))/length(ARS4mot)
ConfARS4mot.motif <- sum(overlapsAny(ConfARS4mot,mPWMgenome))/length(ConfARS4mot)
LikARS4mot.motif <- sum(overlapsAny(LikARS4mot,mPWMgenome))/length(LikARS4mot)
DubARS4mot.motif <- sum(overlapsAny(DubARS4mot,mPWMgenome))/length(DubARS4mot)

ini.cl4mot.motifbyeffcl <-sapply(levels(ini.cl4mot$eff.cl), function(i) sum(overlapsAny(ini.cl4mot[ini.cl4mot$eff.cl==i],mPWMgenome))/length(ini.cl4mot[ini.cl4mot$eff.cl==i]))
names(ini.cl4mot.motifbyeffcl) <- paste0("FS_ini_",names(ini.cl4mot.motifbyeffcl))

# rd.spo.l <- mclapply(1:1000, function(i) trim(suppressWarnings(shuffleGRgen(i,gen=genome,inputGR=ini.spo,gap2=Ngaps2,chrlist=1:16))),mc.cores=5L)
# saveRDS(rd.spo.l,file=paste0(pathR,"rd.spo.l.rds"))
rd.spo.l <- readRDS(paste0(pathR,"rd.spo.l.rds"))
rd.spo.l4motif <- lapply(rd.spo.l,resize,fix="center",width=2001)
rd.spo.l4motif.motif.l <- do.call(rbind,lapply(rd.spo.l4motif, function(x) sum(overlapsAny(x,mPWMgenome))/length(x)))

toplot2 <- c(ARS4mot.motif,ConfARS4mot.motif,LikARS4mot.motif,DubARS4mot.motif,ini.cl4mot.motifbyeffcl,median(rd.spo.l4motif.motif.l))
names(toplot2)[1:4] <- c("allARS","ConfARS","LikARS","DubARS")
names(toplot2)[9] <- "FS_ini.spo shuffled*1000"
d2plot2 <- data.frame(toplot2,names(toplot2))
tbACS <- enframe(toplot2,name="type",value="ACSPWM")
tbACS$q01 <- c(rep(0,8),quantile(rd.spo.l4motif.motif.l,0.01))
tbACS$q99 <- c(rep(0,8),quantile(rd.spo.l4motif.motif.l,0.99))
ggbarplot(tbACS,x="type",y="ACSPWM",fill="type")+
	geom_errorbar(aes(ymin=q01, ymax=q99), width=.2,position=position_dodge(.9))+
	rremove("x.axis")+
	rremove("x.text")+
	rremove("xlab")+
	rremove("x.ticks")+
	grids(axis="xy")+
	ylim(c(0,1))
ggsave(paste0(pathR,"fig5d.pdf"),width=10)

# ORC and MCM (fig 5ed)
orc <- import(paste0(pathD,"GSM424494_wt_G2_orc_chip_combined_MODLL.bed"))
orc.RO <- paste0('chr',as.character(as.roman(as.numeric(sapply(as.vector(seqnames(orc)),function(z) strsplit(z,'chr')[[1]][2])))))
orc <- GRanges(seqnames=orc.RO, ranges=ranges(orc), seqinfo=seqinf)
mcm <- import(paste0(pathD,"GSM932790_Hu-MCM-Comb-Norm-Peaks-WT.bed"))
seqinfo(mcm) <- seqinf
ini.all$dtORCc <- data.frame(distanceToNearest(ini.all,resize(orc,fix="center",width=1)))[,3]
ini.all$dtMCMc <- data.frame(distanceToNearest(ini.all,resize(mcm,fix="center",width=1)))[,3]

ini.rd$dtORCc <- data.frame(distanceToNearest(ini.rd,resize(orc,fix="center",width=1)))[,3]
ini.rd$dtMCMc <- data.frame(distanceToNearest(ini.rd,resize(mcm,fix="center",width=1)))[,3]

pdf(file=paste0(pathR,'fig5e.pdf'),width=10)
Ecdf(ini.all[ini.all$eff>1]$dtORCc,xlim=c(0,50000),col="red",main="FORKseq eff or spo vs shuffled ini",ylab="ECDF",xlab="distance to the nearest ORC peak center",lwd=2)
Ecdf(ini.all[ini.all$eff==1]$dtORCc,col="blue",add=T,lwd=2)
Ecdf(ini.rd$dtORCc,col="black",add=T,lwd=2)
legend("bottomright",legend=c(paste0("FORK-seq_ini eff (n=",length(ini.all[ini.all$eff>1]),",med=",median(ini.all[ini.all$eff>1]$dtORCc),")"),paste0("FORK-seq_ini spo (n=",length(ini.all[ini.all$eff==1]),",med=",median(ini.all[ini.all$eff==1]$dtORCc),")"),paste0("shuffled ini (n=4964,med=",median(ini.rd$dtORCc),")")), text.col=c("red","blue","black"),bty='n')
dev.off()
pdf(file=paste0(pathR,'fig5f.pdf'),width=10)
Ecdf(ini.all[ini.all$eff>1]$dtMCMc,xlim=c(0,50000),col="red",main="FORKseq eff or spo vs shuffled ini",ylab="ECDF",xlab="distance to the nearest MCM peak center",lwd=2)
Ecdf(ini.all[ini.all$eff==1]$dtMCMc,col="blue",add=T,lwd=2)
Ecdf(ini.rd$dtMCMc,col="black",add=T,lwd=2)
legend("bottomright",legend=c(paste0("FORK-seq_ini eff (n=",length(ini.all[ini.all$eff>1]),",med=",median(ini.all[ini.all$eff>1]$dtMCMc),")"),paste0("FORK-seq_ini spo (n=",length(ini.all[ini.all$eff==1]),",med=",median(ini.all[ini.all$eff==1]$dtMCMc),")"),paste0("shuffled ini (n=4964,med=",median(ini.rd$dtMCMc),")")), text.col=c("red","blue","black"),bty='n')
dev.off()

### Termination data
clust1 <- sapply(ter.all.clust,function(x) length(x[x$eff==1]))
clust.dim.l <- lapply(c(1,2,3,5,10), function(y) sapply(ter.all.clust,function(x) length(x[x$eff>y])))
dt.clust <- data.frame(clust.list, do.call(cbind,clust.dim.l))
colnames(dt.clust) <- c('clust.dist',paste0("dim_",c(1,2,3,5,10)))
mypal <- rainbow(10)

pdf(file=paste0(pathR,'TerCluster.pdf'),width=10)
plot(x=clust.list,clust1,log="xy",ylab="number of clusters",xlab="interevent distance(bp)",type='b',xlim=c(1,20000),ylim=c(1,5000))
tt <- lapply(2:6, function(i) lines(clust.list,dt.clust[,i],col=mypal[i]))
legend("bottomleft",legend=c("clust.dim=1",paste0("clust.dim>",c(1,2,3,5,10))), text.col=c('black',mypal[2:9]),bty='n')
bp <- boxplot(lapply(ter.all.clust,function(x) width(x[x$eff>1])),outline=F,range=0,log="y",names=clust.list,varwidth=T,ylim=c(1,1e6),ylab=paste0("width(clust,eff>",1,")"));mtext(text=bp$n,1,2,at=1:22,font=4,cex=0.5)
abline(h=1500,lty=3)
dev.off()

## select the 1.5kb clusters for terminations
ter.clust <- ter.all.clust[[12]]
table(ter.clust$eff)
#  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23 
#977 365 154  82  44  33  14  13   8   4   6   6   8   3   4   7   4   7   5   1   4   3   2 
# 24  26  27  31  33  35  39  80 
#  1   1   1   1   1   1   1   1 
summary(ter.clust[ter.clust$eff>1 & ter.clust$eff<80]$eff)
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#  2.000   2.000   3.000   4.372   4.000  39.000 
summary(width(ter.clust[ter.clust$eff>1]))
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#      9     600    1223    1841    2260   14976 

ter.cl <- GRanges(seqnames=seqnames(ter.clust), ranges=IRanges(start=ter.clust$med, width=1),seqinfo=seqinf,eff=ter.clust$eff)
ter.cl$eff.cl <- cut(ter.cl$eff,c(-Inf,1,4,12,+Inf),include.lowest = T, labels=c("spo","2-4","5-12","13+"))

fov1 <- findOverlaps(ter.clust,ter.all)
ter.all$eff <- sapply(seq_along(ter.all), function(x) ter.clust[queryHits(fov1)[subjectHits(fov1)==x]]$eff)
ter.all$eff.cl <- cut(ter.all$eff,c(-Inf,1,4,12,+Inf),include.lowest = T, labels=c("spo","2-4","5-12","13+"))

export(coverage(ter.clust, weight=ter.clust$eff), con=paste0(pathR,"FS_ter_clust.bw"))
export(ter.clust,con=paste0(pathR,"FS_ter_clust.bed"))

# fig 6a
cv <- rfd.BCDnt.merge$RFD2
feat <- ter.cl
feat <- feat[overlapsAny(feat,as(seqinfo(genome),'GRanges')-10000)]
# let also exclude rDNA region
feat <- feat[!overlapsAny(feat,GRanges("chrXII",ranges=IRanges(start=451000,end=469000),strand="*"))]
feat2 <- resize(feat,fix="center",width=20001)
eff.cl.list <- levels(ini.all$eff.cl)
feat3 <- feat2[which(feat2$eff.cl==eff.cl.list[1])]
profeat <- cv[feat3]
profeat2 <- RleList2matrix(profeat)
toplot <- colMeans(profeat2,na.rm=T)
mypal <- rainbow(10)
xlarg <-10000
ylableg <- 'mean RFD'
pdf(file=paste0(pathR,"fig6a.pdf"),width=10)
plot(toplot[(10001L-xlarg):(10001L+xlarg)],x=((-xlarg):xlarg),lwd=2,type='l',xlab="Distance to ter cluster center",col='red',yaxs='i',xaxs='i',ylim=c(-0.8,0.8),cex.axis=1.3,cex.lab=1.4,font=2,font.lab=2,ylab=ylableg)
abline(v=0,lty=2)
abline(h=0,lty=2)
dum <- lapply(2:4, function(i)
	{
feat3 <- feat2[which(feat2$eff.cl==eff.cl.list[i])]
profeat <- cv[feat3]
profeat2 <- RleList2matrix(profeat)
toplot <- colMeans(profeat2,na.rm=T)
lines(toplot[(10001L-xlarg):(10001L+xlarg)],x=((-xlarg):xlarg),lwd=2,type='l',col=mypal[2*(i-1)+3])
	})
legend("topright",legend=eff.cl.list,text.col=mypal[c(1,5,7,9)],bty="n",cex=1.4)
dev.off()

# adding Fachinetti data (copied from sup mat)
ter.f <- import(paste0(pathD,"TER_MC2010.bed"))


## adding noem
oem_FS[is.na(oem_FS)] <- 0
noemFS <- Views(oem_FS,oem_FS<0)
gr.noemFS <- do.call(c,lapply(seq_along(noemFS), function(n) GRanges(seqnames=names(noemFS)[n], IRanges(start(noemFS[[n]]),end(noemFS[[n]])), strand='*',seqinfo=seqinf,maxoem=sapply(noemFS[[n]],min))))
gr2.noemFS <- gr.noemFS[width(gr.noemFS)>1000]

cv <- rfd.BCDnt.merge$RFD2

feat <- ter.cl
feat <- feat[overlapsAny(feat,as(seqinfo(genome),'GRanges')-10000)]
# let also exclude rDNA region
feat <- feat[!overlapsAny(feat,GRanges("chrXII",ranges=IRanges(start=451000,end=469000),strand="*"))]
feat2 <- resize(feat,fix="center",width=20001)
eff.cl.list <- levels(ini.all$eff.cl)
feat3 <- feat2[which(feat2$eff.cl==eff.cl.list[1])]
profeat <- cv[feat3]
profeat2 <- RleList2matrix(profeat)
toplot <- colMeans(profeat2,na.rm=T)
mypal <- rainbow(10)
xlarg <-10000
ylableg <- 'mean RFD'
pdf(file=paste0(pathR,"supfig_S7b.pdf"),width=10)
plot(toplot[(10001L-xlarg):(10001L+xlarg)],x=((-xlarg):xlarg),lwd=2,type='l',xlab="Distance init center",col='red',yaxs='i',xaxs='i',ylim=c(-0.8,0.8),cex.axis=1.3,cex.lab=1.4,font=2,font.lab=2,ylab=ylableg)
abline(v=0,lty=2)
abline(h=0,lty=2)
dum <- lapply(2:4, function(i)
	{
feat3 <- feat2[which(feat2$eff.cl==eff.cl.list[i])]
profeat <- cv[feat3]
profeat2 <- RleList2matrix(profeat)
toplot <- colMeans(profeat2,na.rm=T)
lines(toplot[(10001L-xlarg):(10001L+xlarg)],x=((-xlarg):xlarg),lwd=2,type='l',col=mypal[2*(i-1)+3])
	})
feat3 <- resize(ter.f,fix="center",width=20001)
profeat <- cv[feat3]
profeat2 <- RleList2matrix(profeat)
toplot <- colMeans(profeat2,na.rm=T)
lines(toplot[(10001L-xlarg):(10001L+xlarg)],x=((-xlarg):xlarg),lwd=2,type='l',col=mypal[10])
feat3 <- resize(gr.noemFS,fix="center",width=20001)
profeat <- cv[feat3]
profeat2 <- RleList2matrix(profeat)
toplot <- colMeans(profeat2,na.rm=T)
lines(toplot[(10001L-xlarg):(10001L+xlarg)],x=((-xlarg):xlarg),lwd=2,type='l',col="black")

legend("topright",legend=c(eff.cl.list,"TER.Fachinetti","NOEM_FS"),text.col=c(mypal[c(1,5,7,9,10)],"black"),bty="n",cex=0.8)
dev.off()

# distribution OEM at ter
ter.all$oem <- as.numeric(oem_FS[ter.all])
sum(ter.all$oem>0,na.rm=T)/length(ter.all)
#18%
sum(ter.all[ter.all$eff==1]$oem>0,na.rm=T)/length(ter.all[ter.all$eff==1])
# 41%
sum(ter.all[ter.all$eff>1]$oem>0,na.rm=T)/length(ter.all[ter.all$eff>1])
# 12%
tbOEM.ter <- enframe(ter.all$oem,value="OEM",name=NULL)
tbOEM.ter$type <- "FS_ter"
tbOEM.ter$eff <- ter.all$eff
tbOEM.ter$eff.cl <- ter.all$eff.cl
tbOEM.ter$type <- "eff"
tbOEM.ter[tbOEM.ter$eff.cl=="spo",]$type <- "spo"
ggplot(tbOEM.ter)+geom_histogram(aes(x=OEM,y=..count..,fill=type),position="dodge",binwidth=0.1,color="black")+xlim(c(-1,1))
ggsave(paste0(pathR,"supfig_S7d.PDF"),width=5,heigh=5)

### fig comparing BC to BD after clustering
BCD <- sapply(gr.ini$inf1, function(x) strsplit(x,split="_")[[1]][1])
library(tm)
gr.ini$BCD <- removeNumbers(BCD)
gr.test <- sort(gr.ini)
gr.test.l <- split(gr.test,gr.test$BCD)
gr.test.chr <- lapply(gr.test.l, function(x) split(x,seqnames(x)))
clust.dist <- 1500
 
BCD.cl2 <- lapply(gr.test.chr, function(y)
 	{
 	ch.cl <- list(GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList())
 	
 	for (ch in 1:16)
 			{
 			xx <- y[[ch]]
 			mcols(xx) <- NULL
 			cl <- 1
 			ch.cl[[ch]][[cl]] <- xx[1]
 			for (i in (1:(length(xx)-1)))
 		 		{
 		 		if (distance(xx[i],xx[i+1])<clust.dist)
 		 			{
 		 			ch.cl[[ch]][[cl]] <- c(ch.cl[[ch]][[cl]],xx[i+1])
 		 			}else{
 					cl <- cl+1
 					ch.cl[[ch]][[cl]] <- xx[i+1]
 					}
 				}
 			}
 	return(ch.cl)
 	})
 
BCD.cl3 <- lapply(BCD.cl2, function(x) do.call(c,x)) 
BCD.cl4 <- endoapply(BCD.cl3, function(x1) endoapply(x1, function(x) GRanges(seqnames=seqnames(x)[1],ranges=IRanges(start=min(start(x)),end=max(start(x))),med=median(start(x)))))
BCD.cl5 <- lapply(BCD.cl3, function(x1) lapply(x1, function(x) start(x)))
BCD.cl6 <- lapply(BCD.cl4,unlist)
BCD.cl7 <- lapply(BCD.cl5, function(x) sapply(x,length))
BCD.cl8 <- mapply(function(x,y) {x$eff=y;return(x)},BCD.cl6,BCD.cl7)

iniBC.cl <- GRanges(seqnames=seqnames(BCD.cl8[[1]]), ranges=IRanges(start=BCD.cl8[[1]]$med, width=1),seqinfo=seqinf,eff=BCD.cl8[[1]]$eff)
iniBD.cl <- GRanges(seqnames=seqnames(BCD.cl8[[2]]), ranges=IRanges(start=BCD.cl8[[2]]$med, width=1),seqinfo=seqinf,eff=BCD.cl8[[2]]$eff)
iniBCBD.BC <- iniBC.cl[overlapsAny(BCD.cl8[[1]],BCD.cl8[[2]])]
iniBCBD.BD <- iniBD.cl[overlapsAny(BCD.cl8[[2]],BCD.cl8[[1]])]
iniBCBD.inter.dtn <- as.data.frame(distanceToNearest(iniBCBD.BC,iniBCBD.BD))[,3]
iniBCBD.inter.dtacC <- as.data.frame(distanceToNearest(iniBCBD.BC,ARSc))[,3]
pdf(paste0(pathR,"figS9d.pdf"),width=10)
Ecdf(iniBCBD.inter.dtacC, ylab="ECDF",col="black",xlim=c(0,2000),q=c(0.5,0.9),xlab="Distance to closest event",lwd=2)
Ecdf(iniBCBD.inter.dtn,col="red",q=c(0.5,0.9),add=T,lwd=2)

legend("bottomright",legend=c("cluster Rep1/Rep2 to ARS center (n=275,med=227)","cluster Rep1 to cluster Rep2 (n=275,med=219)"), text.col=c("black","red"),bty='n')
dev.off()

### remain to do the tables
### for the distance table
ini.all$dtac <- data.frame(distanceToNearest(ini.all,ARSc))[,3]
fov1 <- findOverlaps(ini.clust,ini.all)
ini.all$dtac.cl <- sapply(seq_along(ini.all), function(x) ini.cl[queryHits(fov1)[subjectHits(fov1)==x]]$dtac)
all.n <- c(sum(ini.all$eff.cl=='spo'),sum(ini.all$eff.cl!='spo'),sum(ini.all$eff.cl=='2-4'),sum(ini.all$eff.cl=='5-12'),sum(ini.all$eff.cl=='13+'))

all.m2 <- c(sum(ini.all[ini.all$eff.cl=="spo"]$dtac<=2000),sum(ini.all[ini.all$eff.cl!="spo"]$dtac<=2000),sum(ini.all[ini.all$eff.cl=="2-4"]$dtac<=2000),sum(ini.all[ini.all$eff.cl=="5-12"]$dtac<=2000),sum(ini.all[ini.all$eff.cl=="13+"]$dtac<=2000))

all.p2 <- c(sum(ini.all[ini.all$eff.cl=="spo"]$dtac>2000),sum(ini.all[ini.all$eff.cl!="spo"]$dtac>2000),sum(ini.all[ini.all$eff.cl=="2-4"]$dtac>2000),sum(ini.all[ini.all$eff.cl=="5-12"]$dtac>2000),sum(ini.all[ini.all$eff.cl=="13+"]$dtac>2000))

all.clm2<- c(sum(ini.all[ini.all$eff.cl=="spo"]$dtac.cl<=2000),sum(ini.all[ini.all$eff.cl!="spo"]$dtac.cl<=2000),sum(ini.all[ini.all$eff.cl=="2-4"]$dtac.cl<=2000),sum(ini.all[ini.all$eff.cl=="5-12"]$dtac.cl<=2000),sum(ini.all[ini.all$eff.cl=="13+"]$dtac.cl<=2000))

all.clp2 <- c(sum(ini.all[ini.all$eff.cl=="spo"]$dtac.cl>2000),sum(ini.all[ini.all$eff.cl!="spo"]$dtac.cl>2000),sum(ini.all[ini.all$eff.cl=="2-4"]$dtac.cl>2000),sum(ini.all[ini.all$eff.cl=="5-12"]$dtac.cl>2000),sum(ini.all[ini.all$eff.cl=="13+"]$dtac.cl>2000))

cl.n <- c(sum(ini.cl$eff.cl=='spo'),sum(ini.cl$eff.cl!='spo'),sum(ini.cl$eff.cl=='2-4'),sum(ini.cl$eff.cl=='5-12'),sum(ini.cl$eff.cl=='13+'))

cl.m2 <- c(sum(ini.cl[ini.cl$eff.cl=="spo"]$dtac<=2000),sum(ini.cl[ini.cl$eff.cl!="spo"]$dtac<=2000),sum(ini.cl[ini.cl$eff.cl=="2-4"]$dtac<=2000),sum(ini.cl[ini.cl$eff.cl=="5-12"]$dtac<=2000),sum(ini.cl[ini.cl$eff.cl=="13+"]$dtac<=2000))

cl.p2 <- c(sum(ini.cl[ini.cl$eff.cl=="spo"]$dtac>2000),sum(ini.cl[ini.cl$eff.cl!="spo"]$dtac>2000),sum(ini.cl[ini.cl$eff.cl=="2-4"]$dtac>2000),sum(ini.cl[ini.cl$eff.cl=="5-12"]$dtac>2000),sum(ini.cl[ini.cl$eff.cl=="13+"]$dtac>2000))

resall <- rbind(all.n,all.m2,all.p2,all.clm2,all.clp2,cl.n,cl.m2,cl.p2)
colnames(resall) <- c("d=1","d>1","d=2-4","d=5-12","d=13+")
write.table(t(resall),file=paste0(pathR,"tableS5.txt"),sep='\t',col.names=NA,row.names=T)



### Generating binned init-ter data for fig7
bin5k <- tileGenome(seqinf,tilewidth=5000, cut.last.tile.in.chrom=T)
olI <- findOverlaps(ini.all,bin5k)
bin5ki <- binnedAverage(bin5k,coverage(ini.all),"ni",na.rm=T)
bin5kt <- binnedAverage(bin5k,coverage(ter.all),"nt",na.rm=T)
bin5k$ni <- bin5ki$ni
bin5k$nt <- bin5kt$nt
bin5k$rem <- bin5k$ni-bin5k$nt
#cv5kni <- coverage(bin5k,weight=bin5k$ni)
#cv5knt <- coverage(bin5k,weight=bin5k$nt)
cv5krem <- coverage(bin5k,weight=bin5k$rem)
#export(cv5kni,con="bin5k_ni.bw")
#export(cv5knt,con="bin5k_nt.bw")
export(cv5krem,con=paste0(pathR,"bin5k_rem.bw"))
rm=10000
#export(endoapply(cv5kni,caTools::runmean,rm),con="bin5k_ni_rm.bw")
#export(endoapply(cv5knt,caTools::runmean,rm),con="bin5k_nt_rm.bw")
export(endoapply(cv5krem,caTools::runmean,rm),con=paste0(pathR,"bin5k_rem_rm.bw"))

## compute correlation
cv5kremrm <- endoapply(cv5krem,caTools::runmean,rm)
cor.rfd(cv5kremrm, oem_FS)
#0.7964614
# try with NA at the 0
oemNA <- oem_FS
oemNA[oemNA==0] <- NA
cv5kremrmNA <- cv5kremrm
cv5kremrmNA[cv5kremrmNA==0] <- NA
cor.rfd(cv5kremrmNA, oemNA)
#0.8012541


### Comparison with DNAscent
oricon <- import(paste0(pathD,"GSM3450332_1x_BrdU_final.calledOrigins.bed.gz"))
seqlevels(oricon, pruning.mode="coarse") <- seqlevels(seqinf)
seqinfo(oricon) <- seqinf
oriconc <- resize(oricon,fix="center",width=1)
oriconc$dtac <- as.data.frame(distanceToNearest(oriconc,ARSc))[,3]

pdf(paste0(pathR,"supfig_S10b.pdf"),width=10)
Ecdf(oriconc$dtac,ylab='ECDF',col="black",xlim=c(0,20000),q=c(0.5),xlab="Distance to closest Ori center",lwd=2)
Ecdf(ini.all$dtac,add=T,col="red",lwd=2,q=c(0.5))
legend("bottomright",legend=c(paste0("D-NAscent (n=6070,med=",round(median(oriconc$dtac),0),")"),paste0("FORK-seq (n=4964,med=",median(ini.all$dtac),")")), text.col=c("black","red"),bty='n')
dev.off()


## figure timing DNAscent vs FORSKseq
oriconc$timing <- sapply(as(oriconc,"GRangesList"), function(x) RTsc3[overlapsAny(RTsc3,x)]$timing)

tb1 <- enframe(ini.all$timing, name=NULL,value="timing")
tb1$type <- "FORKseq ini"
tb2 <- enframe(oriconc$timing, name=NULL,value="timing")
tb2$type <- "DNAscent ini"
tbRT3 <- bind_rows(tb1,tb2)


#ggplot(tbRT3)+geom_histogram(aes(x=timing,y=..density..,fill=type),position="identity",binwidth=0.1,alpha=0.5,color="black")
ggplot(tbRT3)+geom_density(aes(x=timing,col=type),alpha=0.5,bw=0.07)+ylim(c(0,2.5))+theme(panel.grid.major = element_blank(),panel.grid.minor=element_blank())
ggsave(paste0(pathR,"supfig_S10a.pdf"),width=10)


ter.all$timing <- sapply(as(ter.all,"GRangesList"), function(x) RTsc3[overlapsAny(RTsc3,x)]$timing)

tb3 <- enframe(ter.all$timing, name=NULL,value="timing")
tb3$type <- "FORKseq ter"
tbRT4 <- bind_rows(tb1,tb3)
ggplot(tbRT4)+geom_density(aes(x=timing,col=type),alpha=0.5,bw=0.07)+ylim(c(0,2.5))+theme(panel.grid.major = element_blank(),panel.grid.minor=element_blank())
ggsave(paste0(pathR,"supfig_S7a.pdf"),width=10)


### Intersection for the Euler plot (fig 6b)

terall <- c(length(ter.all[overlapsAny(ter.all,ter.f) & !overlapsAny(ter.all,gr.noemFS)]),length(ter.all[overlapsAny(ter.all,ter.f) & overlapsAny(ter.all,gr.noemFS)]),length(ter.all[!overlapsAny(ter.all,ter.f) & overlapsAny(ter.all,gr.noemFS)]),length(ter.all[!overlapsAny(ter.all,ter.f) & !overlapsAny(ter.all,gr.noemFS)]))
# 9  495 3135  846
length(ter.all)
# 4485
terf <- c(length(ter.f[overlapsAny(ter.f,ter.all) & !overlapsAny(ter.f,gr.noemFS)]),length(ter.f[overlapsAny(ter.f,ter.all) & overlapsAny(ter.f,gr.noemFS)]),length(ter.f[!overlapsAny(ter.f,ter.all) & overlapsAny(ter.f,gr.noemFS)]),length(ter.f[!overlapsAny(ter.f,ter.all) & !overlapsAny(ter.f,gr.noemFS)]))
# 3 63  4  1
length(ter.f)
# 71
noem <- c(length(gr.noemFS[!overlapsAny(gr.noemFS,ter.all) & overlapsAny(gr.noemFS,ter.f)]),length(gr.noemFS[overlapsAny(gr.noemFS,ter.all) & overlapsAny(gr.noemFS,ter.f)]),length(gr.noemFS[overlapsAny(gr.noemFS,ter.all) & !overlapsAny(gr.noemFS,ter.f)]),length(gr.noemFS[!overlapsAny(gr.noemFS,ter.all) & !overlapsAny(gr.noemFS,ter.f)]))
# 3  64 240  34
length(gr.noemFS)
# 341


### RFD on DNAscent data
toread <- dir(paste0(pathD,"DNAscentforks"))
seg.dn <- lapply(toread, function(i) read.table(paste0(pathD,"DNAscentforks/",i),sep="\t",as.is=T))
seg.dn2 <- lapply(seg.dn, function(x) {x[x[,4]=="L",4]="+";x[x[,4]=="R",4]="-";x[,1]=paste0('chr',as.character(as.roman(as.numeric(sapply(as.vector(x[,1]),function(z) strsplit(z,'chr')[[1]][2])))));return(x)})
seg_dn <- lapply(seg.dn2, function(x) GRanges(seqnames=x[,1],ranges=IRanges(start=as.numeric(x[,2]),end=as.numeric(x[,3])),strand=x[,4],inf1=x[,5],inf2=x[,6],str.map=x[,7],seqinfo=seqinf))
segBCD_dn <- do.call(c,seg_dn)
rfd.BCDnt.dn <- simpleRFD(segBCD_dn,lr=0,na2zero=F,expor=T,OKcheck=F,outname=paste0(pathR,"segBCD_dnascent"))
export(endoapply(rfd.BCDnt.dn$RFD2,runmean,k=1000,align="center",endrule="NA"),con=paste0(pathR,"segBCD_dnascent_RFDrm.bw"))
rfd.listnt <- list(rfd.IWnt$RFD2,rfd.OHnt$RFD2,rfd.BCDnt.cnn$RFD2,rfd.BCDnt.tm$RFD2,rfd.BCDnt.dn$RFD2)
rfd.listnt.rm <- lapply(rfd.listnt, function(x) endoapply(x,runmean,k=1000,align="center",endrule="NA"))

mcornt.rm <- sapply(rfd.listnt.rm, function(x) sapply(rfd.listnt.rm, function(y) cor.rfd(x,y)))
colnames(mcornt.rm) <- rownames(mcornt.rm) <- c("IWnt","OHnt","BCDnt.cnn","BCDnt.tm","BCDnt.dn")
ggcorrplot(mcornt.rm,lab=T,lab_size=6)+
	scale_fill_gradient2(limit = c(0,1), low = "blue", high =  "red", mid = "white", midpoint = 0.5)
ggsave(paste0(pathR,"fig2eDNAscent.PDF"),width=5,heigh=5)