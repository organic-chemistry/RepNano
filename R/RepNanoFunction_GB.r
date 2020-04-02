## Functions for the RepNanoPaper

# A function to import OK seq reads
bam2gr <- function(bf)
{
	require(GenomicAlignments)
	require(GenomicRanges)
	paired <- testPairedEndBam(bf)
	close(bf)
	# test if the BAM file is pair-end or not
		if (paired)
		{
			scanpar2 <- ScanBamParam(flag=scanBamFlag(isDuplicate=FALSE,isSecondaryAlignment=FALSE,isProperPair=TRUE),mapqFilter=10)
			# keep only properly paired reads
			ga <- readGAlignmentPairs(bf,param=scanpar2)
			gra <- granges(ga)
			gra <- gra[!duplicated(gra)]
			# remove duplicated fragments
		}
		else
		{
			scanpar <- ScanBamParam(flag=scanBamFlag(isDuplicate=FALSE,isSecondaryAlignment=FALSE),mapqFilter=10)
			ga <- readGAlignments(bf,param=scanpar)
			gra <- granges(ga)
			gra <- gra[!duplicated(gra)]
			# remove duplicated reads
		}
	return(gra)
}
#####
simpleRFD <- function(gr,lr=1,na2zero=T,OKcheck=T,expor=F,outname='myRFDdata')
{
require(GenomicRanges)
require(rtracklayer)

bs <- 1

# check for strand symetry
pl <- length(gr[strand(gr)=='+'])
mi <- length(gr[strand(gr)=='-'])
(corpm <- pl/mi)


cv_L <- coverage(gr[strand(gr)=='+'])
cv_R <- coverage(gr[strand(gr)=='-'])

if (OKcheck)
	{
	if (abs(corpm-1)>0.01)
		{print('data are imbalanced, correction applied'); cv_R <- cv_R*corpm}
	}


cv <- cv_L+cv_R
RFD <- (cv_R-cv_L)/(cv_R+cv_L)
lr_index <- which(cv<=lr)
RFD2 <- RFD
RFD2[lr_index] <- NA

naname <- '_wiNA'
if (na2zero)
	{
	RFD[is.na(RFD)] <- 0
	RFD2[is.na(RFD2)] <- 0
	naname <- '_noNA'
	}

if (expor)
	{
		export(cv,con=paste0(outname,'_cov_tot_bs',bs/1000,'k_lr',lr,'.bw'))
		export(cv_L,con=paste0(outname,'_cov_2left_bs',bs/1000,'k_lr',lr,'.bw'))
		export(cv_R,con=paste0(outname,'_cov_2right_bs',bs/1000,'k_lr',lr,'.bw'))
		export(RFD2,con=paste0(outname,'_RFD_bs',bs/1000,'k_lr',lr,naname,'.bw'))
	}
res <- list(cv,cv_L,cv_R,RFD,RFD2)
names(res) <- c('cv','cv_L','cv_R','RFD','RFD2')
return(res)
}

##

### a function to check correlation between RFD (or other coverage like type of data)
cor.rfd <- function(a,b,met='s')
{cor(as.numeric(unlist(a)[!is.na(unlist(a)) & !is.na(unlist(b))]),as.numeric(unlist(b)[!is.na(unlist(a)) & !is.na(unlist(b))]),method=met)}
##

#### shuffleGR NOT strand aware
# function to resample on a given chromosome. Strand information is not taken into account
shuffleGR4=function(gen=genome,chrnb=24,inputGR=G4data,gap=Ngaps2)
	{	require(GenomicRanges)
		if (class(gen)=="DNAStringSet")
		{seqname=names(gen)
		}else{
		seqname=seqnames(gen)
		}
		hit <- inputGR[seqnames(inputGR)==seqname[chrnb]]
		gapchr=gap[seqnames(gap)==seqname[chrnb]]
# altenative to deal with no gap
		if (length(gapchr)==0) {gapchr=GRanges(seqnames=seqname[chrnb],ranges=IRanges(start=1,width=1),seqinfo=seqinfo(inputGR))}
		ravail <- ranges(gaps(gapchr)[seqnames(gaps(gapchr))==seqname[chrnb] & strand(gaps(gapchr))=="*"])
#		st_avail <- unlist(as.vector(ravail))
# broken in BioC3.7, should come back in BioC3.8
# Temporary fix
		st_avail <- IRanges:::unlist_as_integer(ravail)
		st_rdgr <- sample(st_avail,length(hit))
		if (length(hit)==1)
				{
				wi_rdgr <- width(hit)
				}else{
				wi_rdgr <- sample(width(hit))
				#necessary if only one range sample(width()) choose a number
				#betwen in 1:width() rather than one width
				}
		ra_rdgr <- sort(IRanges(start=st_rdgr,width=wi_rdgr))
		rgap <- ranges(gapchr)
		#sum(overlapsAny(ra_rdgr,ranges(gapchr)))

		keep <- IRanges()
		ra_rdgr2 <- IRanges()
		while ((sum(overlapsAny(ra_rdgr,rgap))!=0) | (sum(overlapsAny(ra_rdgr2,keep))!=0))
			{
			keep <- ra_rdgr[overlapsAny(ra_rdgr,rgap)==0]
			hit2 <- ra_rdgr[overlapsAny(ra_rdgr,rgap)!=0]
			st_rdgr2 <- sample(st_avail,length(hit2))
			if (length(hit2)==1)
				{
				wi_rdgr2 <- width(hit2)
				}else{
				wi_rdgr2 <- sample(width(hit2))
				}
			ra_rdgr2 <- IRanges(start=st_rdgr2,width=wi_rdgr2)
			ra_rdgr <- c(keep,ra_rdgr2)
			}
		rdgr <- sort(GRanges(seqnames=Rle(rep(seqname[chrnb],length(hit))),ranges=ra_rdgr,strand=Rle(rep('*',length(hit))),seqinfo=seqinfo(inputGR)))
		return(rdgr)
	}

# function to resample on a genome

shuffleGRgen <- function(dummy=1,gen2=genome,inputGR2=G4data,gap2=Ngaps2,chrlist=1:chnb)
	{
	rdlist=GRangesList()
	for (i in chrlist) {rdlist[[i]] <- shuffleGR4(gen=gen2,chrnb=i,inputGR=inputGR2,gap=gap2)}
	y<- do.call(c,rdlist)
	return(y)
	}

# Gap annotation
findNgaps <- function(x)
# x is a DNAString
	{ y=Rle(strsplit(as.character(x),NULL)[[1]])
	  y2=ranges(Views(y,y=='N'))
	  return(y2)	# y2 is a list of IRanges
	}

## Convert RleList to matrix (gift from PGP Martin)
RleList2matrix <- function(rlelist)
{
matrix(as.numeric(unlist(rlelist,use.names=F)),
        nrow=length(rlelist),
        byrow=T,
        dimnames=list(names(rlelist),NULL))
}

### clustering function
clust.list <- c(1,5,10,20,50,100,150,200,300,500,1000,1500,2000,2500,3000,4000,5000,7000,10000,20000,50000,100000)
function.cluster <- function(input,clust.list0=clust.list,mc=1)
{
	gr.test <- sort(input)
	mcols(gr.test) <- NULL
	gr.ini.chr <- split(gr.test, seqnames(gr.test))
	resu.cl2 <- mclapply(clust.list0, function(y)
	{
		ch.cl <- list(GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList(),GRangesList())

		for (ch in 1:16)
		{
			xx <- gr.ini.chr[[ch]]
			cl <- 1
			ch.cl[[ch]][[cl]] <- xx[1]
			for (i in (1:(length(xx)-1)))
			{
				if (distance(xx[i],xx[i+1])<y)
				{
					ch.cl[[ch]][[cl]] <- c(ch.cl[[ch]][[cl]],xx[i+1])
				}else{
					cl <- cl+1
					ch.cl[[ch]][[cl]] <- xx[i+1]
				}
			}
		}
		return(ch.cl)
	},mc.cores=mc)

	resu.cl3 <- lapply(resu.cl2, function(x) do.call(c,x))
	resu.cl4 <- endoapply(resu.cl3, function(x1) endoapply(x1, function(x) GRanges(seqnames=seqnames(x)[1],ranges=IRanges(start=min(start(x)),end=max(start(x))),med=median(start(x)),mad=mad(start(x)))))
	resu.cl5 <- lapply(resu.cl3, function(x1) lapply(x1, function(x) start(x)))
	resu.cl6 <- lapply(resu.cl4,unlist)
	resu.cl7 <- lapply(resu.cl5, function(x) sapply(x,length))
	resu.cl8 <- mapply(function(x,y) {x$eff=y;return(x)},resu.cl6,resu.cl7)
	return(resu.cl8)
}
