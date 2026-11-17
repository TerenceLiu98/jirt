.libPaths(c('outputs/r-library',.libPaths()))
library(mirt)
library(jsonlite)
root<-'outputs/polytomous'
inp<-read_json(file.path(root,'inputs.json'),simplifyVector=TRUE)
out<-list(mirt_version=as.character(packageVersion('mirt')),r_version=R.version.string,cases=list())
for(model in names(inp$cases)) {
 case<-inp$cases[[model]];K<-case$category_counts;J<-length(K)
 y<-case$patterns[rep(seq_len(nrow(case$patterns)),case$counts),,drop=FALSE]
 colnames(y)<-paste0('item',seq_len(J))
 type<-if(model=='grm') 'graded' else if(model=='rsm') 'rsm' else 'gpcmIRT'
 pars<-mirt(y,1,itemtype=type,pars='values',verbose=FALSE)
 pars$value[pars$name=='a1']<-1
 pars$lbound[pars$name=='a1']<-1e-6
 if(model %in% c('pcm','rsm')) pars$est[pars$name=='a1']<-FALSE
 pars$est[pars$class=='GroupPars']<-FALSE
 pars$value[pars$name=='MEAN_1']<-0;pars$value[pars$name=='COV_11']<-1
 for(j in seq_len(J)) {
   b<-if(K[j]>2) seq(-1,1,length.out=K[j]-1) else 0
   rows<-which(pars$item==paste0('item',j))
   for(k in seq_len(K[j]-1)) pars$value[intersect(rows,which(pars$name==paste0(if(model=='grm') 'd' else 'b',k)))]<-if(model=='grm') -b[k] else b[k]
 }
 if(model=='rsm') pars$value[pars$name=='c']<-0
 cat('Fitting',model,'\n');flush.console()
 fit<-mirt(y,1,itemtype=type,pars=pars,method='EM',optimizer='nlminb',
   control=list(rel.tol=1e-14,x.tol=1e-12,iter.max=1000,eval.max=2000),
   TOL=1e-9,accelerate='none',verbose=FALSE,
   technical=list(customTheta=matrix(inp$nodes,ncol=1),customPriorFun=function(Theta,Etable) inp$weights,NCYCLES=10000))
 cf<-coef(fit,simplify=TRUE)$items
 steps<-matrix(0,J,max(K)-1)
 for(j in seq_len(J)) for(k in seq_len(K[j]-1)) {
   steps[j,k]<-if(model=='grm') -cf[j,paste0('d',k)]/cf[j,'a1'] else cf[j,paste0('b',k)]-cf[j,'c']/cf[j,'a1']
 }
 theta<-seq(-4,4,length.out=41)
 probabilities<-lapply(seq_len(J),function(j)probtrace(extract.item(fit,j),matrix(theta,ncol=1)))
 scores<-fscores(fit,method='EAP',response.pattern=case$patterns,full.scores.SE=TRUE,custom_theta=matrix(inp$nodes,ncol=1),custom_den=function(Theta,...)inp$weights)
 out$cases[[model]]<-list(a=cf[,'a1'],steps=steps,theta=theta,probabilities=probabilities,scores=scores[,1],variances=scores[,2]^2,log_likelihood=as.numeric(logLik(fit)),converged=extract.mirt(fit,'converged'),iterations=extract.mirt(fit,'iterations'))
 write_json(out,file.path(root,'reference.json'),digits=NA,auto_unbox=TRUE,matrix='rowmajor',pretty=TRUE)
 cat(model,extract.mirt(fit,'converged'),as.numeric(logLik(fit)),'\n')
}
