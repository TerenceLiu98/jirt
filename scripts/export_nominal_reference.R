.libPaths(c('outputs/r-library',.libPaths()))
library(mirt);library(jsonlite)
root<-'outputs/nominal';inp<-read_json(file.path(root,'inputs.json'),simplifyVector=TRUE)
out<-list(mirt_version=as.character(packageVersion('mirt')),r_version=R.version.string,cases=list())
for(model in names(inp$cases)) {
 case<-inp$cases[[model]];K<-case$category_counts;J<-length(K);keys<-case$answer_keys
 y<-case$patterns[rep(seq_len(nrow(case$patterns)),case$counts),,drop=FALSE];colnames(y)<-paste0('item',seq_len(J))
 type<-switch(model,nominal='nominal','2pl'='2PLNRM','3pl'='3PLNRM','3plu'='3PLuNRM','4pl'='4PLNRM')
 key<-if(model=='nominal') NULL else keys
 pars<-mirt(y,1,itemtype=type,key=key,pars='values',verbose=FALSE)
 for(j in seq_len(J)) {
  rows<-which(pars$item==paste0('item',j));ak<-rows[grepl('^ak',pars$name[rows])];dk<-rows[grepl('^d[0-9]',pars$name[rows])]
  pars$value[ak]<-.5;pars$value[ak[1]]<-0;pars$est[ak]<-TRUE;pars$est[ak[1]]<-FALSE
  pars$value[dk]<-0;pars$est[dk]<-TRUE;pars$est[dk[1]]<-FALSE
  ar<-intersect(rows,which(pars$name=='a1'));pars$value[ar]<-1
  if(model=='nominal') {
   pars$est[ar]<-FALSE
   if(j==1)pars$lbound[ak[2]]<-1e-6
  } else {
   pars$lbound[ar]<-1e-6
   pars$value[intersect(rows,which(pars$name=='d'))]<-0
   pars$value[intersect(rows,which(pars$name=='g'))]<-if(model %in% c('3pl','4pl')) .1 else 0
   pars$value[intersect(rows,which(pars$name=='u'))]<-if(model %in% c('3plu','4pl')) .9 else 1
  }
 }
 asym<-pars$name %in% c('g','u') & pars$est;pars$lbound[asym]<-1e-6;pars$ubound[asym]<-1-1e-6
 pars$est[pars$class=='GroupPars']<-FALSE;pars$value[pars$name=='MEAN_1']<-0;pars$value[pars$name=='COV_11']<-1
 cat('Fitting',model,'\n');flush.console()
 fit<-mirt(y,1,itemtype=type,key=key,pars=pars,method='EM',optimizer='nlminb',control=list(rel.tol=1e-14,x.tol=1e-12,iter.max=1000,eval.max=2000),TOL=1e-8,accelerate='none',verbose=FALSE,
  technical=list(customTheta=matrix(inp$nodes,ncol=1),customPriorFun=function(Theta,Etable)inp$weights,NCYCLES=10000))
 cf<-coef(fit,simplify=TRUE)$items
 slopes<-intercepts<-matrix(0,J,max(K));a<-b<-c<-u<-numeric(J)
 for(j in seq_len(J)) {
  options<-if(model=='nominal') seq_len(K[j]) else setdiff(seq_len(K[j]),keys[j]+1)
  for(t in seq_along(options)) {
   slopes[j,options[t]]<-cf[j,paste0('ak',t-1)]*if(model=='nominal') cf[j,'a1'] else 1
   intercepts[j,options[t]]<-cf[j,paste0('d',t-1)]
  }
  if(model!='nominal'){a[j]<-cf[j,'a1'];b[j]<- -cf[j,'d']/a[j];c[j]<-cf[j,'g'];u[j]<-cf[j,'u']}
 }
 theta<-seq(-4,4,length.out=41)
 probs<-lapply(seq_len(J),function(j)probtrace(extract.item(fit,j),matrix(theta,ncol=1)))
 scores<-fscores(fit,method='EAP',response.pattern=case$patterns,full.scores.SE=TRUE,custom_theta=matrix(inp$nodes,ncol=1),custom_den=function(Theta,...)inp$weights)
 means<-list(slopes=slopes,intercepts=intercepts)
 if(model!='nominal')means<-c(means,list(a=a,b=b,c=c,u=u))
 out$cases[[model]]<-list(means=means,theta=theta,probabilities=probs,scores=scores[,1],variances=scores[,2]^2,log_likelihood=as.numeric(logLik(fit)),converged=extract.mirt(fit,'converged'),iterations=extract.mirt(fit,'iterations'))
 write_json(out,file.path(root,'reference.json'),digits=NA,auto_unbox=TRUE,matrix='rowmajor',pretty=TRUE)
 cat(model,extract.mirt(fit,'converged'),as.numeric(logLik(fit)),'\n');flush.console()
}
