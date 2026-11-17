# Run from repository root after prepare_multidimensional_reference.py.
.libPaths(c('outputs/r-library',.libPaths()))
library(mirt)
library(jsonlite)
root <- 'outputs/multidimensional'
inputs <- read_json(file.path(root,'inputs.json'),simplifyVector=TRUE)
out <- list(mirt_version=as.character(packageVersion('mirt')),r_version=R.version.string,cases=list())
for(name in names(inputs)) {
  case <- inputs[[name]]
  mask <- case$mask; J<-nrow(mask); D<-ncol(mask)
  y <- case$patterns[rep(seq_len(nrow(case$patterns)),case$counts),,drop=FALSE]
  colnames(y)<-paste0('item',seq_len(J))
  model <- paste(vapply(seq_len(D),function(k)paste0('F',k,' = ',paste(which(mask[,k]),collapse=',')),character(1)),collapse='\n')
  pars <- mirt(y,model,itemtype='2PL',pars='values',verbose=FALSE)
  for(k in seq_len(D)) {
    rows <- which(pars$name==paste0('a',k))
    pars$value[rows] <- if(k==1) 1 else .5
    pars$value[rows[!mask[,k]]] <- 0
    pars$est[rows] <- mask[,k]
    anchor <- if(name=='exploratory') k else which(mask[,k])[1]
    pars$lbound[rows[anchor]] <- if(name=='testlet' && k>1) 0 else 1e-6
  }
  pars$value[pars$name=='d']<-0
  group_rows<-pars$class=='GroupPars'
  pars$est[group_rows]<-FALSE
  pars$value[group_rows]<-0
  for(k in seq_len(D)) pars$value[pars$name==paste0('COV_',k,k)]<-1
  equal<-list()
  if(name=='testlet') for(k in 2:D) equal[[k-1]]<-pars$parnum[pars$name==paste0('a',k)&pars$est]
  weights<-case$weights
  cat('Fitting',name,'\n');flush.console()
  fit<-mirt(y,model,itemtype='2PL',pars=pars,constrain=equal,
    method='EM',optimizer='NR',control=list(tol=1e-11,maxit=100),
    TOL=1e-8,accelerate='none',verbose=FALSE,
    technical=list(customTheta=case$nodes,customPriorFun=function(Theta,Etable)weights,NCYCLES=10000))
  cf<-coef(fit,simplify=TRUE)$items
  probabilities<-sapply(seq_len(J),function(j)probtrace(extract.item(fit,j),case$theta)[,2])
  # A triangular mask is auto-classified as exploratory by mirt. Score in the
  # supplied anchored coordinates instead of fscores' default oblimin rotation.
  scoring_fit <- fit
  scoring_fit@Options$exploratory <- FALSE
  scores<-fscores(scoring_fit,method='EAP',response.pattern=case$patterns,full.scores.SE=TRUE,
    custom_theta=case$nodes,custom_den=function(Theta,...)weights)
  result<-list(a=cf[,seq_len(D),drop=FALSE],d=cf[,'d'],probabilities=probabilities,
    scores=scores[,seq_len(D),drop=FALSE],variances=scores[,D+seq_len(D),drop=FALSE]^2,
    log_likelihood=as.numeric(logLik(fit)),converged=extract.mirt(fit,'converged'),iterations=extract.mirt(fit,'iterations'))
  if(name %in% c('bifactor','testlet')) {
    # Check the bfactor engine's conditional predictions at the fitted JAX parameters.
    specific<-c(1,1,1,2,2,2,NA,NA,NA)
    bp<-bfactor(y,specific,itemtype='2PL',pars='values',verbose=FALSE)
    for(k in seq_len(D)) bp$value[bp$name==paste0('a',k)]<-case$jirt$a[,k]
    bp$value[bp$name=='d']<-case$jirt$d
    bp$est[]<-FALSE
    bf<-bfactor(y,specific,itemtype='2PL',pars=bp,quadpts=15,TOL=NaN,verbose=FALSE)
    result$bfactor_at_jirt<-sapply(seq_len(J),function(j)probtrace(extract.item(bf,j),case$theta)[,2])
  }
  out$cases[[name]]<-result
  write_json(out,file.path(root,'reference.json'),digits=NA,auto_unbox=TRUE,matrix='rowmajor',pretty=TRUE)
  cat(name,result$converged,result$log_likelihood,'\n');flush.console()
}
