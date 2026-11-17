# Run from the repository root after prepare_ggum_reference.py.
.libPaths(c('outputs/r-library',.libPaths()))
library(mirt)
library(jsonlite)
root <- 'outputs/ggum'
inp <- read_json(file.path(root,'inputs.json'),simplifyVector=TRUE)
out <- list(mirt_version=as.character(packageVersion('mirt')),r_version=R.version.string,cases=list())
for(name in names(inp$cases)) {
  case <- inp$cases[[name]]; K <- case$category_counts; J <- length(K)
  y <- case$patterns[rep(seq_len(nrow(case$patterns)),case$counts),,drop=FALSE]
  colnames(y) <- paste0('item',seq_len(J))
  pars <- mirt(y,1,itemtype='ggum',pars='values',verbose=FALSE)
  pars$est[pars$class=='GroupPars'] <- FALSE
  pars$value[pars$name=='MEAN_1'] <- 0
  pars$value[pars$name=='COV_11'] <- 1
  for(j in seq_len(J)) {
    rows <- pars$item==paste0('item',j)
    pars$value[rows & pars$name=='a1'] <- case$starts$a[j]
    pars$lbound[rows & pars$name=='a1'] <- 1e-6
    pars$value[rows & pars$name=='b1'] <- case$starts$delta[j]
    if(j==1) pars$lbound[rows & pars$name=='b1'] <- 1e-6
    for(k in seq_len(K[j]-1)) {
      r <- rows & pars$name==paste0('t',k)
      pars$value[r] <- case$starts$tau[j,k]
      pars$lbound[r] <- 1e-6
    }
  }
  cat('Fitting',name,'\n');flush.console()
  fit <- mirt(y,1,itemtype='ggum',pars=pars,method='EM',optimizer='nlminb',
    control=list(rel.tol=1e-14,x.tol=1e-12,iter.max=1000,eval.max=2000),
    TOL=1e-9,accelerate='none',verbose=FALSE,
    technical=list(customTheta=matrix(inp$nodes,ncol=1),customPriorFun=function(Theta,Etable)inp$weights,NCYCLES=10000))
  cf <- coef(fit,simplify=TRUE)$items
  tau <- matrix(0,J,max(K)-1)
  for(j in seq_len(J)) {
    tau[j,seq_len(K[j]-1)] <- cf[j,paste0('t',seq_len(K[j]-1))]
    # mirt does not impose threshold order. Require an interior ordered solution
    # before treating its fit as matched to jirt's ordered threshold constraints.
    stopifnot(all(tau[j,seq_len(K[j]-1)]>1e-6))
    if(K[j]>2) stopifnot(all(diff(tau[j,seq_len(K[j]-1)]) < -1e-6))
  }
  theta <- seq(-4,4,length.out=41)
  probabilities <- lapply(seq_len(J),function(j)probtrace(extract.item(fit,j),matrix(theta,ncol=1)))
  scoring_fit <- fit; scoring_fit@Options$exploratory <- FALSE
  scores <- fscores(scoring_fit,method='EAP',response.pattern=case$patterns,full.scores.SE=TRUE,
    custom_theta=matrix(inp$nodes,ncol=1),custom_den=function(Theta,...)inp$weights)
  out$cases[[name]] <- list(a=cf[,'a1'],delta=cf[,'b1'],tau=tau,theta=theta,probabilities=probabilities,
    scores=scores[,1],variances=scores[,2]^2,log_likelihood=as.numeric(logLik(fit)),
    converged=extract.mirt(fit,'converged'),iterations=extract.mirt(fit,'iterations'))
  write_json(out,file.path(root,'reference.json'),digits=NA,auto_unbox=TRUE,matrix='rowmajor',pretty=TRUE)
  cat(name,extract.mirt(fit,'converged'),as.numeric(logLik(fit)),'\n');flush.console()
}
