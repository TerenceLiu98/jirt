# Run after prepare_response_reference.py; reference sources remain read-only.
.libPaths(c('outputs/r-library',.libPaths()))
library(mirt)
library(jsonlite)
root <- 'outputs/response-models'
inputs <- read_json(file.path(root,'inputs.json'),simplifyVector=TRUE)
out <- list(mirt_version=as.character(packageVersion('mirt')),r_version=R.version.string,cases=list())
for(name in names(inputs$cases)) {
  case <- inputs$cases[[name]]; pc <- startsWith(name,'pc')
  J <- ncol(case$patterns); D <- ncol(case$nodes)
  y <- case$patterns[rep(seq_len(nrow(case$patterns)),case$counts),,drop=FALSE]
  colnames(y) <- paste0('item',seq_len(J))
  model <- if(pc) paste(vapply(seq_len(D),function(k) paste0('F',k,' = ',paste(which(case$loading_mask[,k]),collapse=',')),character(1)),collapse='\n') else 1
  type <- if(pc) toupper(name) else if(name=='tutz') 'Tutz' else 'sequential'
  pars <- mirt(y,model,itemtype=type,pars='values',verbose=FALSE)
  pars$est[pars$class=='GroupPars'] <- FALSE
  pars$value[pars$class=='GroupPars'] <- 0
  for(k in seq_len(D)) pars$value[pars$name==paste0('COV_',k,k)] <- 1
  for(j in seq_len(J)) {
    rows <- pars$item==paste0('item',j)
    for(k in seq_len(D)) {
      r <- rows & pars$name==paste0('a',k)
      active <- !pc || case$loading_mask[j,k]
      pars$value[r] <- as.numeric(active)
      pars$est[r] <- active && !(name %in% c('pc1pl','tutz'))
      if(active) pars$lbound[r] <- 1e-6
    }
    if(pc) {
      for(k in seq_len(D)) {
        r <- rows & pars$name==paste0('d',k)
        pars$value[r] <- 0; pars$est[r] <- case$loading_mask[j,k]
      }
      r <- rows & pars$name=='g'
      pars$value[r] <- if(name=='pc3pl') .1 else 0
      if(name=='pc3pl') {pars$lbound[r]<-1e-6;pars$ubound[r]<-1-1e-6}
    } else {
      K <- case$category_counts[j]
      b <- if(K>2) seq(-1,1,length.out=K-1) else 0
      for(k in seq_len(K-1)) pars$value[rows & pars$name==paste0('d',k)] <- -b[k]
    }
  }
  weights <- case$weights
  cat('Fitting',name,'\n');flush.console()
  fit <- mirt(y,model,itemtype=type,pars=pars,method='EM',optimizer='nlminb',
    control=list(rel.tol=1e-14,x.tol=1e-12,iter.max=1000,eval.max=2000),
    TOL=1e-8,accelerate='none',verbose=FALSE,
    technical=list(customTheta=case$nodes,customPriorFun=function(Theta,Etable)weights,NCYCLES=10000))
  cf <- coef(fit,simplify=TRUE)$items
  theta <- if(pc) as.matrix(expand.grid(c(-3,0,3),c(-3,0,3))) else matrix(seq(-4,4,length.out=41),ncol=1)
  probabilities <- lapply(seq_len(J),function(j)probtrace(extract.item(fit,j),theta))
  scoring_fit <- fit; scoring_fit@Options$exploratory <- FALSE
  scores <- fscores(scoring_fit,method='EAP',response.pattern=case$patterns,full.scores.SE=TRUE,
    custom_theta=case$nodes,custom_den=function(Theta,...)weights)
  result <- list(theta=theta,probabilities=probabilities,
    scores=scores[,seq_len(D),drop=FALSE],variances=scores[,D+seq_len(D),drop=FALSE]^2,
    log_likelihood=as.numeric(logLik(fit)),converged=extract.mirt(fit,'converged'),iterations=extract.mirt(fit,'iterations'))
  if(pc) {
    result$a <- cf[,paste0('a',seq_len(D)),drop=FALSE]
    result$d <- cf[,paste0('d',seq_len(D)),drop=FALSE]
    result$c <- cf[,'g']
  } else {
    result$a <- cf[,'a1']
    steps <- matrix(0,J,max(case$category_counts)-1)
    for(j in seq_len(J)) for(k in seq_len(case$category_counts[j]-1)) steps[j,k] <- -cf[j,paste0('d',k)]/result$a[j]
    result$steps <- steps
  }
  out$cases[[name]] <- result
  write_json(out,file.path(root,'reference.json'),digits=NA,auto_unbox=TRUE,matrix='rowmajor',pretty=TRUE)
  cat(name,result$converged,result$log_likelihood,'\n');flush.console()
}
