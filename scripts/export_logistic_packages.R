# Independent package fits; never alter any reference package source.
.libPaths(c('outputs/r-library', .libPaths()))
library(jsonlite)
root <- 'outputs/logistic-mirt'
inp <- read_json(file.path(root,'inputs.json'), simplifyVector=TRUE)
jax <- read_json(file.path(root,'jax-multistart.json'), simplifyVector=TRUE)
mirt <- read_json(file.path(root,'reference.json'), simplifyVector=TRUE)
outfile <- Sys.getenv('IRT_COMPARISON_OUTPUT',unset=file.path(root,'other-packages.json'))
out <- if(file.exists(outfile)) read_json(outfile,simplifyVector=TRUE) else
  list(r_version=R.version.string, versions=list(), fits=list('3pl'=list(),'4pl'=list()))
engine <- commandArgs(trailingOnly=TRUE)[1]
stopifnot(engine %in% c('ltm','sirt','sirt_fine'))
out$versions[[engine]] <- as.character(packageVersion(if(engine=='sirt_fine') 'sirt' else engine))
for(model in if(engine=='ltm') '3pl' else c('3pl','4pl')) {
  case <- inp$cases[[model]]
  y <- case$patterns[rep(seq_len(nrow(case$patterns)),case$counts),,drop=FALSE]
  colnames(y) <- paste0('item',seq_len(ncol(y)))
  J <- ncol(y)
  best <- which.max(vapply(jax[[model]],function(x)x$log_likelihood,numeric(1)))
  starts <- list(
    neutral=list(a=rep(1,J),b=rep(0,J),c=rep(.1,J),u=rep(if(model=='4pl') .9 else 1,J)),
    generating=list(a=seq(.8,1.6,length.out=J),b=seq(-1.5,1.5,length.out=J),c=rep(.1,J),u=rep(if(model=='4pl') .9 else 1,J)),
    jax_best=jax[[model]][[best]]$means,
    mirt=mirt$cases[[model]]$means)
  if(engine=='sirt_fine') starts <- starts['jax_best']
  for(name in names(starts)) {
    s <- starts[[name]]
    cat(engine,model,name,'\n');flush.console()
    warnings <- character()
    record <- withCallingHandlers({
      if(engine=='ltm') {
        fit <- ltm::tpm(y,start.val=cbind(s$c,-s$a*s$b,s$a),
          control=list(GHk=41,iter.qN=10000,method='BFGS'))
        cf <- coef(fit)
        stopifnot(max(abs(sort(fit$GH$Z[,2])-inp$nodes))<1e-10,
                  max(abs(fit$GH$GHw[order(fit$GH$Z[,2])]-inp$weights))<1e-12)
        list(means=list(a=cf[,3],b=cf[,2],c=cf[,1],u=rep(1,J)),
             log_likelihood=fit$log.Lik,status=fit$convergence,package_score_max=fit$max.sc)
      } else {
        fit <- sirt::rasch.mml2(y,theta.k=seq(-8,8,length.out=101),variance.fixed=matrix(c(1,1,1),nrow=1),
          center.trait=TRUE,center.b=FALSE,alpha1=0,alpha2=0,est.alpha=FALSE,
          fixed.a=s$a,b.init=s$b,fixed.c=s$c,fixed.d=s$u,
          est.a=seq_len(J),est.c=seq_len(J),
          est.d=if(model=='4pl') seq_len(J) else NULL,
          min.a=1e-6,max.a=Inf,min.b=-Inf,max.b=Inf,
          min.c=1e-6,max.c=1-1e-6,min.d=1e-6,max.d=1-1e-6,
          numdiff.parm=if(engine=='sirt_fine') 1e-6 else 1e-5,
          glob.conv=1e-10,parm.conv=1e-8,mmliter=20000,mitermax=100,progress=FALSE)
        saveRDS(fit,file.path(root,paste0(engine,'-',model,'-',name,'.rds')))
        cf <- fit$item
        stopifnot(abs(fit$mean.trait)<1e-12,abs(fit$sd.trait-1)<1e-12,
                  max(abs(as.vector(fit$pi.k)-dnorm(fit$theta.k)/sum(dnorm(fit$theta.k))))<1e-12)
        list(nodes=as.vector(fit$theta.k),weights=as.vector(fit$pi.k),
             iteration_limit_reached=fit$iter>=20000,means=list(a=cf$a,b=cf$b,c=cf$c,u=cf$d),
             log_likelihood=as.numeric(logLik(fit)),status=fit$iter)
      }
    }, warning=function(w){warnings <<- c(warnings,conditionMessage(w)); invokeRestart('muffleWarning')})
    record$warnings <- warnings
    record$starts <- s
    record$settings <- if(engine=='ltm') list(method='BFGS',GHk=41,maxit=10000) else
      list(method='EM',numdiff.parm=if(engine=='sirt_fine') 1e-6 else 1e-5,
           glob.conv=1e-10,parm.conv=1e-8,mmliter=20000,mean=0,variance=1)
    out$fits[[model]][[paste(engine,name,sep='_')]] <- record
    write_json(out,outfile,digits=NA,auto_unbox=TRUE,pretty=TRUE)
    cat('logLik',record$log_likelihood,'status',record$status,'\n');flush.console()
  }
}
