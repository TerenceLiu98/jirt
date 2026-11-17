# Independent mirt calibrations. Generated files stay in outputs/groups/.
.libPaths(c('outputs/r-library',.libPaths()))
library(mirt)
library(jsonlite)
root <- 'outputs/groups'
inp <- read_json(file.path(root,'inputs.json'),simplifyVector=TRUE)
raw_inputs <- read_json(file.path(root,'inputs.json'))
out <- list(mirt_version=as.character(packageVersion('mirt')),r_version=R.version.string,cases=list())
for(name in names(inp$cases)) {
  case <- inp$cases[[name]]
  case$starts <- lapply(raw_inputs$cases[[name]]$starts,function(s)fromJSON(toJSON(s,auto_unbox=TRUE,digits=NA),simplifyVector=TRUE))
  kind <- case$kind; K <- case$category_counts; J <- length(K)
  C <- case$classes; total <- C+as.integer(kind=='zero')
  if(kind=='group') {
    data <- lapply(seq_len(C),function(c)case$patterns[rep(seq_len(nrow(case$patterns)),case$counts[c,]),,drop=FALSE])
    y <- do.call(rbind,data)
    groups <- factor(rep(paste0('G',seq_len(C)),sapply(data,nrow)))
  } else {y <- case$patterns[rep(seq_len(nrow(case$patterns)),case$counts),,drop=FALSE];groups<-NULL}
  colnames(y)<-paste0('item',seq_len(J))
  type <- if(case$model=='grm') 'graded' else '2PL'
  inv <- if(kind=='group') c('item1','item2','free_means','free_var') else character(0)
  dentype <- if(kind=='group') 'Gaussian' else paste0('mixture-',total)
  grid <- if(kind=='mixture') matrix(inp$nodes,ncol=1) else matrix(c(if(kind=='zero') -100,seq(-10,10,by=.1)),ncol=1)
  technical <- list(customTheta=grid,NCYCLES=10000)
  if(kind=='mixture')technical$customPriorFun<-function(Theta,Etable)inp$weights
  fits<-list();likelihoods<-numeric()
  for(attempt in seq_along(case$starts)) {
    start<-case$starts[[attempt]]
    pars<-multipleGroup(y,1,group=groups,itemtype=type,dentype=dentype,invariance=inv,pars='values',verbose=FALSE)
    names_group<-unique(pars$group)
    for(c in seq_len(total)) {
      cg<-pars$group==names_group[c]
      pars$est[cg & pars$name %in% c('MEAN_1','COV_11')]<-FALSE
      pars$value[cg & pars$name=='MEAN_1']<-if(c>C) -100 else start$mu[c]
      pars$value[cg & pars$name=='COV_11']<-if(c>C) 1e-5 else start$sigma[c]^2
      if(kind=='group' && c>1)pars$est[cg & pars$name %in% c('MEAN_1','COV_11')]<-TRUE
      for(j in seq_len(J)) {
        rows<-cg & pars$item==paste0('item',j)
        pars$value[rows & pars$name=='a1']<-if(c>C) 1 else start$a[c,j]
        pars$lbound[rows & pars$name=='a1']<-1e-6
        for(k in seq_len(K[j]-1)) {
          r<-rows & pars$name==if(case$model=='grm') paste0('d',k) else 'd'
          pars$value[r]<-if(c>C) -k else -start$a[c,j]*start$steps[c,j,k]
        }
        if(c>C)pars$est[rows]<-FALSE
      }
      if(kind!='group') {
        pars$value[cg & pars$name=='PI']<-log(start$weights[c]/start$weights[1])
        pars$est[cg & pars$name=='PI']<-c>1
      }
    }
    cat('Fitting',name,'start',attempt,'\n');flush.console()
    fit<-multipleGroup(y,1,group=groups,itemtype=type,dentype=dentype,invariance=inv,pars=pars,
      method='EM',optimizer='nlminb',control=list(rel.tol=1e-14,x.tol=1e-12,iter.max=1000,eval.max=2000),
      TOL=1e-9,accelerate='none',verbose=FALSE,technical=technical)
    stopifnot(extract.mirt(fit,'converged'))
    fits[[attempt]]<-fit;likelihoods[attempt]<-as.numeric(logLik(fit))
    cat(name,attempt,likelihoods[attempt],'\n');flush.console()
  }
  fit<-fits[[which.max(likelihoods)]]
  pv<-mod2values(fit);names_group<-unique(pv$group)
  aa<-matrix(0,C,J);bb<-array(0,c(C,J,max(K)-1));mu<-sigma<-numeric(C)
  theta<-seq(-4,4,length.out=41);curves<-scores<-variances<-list();mass<-matrix(0,nrow(case$patterns),total)
  for(c in seq_len(C)) {
    sub<-extract.group(fit,c);sub@Options$exploratory<-FALSE
    cf<-coef(sub,simplify=TRUE)
    aa[c,]<-cf$items[,'a1'];mu[c]<-cf$means[1];sigma[c]<-sqrt(cf$cov[1,1])
    for(j in seq_len(J))for(k in seq_len(K[j]-1))bb[c,j,k]<- -cf$items[j,if(case$model=='grm')paste0('d',k) else 'd']/aa[c,j]
    curves[[c]]<-lapply(seq_len(J),function(j)probtrace(extract.item(sub,j),matrix(theta,ncol=1)))
    ability<-mu[c]+sigma[c]*inp$nodes
    score<-fscores(sub,method='EAP',response.pattern=case$patterns,full.scores.SE=TRUE,custom_theta=matrix(ability,ncol=1),custom_den=function(Theta,...)inp$weights)
    scores[[c]]<-score[,1];variances[[c]]<-score[,2]^2
    pattern_prob<-matrix(1,nrow(case$patterns),length(ability))
    for(j in seq_len(J)) {
      p<-probtrace(extract.item(sub,j),matrix(ability,ncol=1))
      pattern_prob<-pattern_prob*t(p[,case$patterns[,j]+1,drop=FALSE])
    }
    mass[,c]<-pattern_prob %*% inp$weights
  }
  weight<-membership<-NULL
  if(kind!='group') {
    logits<-pv$value[pv$name=='PI'];weight<-exp(logits-max(logits));weight<-weight/sum(weight)
    if(kind=='zero')mass[,total]<-as.numeric(rowSums(case$patterns)==0)
    membership<-sweep(mass,2,weight,'*');membership<-membership/rowSums(membership)
  }
  # Convert arrays to nested lists explicitly to retain class/item/step axes.
  result<-list(a=aa,steps=lapply(seq_len(C),function(c)matrix(bb[c,,],nrow=J)),mu=mu,sigma=sigma,weights=weight,
    theta=theta,probabilities=curves,scores=scores,variances=variances,memberships=membership,
    log_likelihood=max(likelihoods),start_log_likelihoods=likelihoods,converged=TRUE)
  out$cases[[name]]<-result
  write_json(out,file.path(root,'reference.json'),digits=NA,auto_unbox=TRUE,matrix='rowmajor',pretty=TRUE)
}
