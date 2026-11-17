# Match jirt's fixed N(0,1) marginal calibration, quadrature and starts.
.libPaths(c('outputs/r-library', .libPaths()))
library(mirt)
library(jsonlite)
root <- 'outputs/logistic-mirt'
inp <- read_json(file.path(root, 'inputs.json'), simplifyVector=TRUE)
Theta <- matrix(inp$nodes, ncol=1)
weights <- inp$weights
prior <- function(Theta, Etable) weights
cases <- inp$cases
# Real LSAT7 frequency table, distributed with mirt.
data(LSAT7, package='mirt')
for (model in c('rasch','1pl','2pl')) {
  cases[[paste0('lsat7_',model)]] <- list(model=model,
    patterns=as.matrix(LSAT7[,1:5]), counts=LSAT7[,6])
}
results <- list()
for (name in names(cases)) {
  case <- cases[[name]]
  model <- case$model
  y <- case$patterns[rep(seq_len(nrow(case$patterns)), case$counts),,drop=FALSE]
  colnames(y) <- paste0('item',seq_len(ncol(y)))
  type <- if(model %in% c('rasch','1pl')) '2PL' else toupper(model)
  pars <- mirt(y, 1, itemtype=type, pars='values', verbose=FALSE)
  pars$value[pars$name=='a1'] <- 1
  pars$value[pars$name=='d'] <- 0
  pars$value[pars$name=='g'] <- if(model %in% c('3pl','4pl')) .1 else 0
  pars$value[pars$name=='u'] <- if(model=='4pl') .9 else 1
  pars$value[pars$name=='MEAN_1'] <- 0
  pars$value[pars$name=='COV_11'] <- 1
  pars$est[pars$name %in% c('MEAN_1','COV_11')] <- FALSE
  if(model=='rasch') pars$est[pars$name=='a1'] <- FALSE
  pars$lbound[pars$name=='a1'] <- 1e-6
  pars$ubound[pars$name=='a1'] <- Inf
  # Same marginal asymptote limits. mirt has no direct u-g >= eps constraint;
  # verify that the fitted gap is strictly interior below.
  pars$lbound[pars$name %in% c('g','u') & pars$est] <- 1e-6
  pars$ubound[pars$name %in% c('g','u') & pars$est] <- 1-1e-6
  equal <- if(model=='1pl') list(pars$parnum[pars$name=='a1']) else list()
  cat('Fitting', name, '\n'); flush.console()
  fit <- mirt(y, 1, itemtype=type, pars=pars, constrain=equal,
    method='EM', optimizer='nlminb',
    control=list(rel.tol=1e-15, x.tol=1e-14, abs.tol=1e-12, iter.max=1000, eval.max=2000), TOL=1e-9, accelerate='none', verbose=FALSE,
    technical=list(customTheta=Theta, customPriorFun=prior, NCYCLES=20000))
  cf <- coef(fit, IRTpars=TRUE, simplify=TRUE)$items
  stopifnot(all(cf[,'u']-cf[,'g'] > 1e-6))
  scores <- fscores(fit, method='EAP', response.pattern=case$patterns,
    full.scores.SE=TRUE, custom_theta=Theta, custom_den=function(Theta,...) weights)
  test_theta <- matrix(seq(-4,4,length.out=81),ncol=1)
  probabilities <- sapply(seq_len(ncol(y)), function(j) probtrace(extract.item(fit,j),test_theta)[,2])
  results[[name]] <- list(model=model,patterns=case$patterns,counts=case$counts,
    means=list(a=cf[,'a'],b=cf[,'b'],c=cf[,'g'],u=cf[,'u']),
    scores=scores[,1], score_variances=scores[,2]^2,
    theta=as.vector(test_theta), probabilities=probabilities,
    log_likelihood=as.numeric(logLik(fit)), converged=extract.mirt(fit,'converged'),
    iterations=extract.mirt(fit,'iterations'))
  write_json(list(mirt_version=as.character(packageVersion('mirt')),
    r_version=R.version.string,
    source_commit=system2('git',c('-C','outputs/reference-build/mirt','rev-parse','HEAD'),stdout=TRUE),
    settings=list(method='EM',optimizer='nlminb',accelerate='none',TOL=1e-9,
      latent_distribution='N(0,1)',item_priors='none',quadrature_points=length(weights)),
    nodes=inp$nodes, weights=weights,cases=results),
    file.path(root,'reference.json'),digits=NA,auto_unbox=TRUE,matrix='rowmajor',pretty=TRUE)
  cat(name, 'converged:', extract.mirt(fit,'converged'), 'logLik:', as.numeric(logLik(fit)), '\n')
}
