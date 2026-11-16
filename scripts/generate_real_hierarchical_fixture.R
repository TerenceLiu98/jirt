.libPaths(c('outputs/r-library',.libPaths()))
library(emIRT);library(jsonlite)
load('references/emIRT/data/dwnom.rda')
set.seed(314)
selected <- sort(sample(seq_len(nrow(dwnom$data.in$y)),2000))
d <- dwnom$data.in
subjects <- sort(unique(as.vector(d$i[selected,1])))
items <- sort(unique(as.vector(d$j[selected,1])))
groups <- sort(unique(as.vector(d$g[subjects+1,1])))
y <- matrix(d$y[selected,1])
i <- matrix(match(d$i[selected,1],subjects)-1)
j <- matrix(match(d$j[selected,1],items)-1)
g <- matrix(match(d$g[subjects+1,1],groups)-1)
z <- d$z[subjects+1,,drop=FALSE]
s <- list(alpha=dwnom$cur$alpha[items+1,,drop=FALSE],beta=dwnom$cur$beta[items+1,,drop=FALSE],gamma=dwnom$cur$gamma[groups+1,,drop=FALSE],eta=dwnom$cur$eta[subjects+1,,drop=FALSE],sigma=dwnom$cur$sigma[groups+1,,drop=FALSE])
data <- list(y=y,i=i,j=j,g=g,z=z,ND=ncol(z),NG=length(groups),NI=length(subjects),NJ=length(items),NL=nrow(y))
fit <- hierIRT(data,.starts=s,.priors=dwnom$priors,.control=list(maxit=5,thresh=1e-30,threads=1))
fit$call <- NULL
fit$vars$gamma <- lapply(seq_along(groups),function(k)fit$vars$gamma[,,k])
fit$vars$beta2 <- lapply(seq_along(items),function(k)fit$vars$beta2[,,k])
p <- dwnom$priors
names(p) <- c('beta_mu','beta_covariance','gamma_mu','gamma_covariance','sigma_v','sigma_s')
write_json(list(kwargs=list(votes=y,subject=i,item=j,groups=g,covariates=z,starts=s,priors=p),expected=unclass(fit)), 'tests/fixtures/hierarchical_real.json', digits=NA,matrix='rowmajor',auto_unbox=TRUE)
