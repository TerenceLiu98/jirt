.libPaths(c('outputs/r-library', .libPaths()))
library(emIRT)
library(jsonlite)
load('references/emIRT/data/manifesto.rda')
x <- manifesto
out <- poisIRT(x$data.manif, i=0:(ncol(x$data.manif)-1), NI=ncol(x$data.manif), .starts=x$starts.manif,.priors=x$priors.manif,.control=list(maxit=1500,threads=1))
out$call <- NULL
out <- unclass(out)
dir.create('outputs/baselines',recursive=TRUE,showWarnings=FALSE)
write_json(out,'outputs/baselines/poisson.json',digits=NA,matrix='rowmajor',auto_unbox=TRUE)
