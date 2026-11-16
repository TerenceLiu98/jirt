.libPaths(c('outputs/r-library',.libPaths()))
library(emIRT);library(jsonlite)
load('references/emIRT/data/AsahiTodai.rda')
x <- AsahiTodai
out <- ordIRT(x$dat.all,.starts=x$start.values,.priors=x$priors,.control=list(maxit=300,threads=1))
out$call <- NULL
write_json(unclass(out),'outputs/baselines/ordinal.json',digits=NA,matrix='rowmajor',auto_unbox=TRUE)
