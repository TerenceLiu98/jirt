.libPaths(c('outputs/r-library',.libPaths()))
library(emIRT);library(jsonlite)
load('references/emIRT/data/ustweet.rda')
out <- networkIRT(ustweet$data,.starts=ustweet$starts,.priors=ustweet$priors,.control=list(maxit=500,convtype=2,threads=1))
out$call <- NULL
write_json(unclass(out),'outputs/baselines/network.json',digits=NA,matrix='rowmajor',auto_unbox=TRUE)
