.libPaths(c('outputs/r-library',.libPaths()))
library(emIRT);library(jsonlite)
load('references/emIRT/data/dwnom.rda')
out <- hierIRT(dwnom$data.in,.starts=dwnom$cur,.priors=dwnom$priors,.control=list(maxit=500,threads=1))
out$call <- NULL
# Normalize cubes to a list of matrices, indexed by item/group.
out$vars$gamma <- lapply(seq_len(dim(out$vars$gamma)[3]),function(k) out$vars$gamma[,,k])
out$vars$beta2 <- lapply(seq_len(dim(out$vars$beta2)[3]),function(k) out$vars$beta2[,,k])
write_json(unclass(out),'outputs/baselines/hierarchical.json',digits=NA,matrix='rowmajor',auto_unbox=TRUE)
