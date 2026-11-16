.libPaths(c('outputs/r-library', .libPaths()))
library(emIRT)
library(jsonlite)
data(s109, package='pscl')
rc <- convertRC(s109)
set.seed(42)
s <- getStarts(rc$n, rc$m, 1)
p <- makePriors(rc$n, rc$m, 1)
out <- list(votes=rc$votes, starts=s)
for (method in c('em','variational')) {
  out[[method]] <- binIRT(rc, .starts=s, .priors=p,
    .control=list(maxit=500,thresh=1e-6,asEM=method=='em',threads=1))
  out[[method]]$call <- NULL
  out[[method]] <- unclass(out[[method]])
}
dir.create('outputs/baselines',recursive=TRUE,showWarnings=FALSE)
write_json(out,'outputs/baselines/binary.json',digits=NA,matrix='rowmajor',auto_unbox=TRUE)
