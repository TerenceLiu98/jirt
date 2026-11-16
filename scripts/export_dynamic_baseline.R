.libPaths(c('outputs/r-library',.libPaths()))
library(emIRT);library(jsonlite)
load('references/emIRT/data/mq_data.rda')
x <- mq_data
out <- dynIRT(x$data.mq,.starts=x$cur.mq,.priors=x$priors.mq,.control=list(maxit=500,threads=1))
out$call <- NULL
write_json(unclass(out),'outputs/baselines/dynamic.json',digits=NA,matrix='rowmajor',auto_unbox=TRUE)
