.libPaths(c('outputs/r-library', .libPaths()))
library(emIRT)
library(jsonlite)
args <- commandArgs(trailingOnly=TRUE)
model <- args[[1]]
names <- c(binary='binIRT',ordinal='ordIRT',dynamic='dynIRT',hierarchical='hierIRT',network='networkIRT',poisson='poisIRT')
original <- getExportedValue('emIRT',names[[model]])
results <- list()
wrapper <- function(...) {
  a <- list(...)
  a$.control$maxit <- 3
  a$.control$thresh <- 1e-30
  elapsed <- system.time(out <- do.call(original,a))
  label <- if(model=='binary') if(isTRUE(a$.control$asEM)) 'binary_em' else 'binary_variational' else model
  results[[label]] <<- list(elapsed_seconds=unname(elapsed[['elapsed']]),user_seconds=unname(elapsed[['user.self']]),iterations=out$runtime$iters,threads=out$runtime$threads)
  out
}
assign(names[[model]],wrapper,envir=.GlobalEnv)
# Export scripts supply the exact validated inputs; suppress baseline writes.
write_json <- function(...) invisible(NULL)
source(paste0('scripts/export_',model,'_baseline.R'))
jsonlite::write_json(results,paste0('outputs/speed-r-',model,'.json'),auto_unbox=TRUE,pretty=TRUE,digits=NA)
