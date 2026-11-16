#!/bin/sh
# Build only a copy; never compile inside the read-only source tree.
set -eu
cd "$(dirname "$0")/.."
mkdir -p outputs/r-library outputs/reference-build
Rscript --vanilla -e 'pkgs <- c("Rcpp", "RcppArmadillo", "pscl", "jsonlite"); .libPaths(c("outputs/r-library",.libPaths())); missing <- pkgs[!vapply(pkgs,requireNamespace,logical(1),quietly=TRUE)]; if(length(missing)) install.packages(missing,lib="outputs/r-library",repos="https://cloud.r-project.org"); stopifnot(all(vapply(pkgs,requireNamespace,logical(1),quietly=TRUE)))'
if [ ! -d outputs/reference-build/package ]; then
    cp -R references/emIRT outputs/reference-build/package
fi
R_LIBS_USER="$PWD/outputs/r-library" R CMD INSTALL --library=outputs/r-library outputs/reference-build/package
