#!/bin/sh
# Install a copy of the local mirt checkout; references/ remains read-only.
set -eu
cd "$(dirname "$0")/.."
mkdir -p outputs/r-library outputs/reference-build
Rscript --vanilla -e '.libPaths(c("outputs/r-library",.libPaths())); d <- read.dcf("references/mirt/DESCRIPTION"); fields <- intersect(c("Depends","Imports","LinkingTo"),colnames(d)); deps <- unique(trimws(gsub("[[:space:]]*[(].*[)]","",unlist(strsplit(paste(d[1,fields],collapse=","),","))))); deps <- setdiff(deps,"R"); missing <- deps[!vapply(deps,requireNamespace,logical(1),quietly=TRUE)]; if(length(missing)) install.packages(missing,lib="outputs/r-library",repos="https://cloud.r-project.org"); stopifnot(all(vapply(deps,requireNamespace,logical(1),quietly=TRUE)))'
if [ ! -d outputs/reference-build/mirt ]; then
    cp -R references/mirt outputs/reference-build/mirt
fi
R_LIBS_USER="$PWD/outputs/r-library" R CMD INSTALL --library=outputs/r-library outputs/reference-build/mirt
