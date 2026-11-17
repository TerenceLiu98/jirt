# Optional validation dependencies; not runtime dependencies of the Python package.
local_library <- 'outputs/r-library'
dir.create(local_library, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(local_library, .libPaths()))
packages <- c('jsonlite','ltm','sirt')
missing <- packages[!vapply(packages,requireNamespace,logical(1),quietly=TRUE)]
if(length(missing)) install.packages(missing,lib=local_library,repos='https://cloud.r-project.org')
stopifnot(all(vapply(packages,requireNamespace,logical(1),quietly=TRUE)))
for(package in packages) cat(package,as.character(packageVersion(package)),'\n')
