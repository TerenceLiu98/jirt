# Export data without building or changing the reference package.
args <- commandArgs(trailingOnly = TRUE)
source_dir <- if (length(args) >= 1) args[[1]] else "references/emIRT/data"
out_dir <- if (length(args) >= 2) args[[2]] else "outputs/reference-data"
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
encode <- function(x) {
  if (is.data.frame(x)) return(list(kind="data.frame", columns=lapply(x, encode)))
  if (is.list(x)) return(lapply(x, encode))
  list(kind=typeof(x), shape=dim(x), names=names(x), dimnames=dimnames(x),
       values=as.vector(x), order="F")
}
for (file in list.files(source_dir, pattern="\\.rda$", full.names=TRUE)) {
  env <- new.env()
  objects <- load(file, envir=env)
  payload <- setNames(lapply(objects, function(n) encode(env[[n]])), objects)
  jsonlite::write_json(payload, file.path(out_dir, paste0(basename(file), ".json")),
                      digits=NA, auto_unbox=FALSE, pretty=FALSE, na="null")
}
