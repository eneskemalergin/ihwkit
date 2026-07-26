#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
root <- dirname(dirname(normalizePath(file_arg[[1]])))
renv_ihw <- Sys.glob(file.path(root, "renv", "library", "*", "*", "*", "IHW"))
if (length(renv_ihw) >= 1L) {
  .libPaths(c(dirname(renv_ihw[[1]]), .libPaths()))
}
trailing <- commandArgs(trailingOnly = TRUE)
nfolds <- 1L
if (length(trailing) >= 1L) {
  nfolds <- as.integer(trailing[[1]])
}
if (length(nfolds) != 1L || is.na(nfolds) || !(nfolds %in% c(1L, 5L))) {
  stop("nfolds must be 1 or 5")
}
sim_tag <- "n2000"
if (length(trailing) >= 2L) {
  sim_tag <- trailing[[2]]
}
if (identical(sim_tag, "n5000")) {
  sim <- file.path(root, "tests", "fixtures", "sim_n5000_seed42.npz")
  out <- file.path(root, "tests", "fixtures", sprintf("r_inf_n%d_n5000.npz", nfolds))
  ihw_seed <- 42L
} else if (identical(sim_tag, "n2000")) {
  sim <- file.path(root, "tests", "fixtures", "sim_n2000_seed1.npz")
  out <- file.path(root, "tests", "fixtures", sprintf("r_inf_n%d.npz", nfolds))
  ihw_seed <- 1L
} else {
  stop("sim must be n2000 or n5000")
}
if (!file.exists(sim)) {
  stop(sprintf("missing sim fixture: %s", sim))
}
if (!requireNamespace("IHW", quietly = TRUE)) {
  stop("IHW is not installed")
}
py <- Sys.which("python3")
if (!nzchar(py)) {
  py <- Sys.which("python")
}
venv_py <- file.path(root, ".venv", "bin", "python3")
if (file.exists(venv_py)) {
  py <- venv_py
}
if (!nzchar(py)) {
  stop("python3 is required to read and write npz")
}
tmpdir <- tempfile("ihw_oracle_")
dir.create(tmpdir)
on.exit(unlink(tmpdir, recursive = TRUE), add = TRUE)
p_csv <- file.path(tmpdir, "p.csv")
x_csv <- file.path(tmpdir, "x.csv")
load_st <- system2(
  py,
  c(
    "-c",
    shQuote(sprintf(
      "import numpy as np; d=np.load(r'%s'); np.savetxt(r'%s', d['p']); np.savetxt(r'%s', d['x'])",
      sim,
      p_csv,
      x_csv
    ))
  )
)
if (!identical(load_st, 0L)) {
  stop("failed to load sim fixture")
}
p <- as.numeric(scan(p_csv, quiet = TRUE))
x <- as.numeric(scan(x_csv, quiet = TRUE))
suppressPackageStartupMessages(library(IHW))
res <- ihw(
  pvalues = p,
  covariates = x,
  alpha = 0.1,
  nbins = 4L,
  nfolds = nfolds,
  lambdas = Inf,
  seed = ihw_seed
)
adj <- as.numeric(adj_pvalues(res))
w <- as.numeric(weights(res))
groups <- as.integer(res@df$group) - 1L
folds <- as.integer(res@df$fold) - 1L
rej <- as.integer(sum(adj <= 0.1, na.rm = TRUE))
if (length(groups) != length(p) || length(folds) != length(p)) {
  stop("r oracle fold or group length does not match p")
}
if (nfolds == 5L) {
  if (!identical(sort(unique(folds)), 0:4)) {
    stop("r oracle folds must be 0 .. 4")
  }
}
g_csv <- file.path(tmpdir, "groups.csv")
f_csv <- file.path(tmpdir, "folds.csv")
a_csv <- file.path(tmpdir, "adj.csv")
w_csv <- file.path(tmpdir, "w.csv")
r_txt <- file.path(tmpdir, "rej.txt")
write(groups, file = g_csv, ncolumns = 1)
write(folds, file = f_csv, ncolumns = 1)
write(adj, file = a_csv, ncolumns = 1)
write(w, file = w_csv, ncolumns = 1)
write(as.character(rej), file = r_txt)
if (identical(nfolds, 1L)) {
  save_code <- sprintf(
    paste(
      "import numpy as np",
      "p=np.loadtxt(r'%s')",
      "x=np.loadtxt(r'%s')",
      "groups=np.loadtxt(r'%s', dtype=int)",
      "adj=np.loadtxt(r'%s')",
      "w=np.loadtxt(r'%s')",
      "rej=np.int64(open(r'%s').read().strip())",
      "np.savez(r'%s', p=p, x=x, groups=groups, adj_pvalues=adj, weights=w, rejections=rej)",
      sep = ";"
    ),
    p_csv,
    x_csv,
    g_csv,
    a_csv,
    w_csv,
    r_txt,
    out
  )
} else {
  save_code <- sprintf(
    paste(
      "import numpy as np",
      "p=np.loadtxt(r'%s')",
      "x=np.loadtxt(r'%s')",
      "groups=np.loadtxt(r'%s', dtype=int)",
      "folds=np.loadtxt(r'%s', dtype=int)",
      "adj=np.loadtxt(r'%s')",
      "w=np.loadtxt(r'%s')",
      "rej=np.int64(open(r'%s').read().strip())",
      "np.savez(r'%s', p=p, x=x, groups=groups, folds=folds, adj_pvalues=adj, weights=w, rejections=rej)",
      sep = ";"
    ),
    p_csv,
    x_csv,
    g_csv,
    f_csv,
    a_csv,
    w_csv,
    r_txt,
    out
  )
}
save_st <- system2(py, c("-c", shQuote(save_code)))
if (!identical(save_st, 0L)) {
  stop("failed to write r oracle npz")
}
cat(sprintf("wrote %s rejections=%d\n", out, rej))
