#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
root <- dirname(dirname(normalizePath(file_arg[[1]])))
sim <- file.path(root, "tests", "fixtures", "sim_n2000_seed1.npz")
out <- file.path(root, "tests", "fixtures", "r_inf_n1.npz")
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
  nfolds = 1L,
  lambdas = Inf,
  seed = 1L
)
adj <- as.numeric(adj_pvalues(res))
w <- as.numeric(weights(res))
groups <- as.integer(res@df$group) - 1L
rej <- as.integer(sum(adj <= 0.1, na.rm = TRUE))
g_csv <- file.path(tmpdir, "groups.csv")
a_csv <- file.path(tmpdir, "adj.csv")
w_csv <- file.path(tmpdir, "w.csv")
r_txt <- file.path(tmpdir, "rej.txt")
write(groups, file = g_csv, ncolumns = 1)
write(adj, file = a_csv, ncolumns = 1)
write(w, file = w_csv, ncolumns = 1)
write(as.character(rej), file = r_txt)
save_st <- system2(
  py,
  c(
    "-c",
    shQuote(sprintf(
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
    ))
  )
)
if (!identical(save_st, 0L)) {
  stop("failed to write r oracle npz")
}
cat(sprintf("wrote %s rejections=%d\n", out, rej))
