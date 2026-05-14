#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
root <- dirname(dirname(normalizePath(file_arg[[1]])))
renv_ihw <- Sys.glob(file.path(root, "renv", "library", "*", "*", "*", "IHW"))
if (length(renv_ihw) >= 1L) {
  .libPaths(c(dirname(renv_ihw[[1]]), .libPaths()))
}
setHook(
  packageEvent("lpsymphony", "onLoad"),
  function(...) {
    ns <- asNamespace("lpsymphony")
    if (!exists("lpsymphony_solve_LP", envir = ns, inherits = FALSE)) {
      return(invisible(NULL))
    }
    unlockBinding("lpsymphony_solve_LP", ns)
    fun <- get("lpsymphony_solve_LP", ns)
    walk <- function(expr) {
      if (is.call(expr)) {
        if (identical(expr[[1L]], quote(double)) && length(expr) == 1L) {
          return(quote(double(nr)))
        }
        return(as.call(lapply(expr, walk)))
      }
      expr
    }
    body(fun) <- walk(body(fun))
    assign("lpsymphony_solve_LP", fun, envir = ns)
    lockBinding("lpsymphony_solve_LP", ns)
  }
)
if (!requireNamespace("IHW", quietly = TRUE)) {
  cat("r skip IHW is not installed\n")
  quit(save = "no", status = 0L)
}
tmp_sim <- file.path(root, "tmp", "bench_sim.npz")
fallback <- file.path(root, "tests", "fixtures", "sim_n2000_seed1.npz")
sim <- if (file.exists(tmp_sim)) tmp_sim else fallback
if (!file.exists(sim)) {
  cat("r skip missing sim\n")
  quit(save = "no", status = 0L)
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
  cat("r skip python3 is required to read npz\n")
  quit(save = "no", status = 0L)
}
tmpdir <- tempfile("ihw_bench_")
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
  cat("r skip failed to load sim\n")
  quit(save = "no", status = 0L)
}
p <- as.numeric(scan(p_csv, quiet = TRUE))
x <- as.numeric(scan(x_csv, quiet = TRUE))
if (length(p) != length(x) || length(p) < 1L) {
  cat("r skip empty sim\n")
  quit(save = "no", status = 0L)
}
suppressPackageStartupMessages(library(IHW))
n_reps <- 5L
src <- if (identical(normalizePath(sim), normalizePath(tmp_sim))) {
  "tmp/bench_sim.npz"
} else {
  "tests/fixtures/sim_n2000_seed1.npz"
}
for (nfolds in c(1L, 5L)) {
  times <- numeric(n_reps)
  rej <- NA_integer_
  for (i in seq_len(n_reps)) {
    t0 <- proc.time()[[3]]
    res <- ihw(
      pvalues = p,
      covariates = x,
      alpha = 0.1,
      nbins = 4L,
      nfolds = nfolds,
      lambdas = Inf,
      seed = 1L
    )
    times[i] <- proc.time()[[3]] - t0
    rej <- as.integer(sum(adj_pvalues(res) <= 0.1, na.rm = TRUE))
  }
  times <- sort(times)
  med <- times[[(n_reps + 1L) %/% 2L]]
  cat(
    sprintf(
      "r sim %s n=%d nfolds=%d median_s %.6f rejections %d reps=%d\n",
      src,
      length(p),
      nfolds,
      med,
      rej,
      n_reps
    )
  )
}
