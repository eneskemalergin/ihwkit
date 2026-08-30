#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
file_args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", grep("^--file=", file_args, value = TRUE))
root <- normalizePath(file.path(dirname(file_arg[[1]]), ".."))
renv_ihw <- Sys.glob(file.path(root, "renv", "library", "*", "*", "*", "IHW"))
if (length(renv_ihw) >= 1L) {
  .libPaths(c(dirname(renv_ihw[[1]]), .libPaths()))
}
setHook(
  packageEvent("lpsymphony", "onLoad"),
  function(...) {
    namespace <- asNamespace("lpsymphony")
    if (!exists("lpsymphony_solve_LP", envir = namespace, inherits = FALSE)) {
      return(invisible(NULL))
    }
    unlockBinding("lpsymphony_solve_LP", namespace)
    solve_lp <- get("lpsymphony_solve_LP", envir = namespace)
    rewrite <- function(expression) {
      if (is.call(expression)) {
        if (identical(expression[[1L]], quote(double)) && length(expression) == 1L) {
          return(quote(double(nr)))
        }
        return(as.call(lapply(expression, rewrite)))
      }
      expression
    }
    body(solve_lp) <- rewrite(body(solve_lp))
    assign("lpsymphony_solve_LP", solve_lp, envir = namespace)
    lockBinding("lpsymphony_solve_LP", namespace)
  }
)

get_arg <- function(name) {
  index <- match(name, args)
  if (is.na(index) || index == length(args)) {
    return(NULL)
  }
  args[[index + 1L]]
}

fail <- function(message, status) {
  cat(message, "\n", file = stderr())
  quit(save = "no", status = status)
}

pvalues_path <- get_arg("--pvalues")
covariates_path <- get_arg("--covariates")
alpha_text <- get_arg("--alpha")
nbins_text <- get_arg("--nbins")
nfolds_text <- get_arg("--nfolds")
lambda_policy <- get_arg("--lambda-policy")
seed_text <- get_arg("--seed")
output_prefix <- get_arg("--output-prefix")

required <- list(
  pvalues = pvalues_path,
  covariates = covariates_path,
  alpha = alpha_text,
  nbins = nbins_text,
  nfolds = nfolds_text,
  lambda_policy = lambda_policy,
  seed = seed_text,
  output_prefix = output_prefix
)
if (any(vapply(required, is.null, logical(1)))) {
  fail("R IHW adapter arguments are incomplete", 1L)
}

if (!requireNamespace("IHW", quietly = TRUE)) {
  fail("R package IHW is not installed", 3L)
}

pvalues <- as.numeric(scan(pvalues_path, quiet = TRUE))
covariates <- as.numeric(scan(covariates_path, quiet = TRUE))
if (length(pvalues) == 0L || length(pvalues) != length(covariates)) {
  fail("R IHW adapter received invalid input lengths", 1L)
}

lambda_value <- if (identical(lambda_policy, "auto")) "auto" else Inf
set.seed(as.integer(seed_text))
fit <- tryCatch(
  IHW::ihw(
    pvalues = pvalues,
    covariates = covariates,
    alpha = as.numeric(alpha_text),
    nbins = as.integer(nbins_text),
    nfolds = as.integer(nfolds_text),
    lambdas = lambda_value,
    seed = as.integer(seed_text)
  ),
  error = function(error) {
    fail(conditionMessage(error), 1L)
  }
)

adjusted <- as.numeric(IHW::adj_pvalues(fit))
weights <- as.numeric(IHW::weights(fit))
rejections <- sum(adjusted <= as.numeric(alpha_text), na.rm = TRUE)
fit_frame <- methods::slot(fit, "df")
groups <- as.integer(fit_frame$group) - 1L
folds <- as.integer(fit_frame$fold) - 1L
fold_lambdas <- as.numeric(IHW::regularization_term(fit))
write.table(
  adjusted,
  file = paste0(output_prefix, ".adj.txt"),
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)
write.table(
  weights,
  file = paste0(output_prefix, ".weights.txt"),
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)
writeLines(
  as.character(as.integer(rejections)),
  con = paste0(output_prefix, ".rejections.txt")
)
writeLines(
  as.character(utils::packageVersion("IHW")),
  con = paste0(output_prefix, ".version.txt")
)
write.table(
  groups,
  file = paste0(output_prefix, ".groups.txt"),
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)
write.table(
  folds,
  file = paste0(output_prefix, ".folds.txt"),
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)
write.table(
  fold_lambdas,
  file = paste0(output_prefix, ".lambdas.txt"),
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)
