#!/usr/bin/env julia

using DelimitedFiles

function get_arg(name)
    index = findfirst(isequal(name), ARGS)
    if index === nothing || index == length(ARGS)
        return nothing
    end
    return ARGS[index + 1]
end

function fail(message, status)
    println(stderr, message)
    exit(status)
end

pvalues_path = get_arg("--pvalues")
covariates_path = get_arg("--covariates")
alpha_text = get_arg("--alpha")
nbins_text = get_arg("--nbins")
nfolds_text = get_arg("--nfolds")
lambda_policy = get_arg("--lambda-policy")
seed_text = get_arg("--seed")
output_prefix = get_arg("--output-prefix")
if any(isnothing, (pvalues_path, covariates_path, alpha_text, nbins_text, nfolds_text, lambda_policy, seed_text, output_prefix))
    fail("Julia IHW adapter arguments are incomplete", 1)
end

try
    using IndependentHypothesisWeighting
catch error
    fail("IndependentHypothesisWeighting.jl is not installed: $(error)", 3)
end

pvalues = vec(Float64.(readdlm(pvalues_path)))
covariates = vec(Float64.(readdlm(covariates_path)))
if isempty(pvalues) || length(pvalues) != length(covariates)
    fail("Julia IHW adapter received invalid input lengths", 1)
end

fit_function = if isdefined(IndependentHypothesisWeighting, :adjust_ihw)
    getfield(IndependentHypothesisWeighting, :adjust_ihw)
elseif isdefined(IndependentHypothesisWeighting, :ihw)
    getfield(IndependentHypothesisWeighting, :ihw)
else
    fail("IndependentHypothesisWeighting.jl exposes no recognized fitting function", 3)
end

lambda_value = lambda_policy == "auto" ? :auto : Inf
fit = try
    fit_function(
        pvalues,
        covariates;
        alpha = parse(Float64, alpha_text),
        nbins = parse(Int, nbins_text),
        nfolds = parse(Int, nfolds_text),
        lambdas = lambda_value,
        seed = parse(Int, seed_text)
    )
catch error
    fail("IndependentHypothesisWeighting.jl call failed: $(error)", 1)
end

if !(hasproperty(fit, :adj_pvalues) && hasproperty(fit, :weights))
    fail("IndependentHypothesisWeighting.jl result has no recognized output fields", 3)
end
adjusted = Float64.(getproperty(fit, :adj_pvalues))
weights = Float64.(getproperty(fit, :weights))
rejections = sum(adjusted .<= parse(Float64, alpha_text))
writedlm(string(output_prefix, ".adj.txt"), adjusted)
writedlm(string(output_prefix, ".weights.txt"), weights)
writedlm(string(output_prefix, ".rejections.txt"), [rejections])
