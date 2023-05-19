using DataFrames, DataFramesMeta, LinearAlgebra, Distributions, Revise, Serialization, CSV, FixedEffectModels, Optim, Parameters

string(@__DIR__) in LOAD_PATH || push!(LOAD_PATH, @__DIR__);
using demest_module; const m = demest_module

const datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data";

df = DataFrame(CSV.File(datadir*"/Analysis/demest_data.csv"));
df.delta = log.(df.shares) - log.(df.shares_out);
delta = log.(df.shares) - log.(df.shares_out);


σ = 0.1
nI = 50
ν = rand(Normal(), nI)
T = [m.Market(rr.dist, ν, zeros(nI), 0., 0., 0.) for (ii, rr) in enumerate(eachrow(df))];

m.compute_deltas!(T, σ, delta, df.shares, tol = 1e-6, maxiter=1000, verbose=true)

df.delta[1:10]
delta[1:10]