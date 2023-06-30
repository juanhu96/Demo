module demest_module

using Optim
using Parameters
using Distributions
using Statistics
using DataFrames
using DataFramesMeta
using CSV
using LinearAlgebra
using FixedEffectModels
using StatsBase


@with_kw mutable struct Market
    dist::Float64 
    ν::Vector{Float64} # draws from normal distribution
    utils_pa::Vector{Float64} #pre-allocated utilities/exp(utilities)/shares
    denom::Float64
    shares::Float64
    abδ::Float64
end


function update_market!(t::Market, δ_t::Float64, σ::Float64)::Float64
    @unpack abδ, denom, shares, utils_pa, ν = t
    
    utils_pa .= σ .* ν .+ abδ .+ δ_t
    utils_pa .= exp.(utils_pa)
    denom = reduce(+, utils_pa) 
    utils_pa .= utils_pa ./ denom
    shares = mean(utils_pa)
    return shares
end


function compute_deltas!(
    T::Vector{Market},
    σ::Float64,
    δ_vec::Vector{Float64},
    shares_obs::Vector{Float64};
    tol::Float64=1e-6,
    maxiter::Int64=1000,
    verbose::Bool=false
)
    counter = 0
    δ_ = deepcopy(δ_vec) #initial
    dist = 1.0
    shares_iter = zeros(length(T))
    while (dist > tol && counter <= maxiter)
        for (ii, tt) in enumerate(T)
            shares_iter[ii] = update_market!(tt, δ_vec[ii], σ)
        end
        δ_vec .= δ_ .+ log.(shares_obs ./ shares_iter)

        dist = maximum(abs.(δ_vec - δ_))
        δ_ .= δ_vec
        counter += 1
    end
    if verbose
        println("Converged in $counter iterations")
    end
    return δ_vec
end

function gmm()
    
end


end