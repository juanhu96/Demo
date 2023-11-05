module Assignment

using Random, Distributions, Parameters, Revise

@with_kw mutable struct Economy
    locs::Matrix{Int64}
    dists::Matrix{Float64}
    n_geogs::Int64
    ordering::Vector{Tuple{Int64, Int64}}
    abd::Vector{Float64}
    distcoefs::Vector{Float64}
    abepsilon::Matrix{Float64}
    epsilon_diff::Vector{Vector{Float64}}
    offers::Matrix{Int64}
    assignments::Matrix{Int64}
    capacity::Int64
    occupancies::Vector{Int64}
    full_bools::Vector{Bool}
end

function make_economy(
    locs::Matrix{Int64},
    dists::Matrix{Float64},
    geog_pops::Vector{Int64},
    abd::Vector{Float64},
    distcoefs::Vector{Float64};
    capacity = 10000,
    max_rank::Int64=200,
    pop_factor::Int64=1,
    shuffle::Bool=true,
    seed::Int64=1234
    )::Economy

    if pop_factor > 1
        println("Dividing population by $pop_factor")
        geog_pops = round.(Int, geog_pops ./ pop_factor)
        println("Fraction of geographies with zero population: ", mean(geog_pops .== 0))
        keepbools = geog_pops .> 0
        dists = dists[keepbools, :]
        locs = locs[keepbools, :]
        abd = abd[keepbools]
        distcoefs = distcoefs[keepbools]
        geog_pops = geog_pops[keepbools]
    end


    n_geogs = length(geog_pops)
    ordering = [(tt,ii) for tt in 1:n_geogs for ii in 1:geog_pops[tt]]
    if shuffle
        Random.seed!(seed)
        Random.shuffle!(ordering)
    end

    abepsilon = zeros(Float64, n_geogs, max_rank)
    epsilon_diff = [rand(Logistic(), geog_pops[tt]) for tt in 1:n_geogs]
    offers = zeros(Int64, n_geogs, max_rank)
    assignments = zeros(Int64, n_geogs, max_rank)
    occupancies = zeros(Int64, n_geogs)
    locs = locs[:, 1:max_rank]
    dists = dists[:, 1:max_rank]
    full_bools = falses(size(locs)[1])

    return Economy(locs, dists, n_geogs, ordering, abd, distcoefs, abepsilon, epsilon_diff, offers, assignments, capacity, occupancies, full_bools)
end


function random_fcfs(economy::Economy)::Nothing
    @unpack locs, dists, n_geogs, ordering, abepsilon, offers, assignments, occupancies, epsilon_diff, full_bools, distcoefs, capacity, abd = economy

    occupancies .= 0
    offers .= 0
    assignments .= 0

    abepsilon .= abd .+ (distcoefs .* dists) 

    for (tt,ii) in ordering
        @inbounds( for (jj,ll) in enumerate(locs[tt])
            if !full_bools[ll]
                offers[tt,jj] += 1
                if abepsilon[tt,jj] > epsilon_diff[tt][ii]
                    assignments[tt,jj] += 1
                    occupancies[ll] += 1
                    if occupancies[ll] >= capacity
                        full_bools[ll] = true
                    end
                end
            end
        end)
    end
end

end # module