module SimModule
using Random, Distributions, DataFrames, DataFramesMeta, CSV, Parameters


@with_kw mutable struct Individual
    tract_id::Int64
    ϵ_ij::Vector{Float64} # ϵ_ij[ll] is the random component of the utility of location ll

    u_ij::Vector{Float64} 
    location_ranking::Vector{Int64} = []
    locations_ranked::Vector{Int64} = []  # locations_ranked[rr] is the location id of the rr-th choice
    location::Int64 = 0 # location id of the location the individual is assigned to
end


struct Tract #can be a tract-HPI pair
    id::Int64
    hpi::Int64
    dist::Vector{Float64} # distance to each location
    distcoef::Float64 # coefficient on distance term
    abd::Float64 # utility except distance term
    abϵ::Vector{Float64} # abϵ[ll] = abd + distcoef * dist[ll]
    individuals::Vector{Individual}
    location_ids::Vector{Int64}
end

mutable struct Location
    id::Int64
    capacity::Int64
    occupancy::Int64
    filled::Bool
end


"""
Initialize the simulation by creating the Tract and Location objects. 
"""
function initialize(;
    distmatrix::Matrix{Float64}, # distmatrix[ll, tt] is the distance from tract tt to location ll
    distcoef::Vector{Float64}, # distcoef[qq] is the coefficient on distance term for hpi group qq
    abd::Vector{Float64},  # abd[tt] is the utility except distance term for tract tt
    tract_ind::Vector{Int64}, # tract_ind[tt] is the index of tract tt in distmatrix
    hpi::Vector{<:Real} = ones(Int64, length(abd)), # hpi[tt] is the hpi of tract tt. if omitted, all tracts are in the same hpi group
    capacity=10000, # capacity of each location
    max_locations=5, # maximum number of locations for each tract
    n_individuals=100, # number of individuals in each tract TODO: make this a vector
    seed=1234)

    if length(distcoef) != length(unique(hpi))
        error("length(distcoef) != length(unique(hpi))")
    end

    # make HPI groups Int64 if they are not already
    hpi = convert.(Int64, hpi)

    Random.seed!(seed)

    n_tracts = length(abd)

    # Identify the locations for each Tract (closest max_locations locations)
    location_ids = [sortperm(distmatrix[:,tract_ind[tt]])[1:max_locations] for tt in 1:n_tracts]
    distances = [sort(distmatrix[:,tract_ind[tt]])[1:max_locations] for tt in 1:n_tracts]

    
    # Draw ϵ_ij for each individual
    ϵ_ij = [rand(Gumbel(), max_locations) for _ in 1:n_individuals, _ in 1:n_tracts]


    # Create Tract objects
    tracts = [
        Tract(
            tt, 
            hpi[tt], 
            distances[tt], 
            distcoef[hpi[tt]], 
            abd[tt], 
            abd[tt] .+ distcoef[hpi[tt]] .* distances[tt],
            [Individual(tract_id = tt, ϵ_ij = ϵ_ij[ii,tt], u_ij = zeros(max_locations), location_ranking = ones(Int64, max_locations)) for ii in 1:n_individuals],
            location_ids[tt]) 
        for tt in 1:n_tracts]

    # Create Location objects
    n_locations = size(distmatrix, 1)
    locations = [Location(ll, capacity, 0, false) for ll in 1:n_locations]

    return tracts, locations

end


# Compute utility and ranking for each individual
function compute_ranking!(tracts::Vector{Tract})
    max_locations = length(tracts[1].location_ids)
    for tt in tracts
        for ii in tt.individuals
            ii.u_ij .= tt.abϵ .+ ii.ϵ_ij
            ii.location_ranking = sortperm(ii.u_ij, rev=true)
            ii.locations_ranked = tt.location_ids[ii.location_ranking]
        end
    end
end


"""
"Random-FCFS": First-come, first-served with a random order over all individuals in all tracts.
"""
function random_fcfs!(tracts::Vector{Tract}, locations::Vector{Location})

    # Assume tracts is your Vector{Tract}
    individuals_nested = [tract.individuals for tract in tracts]
    individuals_shuffled = shuffle(vcat(individuals_nested...))

    # Iterate over individuals in random order
    for ii in individuals_shuffled
        for ll in ii.locations_ranked
            if locations[ll].capacity > locations[ll].occupancy
                locations[ll].occupancy += 1
                ii.location = ll
                break
            end
        end
    end
end




############
"""
Mechanism "Sequential": Everyone tries their first-choice and ties are broken randomly, then everyone tries their second choice and ties are broken randomly, etc. Narratively, this would be people signing on and trying to schedule an appointment and if they fail, by the time they get to try again, everyone else will have tried once. 
"""

function sequential!(tracts::Vector{Tract}, locations::Vector{Location})
    individuals = [tract.individuals for tract in tracts]
    individuals = vcat(individuals...)

    max_locations = length(tracts[1].location_ids)
    for round in 1:max_locations
        for ii in shuffle(individuals)
            ll = ii.location_ranking[round]
            if locations[ll].capacity > locations[ll].occupancy
                locations[ll].occupancy += 1
                ii.location = ll
            end
        end
    end
end


end