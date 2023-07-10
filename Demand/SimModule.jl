
using Random, Distributions, DataFrames, DataFramesMeta, CSV, Parameters


@with_kw struct Individual
    tract_id::Int64
    ϵ_ij::Vector{Float64} # ϵ_ij[ll] is the random component of the utility of location ll

    # u_ij[ll] = abd[tract_id]  distcoef[hpi[tract_id]] * dist[tract_id,ll] + ϵ_ij[ll]
    u_ij::Vector{Float64} = zeros(length(ϵ_ij))

    # location_ranking contains the location ids in order of preference (up to max_locations)
    location_ranking::Vector{Int64} = zeros(length(ϵ_ij))

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

struct Location
    id::Int64
    capacity::Int64
    occupancy::Int64
    filled::Bool
end


"""
Initialize the simulation by creating the Tract and Location objects. 
"""
function initialize(
    hpi::Vector{Int64}, # hpi[tt] is the hpi of tract tt
    distmatrix::Matrix{Float64}, 
    distcoef::Vector{Float64}, # distcoef[qq] is the coefficient on distance term for hpi group qq
    abd::Vector{Float64},  # abd[tt] is the utility except distance term for tract tt
    capacity=10000, # capacity of each location
    max_locations=5, # maximum number of locations for each tract
    n_individuals=100, # number of individuals in each tract
    seed=1234)

    Random.seed!(seed)

    n_tracts = length(abd)
    
    # Identify the locations for each Tract (closest max_locations locations)
    location_ids = [sortperm(distmatrix[tt,:])[1:max_locations] for tt in 1:n_tracts]
    distances = [sort(distmatrix[tt,:])[1:max_locations] for tt in 1:n_tracts]
    
    # Draw ϵ_ij for each individual
    ϵ_ij = [rand(Gumbel(), max_locations) for ii in 1:n_individuals, tt in 1:n_tracts]

    # Create Tract objects
    tracts = [
        Tract(
            tt, 
            hpi[tt], 
            distances[tt], 
            distcoef[hpi[tt]], 
            abd[tt], 
            abd[tt] .+ distcoef[hpi[tt]] .* distances[tt],
            [Individual(tt, ϵ_ij[ii,tt]) for ii in 1:n_individuals],
            location_ids[tt]) 
        for tt in 1:n_tracts]

    # Compute utility and ranking for each individual
    for tt in tracts
        for ii in tt.individuals
            for ll in 1:max_locations
                ii.u_ij[ll] = tt.abϵ[ll] + ii.ϵ_ij[ll]
            end
            ii.location_ranking = sortperm(ii.u_ij)
        end
    end

    # Create Location objects
    n_locations = size(distmatrix, 2)
    locations = [Location(ll, capacity, 0, false) for ll in 1:n_locations]

    return tracts, locations

end


"""
"Random-FCFS": First-come, first-served with a random order over all individuals in all tracts.
"""
function random_fcfs!(tracts::Vector{Tract}, locations::Vector{Location})
    individuals = [tract.individuals for tract in tracts]
    individuals = shuffle(vcat(individuals...))
    for ii in individuals
        for ll in ii.location_ranking
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