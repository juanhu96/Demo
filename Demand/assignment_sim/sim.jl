# TODO: take coordinates instead of a ready-made distance matrix


using DataFrames, DataFramesMeta, Distributions, Revise, DebuggingUtilities, CSV, Parameters, Random
string(@__DIR__) in LOAD_PATH || push!(LOAD_PATH, @__DIR__);
using SimModule; const m = SimModule


datadir = "/export/storage_covidvaccine/Data"

distdf = CSV.read("$datadir/CA_dist_matrix_current.csv", DataFrame)
distmatrix = log.(Matrix(distdf) ./ 1000.)
# rows represent locations, columns represent tracts


tract_data = CSV.read("$datadir/Analysis/Demand/agent_data.csv", DataFrame)
# remove the "tracts" that are actually ZIPs TODO:

n_individuals = tract_data.weights .* tract_data.tr_pop
tract_data[!, [:weights, :tr_pop]] 
names(tract_data)

@subset!(tract_data, :tract .> 100000)

tractids = tract_data.tract
tractid_dist = CSV.read("$datadir/Intermediate/tract_nearest_dist.csv", DataFrame).tract #

tract_ind = [findfirst(tractid_dist .== tractid) for tractid in tractids]

# abd = 
tract_data.abd .= 0.; #TODO: temporary, fill in

tracts, locations = m.initialize(distmatrix = distmatrix, distcoef = [-0.5], abd = tract_data.abd, tract_ind = tract_ind, capacity = 10000, max_locations = 5);

# inspect
inspect = false
if inspect
    println(tracts[1].location_ids)
    println(tracts[1].dist)
    println(tracts[1].individuals[1].ϵ_ij)
    println(tracts[1].individuals[1].u_ij)
    println(tracts[1].individuals[1].location_ranking)
    println(tracts[1].individuals[2].ϵ_ij)
    println(tracts[1].individuals[2].u_ij)
    println(tracts[1].individuals[2].location_ranking)
end

###############
# implement Random-FCFS Mechanism
tracts1, locations1 = m.initialize(distmatrix = distmatrix, distcoef = [-0.5], abd = tract_data.abd, tract_ind = tract_ind, capacity = 10000, max_locations = 5, n_individuals = tract_data.n_individuals);
m.random_fcfs!(tracts1, locations1)

tracts2, locations2 = m.initialize(distmatrix = distmatrix, distcoef = [-0.5], abd = tract_data.abd, tract_ind = tract_ind, capacity = 500, max_locations = 5);
m.random_fcfs!(tracts2, locations2)

tracts3, locations3 = m.initialize(distmatrix = distmatrix, distcoef = [-0.5], abd = tract_data.abd, tract_ind = tract_ind, capacity = 500, max_locations = 10);
m.random_fcfs!(tracts3, locations3)



if inspect
    println(tracts[1].individuals[1].location)
    println(tracts[1].individuals[2].location)
end

###############
# implement Sequential Mechanism

m.sequential!(tracts, locations)


if inspect
    println(tracts[1].individuals[1].location)
    println(tracts[1].individuals[2].location)
end



