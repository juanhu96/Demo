# TODO: take coordinates instead of a ready-made distance matrix


using DataFrames, DataFramesMeta, Distributions, Revise, DebuggingUtilities, CSV, Parameters, Random
string(@__DIR__) in LOAD_PATH || push!(LOAD_PATH, @__DIR__);
using SimModule; const m = SimModule


datadir = "/export/storage_covidvaccine/Data"

distdf = CSV.read("$datadir/CA_dist_matrix_current.csv", DataFrame)
distmatrix = log.(Matrix(distdf) ./ 1000.)
# rows represent locations, columns represent tracts


tract_data = CSV.read("$datadir/Analysis/Demand/agent_data.csv", DataFrame)
# remove the "tracts" that are actually ZIPs
@subset!(tract_data, :tract .> 100000)


tractids = tract_data.tract
tractid_dist = CSV.read("$datadir/Intermediate/tract_nearest_dist.csv", DataFrame).tract #

tract_ind = [findfirst(tractid_dist .== tractid) for tractid in tractids]

# abd = 
tract_data.abd .= 0.; #TODO: temporary, fill in

tracts, locations = m.initialize(distmatrix = distmatrix, distcoef = [-0.5], abd = tract_data.abd, tract_ind = tract_ind);


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
m.random_fcfs!(tracts, locations)



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



