# test how fast randon first-come-first-serve assignment is in julia. 


using DataFrames, DataFramesMeta, CSV, Parameters, Dates, Revise, BenchmarkTools, Serialization, StatsBase

push!(LOAD_PATH, @__DIR__)
using Assignment
A = Assignment

datadir = "/export/storage_covidvaccine/Data"
outdir = "/export/storage_covidvaccine/Result/Demand"

testing = false #TODO:

if testing
    economy = deserialize("economy.jls")
end


capacity = 10000;

distdf = CSV.File("$datadir/Intermediate/ca_blk_pharm_dist_blockstokeep.csv", ) |> DataFrame
# blkid, locid, logdist
distdf.locid = convert.(Int64, distdf.locid) .+ 1 # julia is 1-indexed
length(unique(distdf.blkid))

agent_results = CSV.File("$outdir/agent_results_$(Int(capacity))_200_3q.csv") |> DataFrame # blkid, hpi_quantile, market_ids, abd, distcoef

cw_pop = CSV.File("$datadir/Analysis/Demand/block_data.csv") |> DataFrame # zip, blkid, logdist, population, weights, market_ids, nodes
cw_pop.blkid = convert.(Int64, cw_pop.blkid)
blocks_inall = unique(agent_results.blkid)
sort!(cw_pop, :blkid)
cw_pop = cw_pop[in.(cw_pop.blkid, Ref(blocks_inall)), :]


if testing 
    @subset!(distdf, :blkid .<= 500)
    @subset!(cw_pop, :blkid .<= 500)
    @subset!(agent_results, :blkid .<= 500)
    mapping = Dict(val => idx for (idx, val) in enumerate(unique(distdf.locid)))
    distdf.locid = map(x -> mapping[x], distdf.locid)
    # so IDs go from 1 to however many locations are left
end

geog_pops = convert(Vector{Int64}, cw_pop.population);

distcoefs = agent_results.distcoef;
abd = agent_results.abd;

dist_grp = groupby(distdf, :blkid); 
n_geogs = length(unique(distdf.blkid));
locs = [convert(Vector{Int64}, bb.locid) for bb in dist_grp]; 
locs = reshape(vcat(locs...), (n_geogs, length(locs[1])));

dists = [convert(Vector{Float64}, bb.logdist) for bb in dist_grp];
dists = reshape(vcat(dists...), (n_geogs, length(dists[1])));


economy = A.make_economy(locs, dists, geog_pops, abd, distcoefs, capacity = capacity, pop_factor = 10, max_rank=100);

if testing
    serialize("economy.jls", economy)
end

precompile(A.random_fcfs, (Assignment.Economy,))



# economy = deserialize("economy.jls")
@benchmark A.random_fcfs(economy) seconds=5
A.random_fcfs(economy)
@btime A.random_fcfs(economy)
@time A.random_fcfs(economy)

@assert all(max_rank .== length.(economy.locs))

mean(cw_pop.population .< 50)