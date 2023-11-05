
julia --track-allocation=user


using Serialization, Profile;
push!(LOAD_PATH, @__DIR__, string(@__DIR__)*"/Demand/assignment_sim")

using Assignment
A = Assignment


precompile(A.random_fcfs, (Assignment.Economy,))
Profile.clear_malloc_data()



economy = deserialize("economy.jls")
A.random_fcfs(economy)

exit()


#########

#=

julia 
using Coverage
analyze_malloc(".")

exit()


=#


economy = deserialize("economy.jls")

@code_warntype A.random_fcfs(economy)