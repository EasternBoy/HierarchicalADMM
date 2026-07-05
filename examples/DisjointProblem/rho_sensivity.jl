import Pkg

Pkg.activate(".")
# Pkg.instantiate()

using Pkg, Plots, Graphs, GraphRecipes, NPZ, Statistics
using LinearAlgebra, JuMP, Ipopt
using Zygote
using ProximalAlgorithms
using ProximalOperators
using ProximalCore
using DifferentiationInterface: AutoZygote

include("TreeGeneration.jl")
include("CentralSolution.jl")
include("hADMM.jl")
include("nADMM.jl")
include("fADMM.jl")

const mode = 3  # 1: optimality gap, 2: run in specific iterations, 3: stopping criteria
const nN   = 10 
const nD   = 3

const λₙ = 0.625e-2
const λₛ = 0.625e-2
const λₕ = 0.625e-2
const tol = 1e-4
const max_iter = 1000


global countID
global opt_value

node_iter  = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])
tt_com     = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])
max_com    = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])

topo_arr = linknode[]

nTestTopo = 100

for tp in 1:nTestTopo
    println("Solving topology $tp")

    global countID = 0
    global opt_value = 0

    root = linknode(string(countID+=1))

    topo_gen!(root, nN-1, nD-1) # first layer and node are root

    #Only for disjoint problem
    assign!(root) 
    dict_result, opt_value = get_CenVars(root)

    ## Hierarchical ADMM
    reset!(root)
    traj_err, traj_res, traj_opt_hADMM = hADMM(root, dict_result)
    J_opt_hADMM     = total_cost(root)
    push!(topo_arr, deepcopy(root))
    total, max_num  = tt_com_iter(root)

    push!(node_iter["hADMM"], max_num["iter"])
    push!(max_com["hADMM"],   max_num["com"])
    push!(tt_com["hADMM"],    total["com"])


    ## Nested ADMM
    reset!(root)  #Reset variables
    nestedADMM!(root)
    total, max_num = tt_com_iter(root)
    J_opt_nADMM = total_cost(root)
    push!(node_iter["nADMM"], max_num["iter"])
    push!(max_com["nADMM"],   max_num["com"])
    push!(tt_com["nADMM"],    total["com"])    

    ## Flatten ADMM
    reset!(root)  #Reset variables
    traj_opt_fADMM = flattenADMM(root)

    total, max_num = tt_com_iter(root)
    push!(node_iter["fADMM"], max_num["iter"])
    push!(max_com["fADMM"],   max_num["com"])
    push!(tt_com["fADMM"],    total["com"])
end

medstep = Dict()
for (key, value) in node_iter
    medstep[key] = round(median(value))
    min_value, min_index = findmin(value)
    max_value, max_index = findmax(value)
    println("$key Maximum number of iterations in a node (min) avg. (max): ($min_value) $(medstep[key]) ($max_value)")
end
