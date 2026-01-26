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
include("HADMM_ProximalSolver.jl")
include("NestedADMM.jl")
include("FlattenADMM.jl")

const nN   = 10
const nD   = 3

const λₙ   = 10e-3
const λₕ   = 8e-3
const tol  = 1e-4
const max_iter = 1000

global countID

node_iter  = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])
tt_com     = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])
max_com    = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])

topo_arr = linknode[]


nTestTopo = 1000

fontsize = 16
figPrime = plot(framestyle = :box, guidefont = font(16), tickfontsize = fontsize, xlabel = "Number of iteration in root node", yticks = [1, 0.1, 1e-2, 1e-3, 1e-4])
figRes = plot(framestyle = :box, guidefont = font(16), tickfontsize = fontsize, xlabel = "Number of iteration in root node", yticks = [1, 0.1, 1e-2, 1e-3, 1e-4])
figJ = plot(framestyle = :box, guidefont = font(16), tickfontsize = fontsize, xlabel = "Number of iteration in root node", yticks = [1, 0.1, 1e-2, 1e-3, 1e-4])



@time for tp in 1:nTestTopo
    println("Solving topology $tp")

    global countID = 0

    root = linknode(string(countID+=1))

    topo_gen!(root, nN-1, nD-1) # first layer and node are root

    #Only for disjoint problem
    assign!(root) 
    dict_result, opt_value = get_CenVars(root)

    ## Hierarchical ADMM
    reset!(root)
    traj_err, traj_res, traj_opt = hADMM(root, dict_result)
    push!(topo_arr, deepcopy(root))
    total, max_num = tt_com_iter(root)

    push!(node_iter["hADMM"], max_num["iter"])
    push!(max_com["hADMM"],   max_num["com"])
    push!(tt_com["hADMM"],    total["com"])

    plot!(figPrime, 1:length(traj_err), traj_err, yscale = :log10, grid = true, label = "")
    plot!(figRes, 1:length(traj_res), traj_res, yscale = :log10, grid = true, label = "")
    plot!(figJ, 1:length(traj_opt), abs.(traj_opt .- opt_value)/maximum(abs.(traj_opt .- opt_value)), yscale = :log10, grid = true, label = "")

    # print_tree(root)

    ## Nested ADMM
    reset!(root)  #Reset variables
    nestedADMM!(root)
    total, max_num = tt_com_iter(root)
    push!(node_iter["nADMM"], max_num["iter"])
    push!(max_com["nADMM"],   max_num["com"])
    push!(tt_com["nADMM"],    total["com"])    

    ## Flatten ADMM
    reset!(root)  #Reset variables
    flattenADMM(root)
    total, max_num = tt_com_iter(root)
    push!(node_iter["fADMM"], max_num["iter"])
    push!(max_com["fADMM"],   max_num["com"])
    push!(tt_com["fADMM"],    total["com"])
    println()
end

savefig(figPrime, joinpath("media","figs","disjoint_problem",string("DJ-Prime-Conver D=",string(nD),"-N=",string(nN),".pdf")))
savefig(figRes, joinpath("media","figs","disjoint_problem",string("DJ-Res-Conver D=",string(nD),"-N=",string(nN),".pdf")))
savefig(figJ, joinpath("media","figs","disjoint_problem",string("DJ-Cost-Conver D=",string(nD),"-N=",string(nN),".pdf")))

medstep = Dict()
for (key, value) in node_iter
    medstep[key] = round(median(value))
    min_value, min_index = findmin(value)
    max_value, max_index = findmax(value)
    println("$key Maximum number of iterations in a node (min) avg. (max): ($min_value) $(medstep[key]) ($max_value)")
end
println()
for (key, value) in max_com
    med = round(median(value))
    min_value, min_index = findmin(value)
    max_value, max_index = findmax(value)
    println("$key Maximum number of scalar variables sent by a node (min) avg. (max): ($min_value) $med ($max_value)")
end
println()
for (key, value) in tt_com
    med = round(median(value))
    min_value, min_index = findmin(value)
    max_value, max_index = findmax(value)
    println("$key Total number of scalar variables sent in network (min) avg. (max): ($min_value) $med ($max_value)")
end

npzwrite(joinpath("data","disjoint-problem",string("Max-iter-D=",nD,"-N=",nN,"-h=",Int(round(medstep["hADMM"])),"-f=",Int(round(medstep["fADMM"])),".npz")), node_iter)
npzwrite(joinpath("data","disjoint-problem",string("Max-com-D=",nD,"-N=",nN,"-h=",Int(round(medstep["hADMM"])),"-f=", Int(round(medstep["fADMM"])),".npz")), max_com)
npzwrite(joinpath("data","disjoint-problem",string("Tot-com-D=",nD,"-N=",nN,"-h=",Int(round(medstep["hADMM"])),"-f=", Int(round(medstep["fADMM"])),".npz")), tt_com)

# println("Fastest Convergence Topology $least_index after $least_iter steps")
# print_tree(topo_arr[least_index])

# println("Slowest Convergence Topology $most_index after $most_iter steps")
# print_tree(topo_arr[most_index])

# g1 = Graph(nN)
# add_edge_graph!(topo_arr[least_index], g1)
# display(graphplot(g1, method=:tree))

# g2 = Graph(nN)
# add_edge_graph!(topo_arr[most_index], g2)
# display(graphplot(g2, method=:tree))
