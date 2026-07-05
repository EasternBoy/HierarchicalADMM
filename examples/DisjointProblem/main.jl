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

include("Parameters.jl")

global countID
global opt_value

node_iter  = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])
tt_com     = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])
max_com    = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])

topo_arr = linknode[]


nTestTopo = 100

fontsize = 16
figPrime = plot(framestyle = :box, guidefont = font(16), tickfontsize = fontsize, xlabel = "Number of iteration in root node", yticks = [1, 0.1, 1e-2, 1e-3, 1e-4])
figRes   = plot(framestyle = :box, guidefont = font(16), tickfontsize = fontsize, xlabel = "Number of iteration in root node", yticks = [1, 0.1, 1e-2, 1e-3, 1e-4])
figJ     = plot(framestyle = :box, guidefont = font(16), tickfontsize = fontsize, xlabel = "Number of iteration in root node", yticks = [1, 0.1, 1e-2, 1e-3, 1e-4])



avg_gap_hADMM = zeros(Niter)
avg_gap_fADMM = zeros(Niter) 

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
    if mode == 2
        avg_gap_hADMM .+= abs.(traj_opt_hADMM .- opt_value)/maximum(abs.(traj_opt_hADMM .- opt_value))/nTestTopo
        avg_gap_hADMM  .= monotone_func(avg_gap_hADMM)
    end

    push!(node_iter["hADMM"], max_num["iter"])
    push!(max_com["hADMM"],   max_num["com"])
    push!(tt_com["hADMM"],    total["com"])

    plot!(figPrime, 1:length(traj_err), traj_err, yscale = :log10, grid = true, label = "")
    plot!(figRes, 1:length(traj_res), traj_res, yscale = :log10, grid = true, label = "")
    plot!(figJ, 1:length(traj_opt_hADMM), abs.(traj_opt_hADMM .- opt_value)/maximum(abs.(traj_opt_hADMM[1] .- opt_value)), yscale = :log10, grid = true, label = "")


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
    if mode == 2
        avg_gap_fADMM .+= abs.(traj_opt_fADMM .- opt_value)/maximum(abs.(traj_opt_fADMM[1] .- opt_value))/nTestTopo
        avg_gap_fADMM  .= monotone_func(avg_gap_fADMM)
    end

    total, max_num = tt_com_iter(root)
    push!(node_iter["fADMM"], max_num["iter"])
    push!(max_com["fADMM"],   max_num["com"])
    push!(tt_com["fADMM"],    total["com"])
    # println("Diff h-f ADMM optimal cost:", J_opt_hADMM - J_opt_fADMM)
    # println("Diff h-n ADMM optimal cost:", J_opt_hADMM - J_opt_nADMM)
    # println("Diff h-true:", J_opt_hADMM - opt_value)
end


savefig(figPrime, joinpath("media","figs","disjoint_problem",string("DJ-Prime-Conver-D=",string(nD),"-N=",string(nN),".pdf")))
savefig(figRes, joinpath("media","figs","disjoint_problem",string("DJ-Res-Conver-D=",string(nD),"-N=",string(nN),".pdf")))
savefig(figJ, joinpath("media","figs","disjoint_problem",string("DJ-Cost-Conver-D=",string(nD),"-N=",string(nN),".pdf")))

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
figComCost = plot(framestyle = :box, guidefont = fontsize, legendfontsize = fontsize, tickfontsize = fontsize, xlabel = "TotalVals", ylabel = "Optimality Gap", yscale = :log10, xscale = :log10, ylims = [1e-7, 2])
for (key, value) in tt_com
    med = round(median(value))
    min_value, min_index = findmin(value)
    max_value, max_index = findmax(value)
    println("$key Total number of scalar variables sent in network (min) avg. (max): ($min_value) $med ($max_value)")
    if key == "hADMM" && mode == 2
        plot!(figComCost, (1:Niter)*mean(value)/Niter, avg_gap_hADMM, label = "hADMM", linewidth = 4)
        savefig(figComCost, joinpath("media","figs","disjoint_problem",string("Cost-Com-ADMM-D=",string(nD),"-N=",string(nN),".pdf")))
    end
    if key == "fADMM" && mode == 2
        plot!(figComCost, (1:Niter)*mean(value)/Niter, avg_gap_fADMM, label = "fADMM", linewidth = 4)
        savefig(figComCost, joinpath("media","figs","disjoint_problem",string("Cost-Com-ADMM-D=",string(nD),"-N=",string(nN),".pdf")))
    end
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
