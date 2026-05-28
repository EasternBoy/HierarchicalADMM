import Pkg

Pkg.activate(".")
Pkg.instantiate()

using Pkg, Plots, Graphs, GraphRecipes, Statistics, CSV, DataFrames
using LinearAlgebra, JuMP, Ipopt
using Zygote
using ProximalAlgorithms
using ProximalOperators
using ProximalCore
using DifferentiationInterface: AutoZygote
using Printf

include("TreeGeneration.jl")
include("CentralSolution.jl")
include("HADMM_ProximalSolver.jl")
include("NestedADMM.jl")
include("FlattenADMM.jl")

const nN   = 9
const nD   = 3

const local_l = -2.0
const local_u = 2.0

const λn   = 1e-3

const λf = let values = Float64[], current_λ = 1e-4
    while current_λ < 1e-2
        push!(values, current_λ)
        current_λ *= 2
    end
    push!(values, 1e-2)
    values
end

const λh = copy(λf)
const tol  = 1e-4
const max_iter = 1000

global countID

topo_arr = linknode[]


nTestTopo = 10

function balanced_topo_gen!(node::linknode)
    global countID

    nN == 9 || error("This balanced topology is defined for nN = 9")
    nD == 3 || error("This balanced topology is defined for nD = 3")

    branch_nodes = [linknode(string(countID += 1)) for _ in 1:2]
    set_relative!(node, branch_nodes)

    for branch in branch_nodes
        leaves = [linknode(string(countID += 1)) for _ in 1:3]
        set_relative!(branch, leaves)
    end
end

function save_topology_shape(root::linknode, name::String)
    mkpath(joinpath("media", "figs", "disjoint_problem"))
    g = Graph(nN)
    add_edge_graph!(root, g)

    fig = graphplot(
        g,
        method = :tree,
        nodeshape = :circle,
        markersize = 0.25,
        fontsize = 10,
        title = name,
    )
    savefig(fig, joinpath("media", "figs", "disjoint_problem", "$(name)-Topology-D=$(nD)-N=$(nN).pdf"))
    return fig
end

fontsize = 16
figPrime = plot(framestyle = :box, guidefont = font(16), tickfontsize = fontsize, xlabel = "Number of iteration in root node", yticks = [1, 0.1, 1e-2, 1e-3, 1e-4])
figRes = plot(framestyle = :box, guidefont = font(16), tickfontsize = fontsize, xlabel = "Number of iteration in root node", yticks = [1, 0.1, 1e-2, 1e-3, 1e-4])
figJ = plot(framestyle = :box, guidefont = font(16), tickfontsize = fontsize, xlabel = "Number of iteration in root node", yticks = [1, 0.1, 1e-2, 1e-3, 1e-4])

# Storage for all trajectories across topologies
all_trajectories = DataFrame[]

for tp in 1:nTestTopo
    println("Solving topology $tp")

    global countID = 0

    root = linknode(string(countID+=1))

    balanced_topo_gen!(root)
    display(save_topology_shape(root, "Balanced"))

    #Only for disjoint problem
    assign!(root) 
    dict_result, opt_value = get_CenVars(root)
    push!(topo_arr, deepcopy(root))

    ## Hierarchical ADMM
    df_hADMMs = DataFrame[]
    for λh_current in λh
        println("  hADMM λ = $(@sprintf("%.3e", λh_current))")
        reset!(root)
        traj_err, traj_res, traj_opt, traj_com_h, traj_root_com_h, traj_primal_res_h, traj_dual_res_h = hADMM(root, dict_result, λ=λh_current, return_residuals = true)
        
        total, max_num = tt_com_iter(root)
        maxnodeiters_h = max_num["iter"]

        plot!(figPrime, 0:(length(traj_err)-1), traj_err, yscale = :log10, grid = true, label = "")
        plot!(figRes, 0:(length(traj_res)-1), traj_res, yscale = :log10, grid = true, label = "")
        plot!(figJ, 0:(length(traj_opt)-1), abs.(traj_opt .- opt_value)/maximum(abs.(traj_opt .- opt_value)), yscale = :log10, grid = true, label = "")

        push!(df_hADMMs, DataFrame(
            topology = fill(tp, length(traj_err)),
            iteration = 0:(length(traj_err)-1),
            alg = fill("hADMM", length(traj_err)),
            lambda = fill(λh_current, length(traj_err)),
            primal_error = traj_err,
            dual_residual = traj_res,
            real_primal_residual = traj_primal_res_h,
            real_dual_residual = traj_dual_res_h,
            objective = traj_opt,
            total_communication = traj_com_h,
            root_communication = traj_root_com_h,
            maxnodeiters = fill(maxnodeiters_h, length(traj_err)),
            max_communication = fill(max_num["com"], length(traj_err)),
            optimal_value = fill(opt_value, length(traj_err))
        ))
    end

    ## Nested ADMM
    reset!(root)
    traj_err_n = Float64[]
    traj_res_n = Float64[]
    traj_opt_n = Float64[]
    traj_com_n = Float64[]
    traj_root_com_n = Float64[]
    traj_primal_res_n = Float64[]
    traj_dual_res_n = Float64[]
    initial_err_n = Float64[]
    get_err!(root, dict_result, initial_err_n)
    push!(traj_err_n, sum(initial_err_n))
    push!(traj_res_n, NaN)
    push!(traj_primal_res_n, NaN)
    push!(traj_dual_res_n, NaN)
    push!(traj_opt_n, total_cost(root))
    push!(traj_com_n, 0.0)
    push!(traj_root_com_n, 0.0)
    nestedADMM!(root, 0., tol=tol, max_iter=max_iter, dict_result=dict_result, traj_err=traj_err_n, traj_res=traj_res_n, traj_opt=traj_opt_n, traj_com=traj_com_n, traj_root_com=traj_root_com_n, traj_primal_res=traj_primal_res_n, traj_dual_res=traj_dual_res_n)
    println("nADMM root node trajectory length: ", length(traj_opt_n))
   
    if length(traj_com_n) < length(traj_opt_n)
        last_com = isempty(traj_com_n) ? 0.0 : traj_com_n[end]
        append!(traj_com_n, fill(last_com, length(traj_opt_n) - length(traj_com_n)))
    end
    if length(traj_root_com_n) < length(traj_opt_n)
        last_root_com = isempty(traj_root_com_n) ? 0.0 : traj_root_com_n[end]
        append!(traj_root_com_n, fill(last_root_com, length(traj_opt_n) - length(traj_root_com_n)))
    end
    total, max_num = tt_com_iter(root)
    maxnodeiters_n = max_num["iter"]
    maxcommunication_n = max_num["com"]

    df_nADMM = DataFrame(
        topology = fill(tp, length(traj_err_n)),
        iteration = 0:(length(traj_err_n)-1),
        alg = fill("nADMM", length(traj_err_n)),
        lambda = fill(λn, length(traj_err_n)),
        primal_error = traj_err_n,
        dual_residual = traj_res_n,
        real_primal_residual = traj_primal_res_n,
        real_dual_residual = traj_dual_res_n,
        objective = traj_opt_n,
        total_communication = traj_com_n,
        root_communication = traj_root_com_n,
        maxnodeiters = fill(maxnodeiters_n, length(traj_err_n)),
        max_communication = fill(maxcommunication_n, length(traj_err_n)),
        optimal_value = fill(opt_value, length(traj_err_n))
    )

    ## Flatten ADMM
    df_fADMMs = DataFrame[]
    for λf_current in λf
        println("  fADMM λ = $(@sprintf("%.3e", λf_current))")
        reset!(root)
        dict_prime_root_f, traj_err_f, traj_res_f, traj_opt_f, traj_com_f, traj_root_com_f, traj_primal_res_f, traj_dual_res_f = flattenADMM(root, tol=tol, λ=λf_current, max_iter=max_iter, dict_result=dict_result, return_residuals = true)

        total, max_num = tt_com_iter(root)
        maxnodeiters_f = max_num["iter"]
        maxcommunication_f = max_num["com"]

        push!(df_fADMMs, DataFrame(
            topology = fill(tp, length(traj_err_f)),
            iteration = 0:(length(traj_err_f)-1),
            alg = fill("fADMM", length(traj_err_f)),
            lambda = fill(λf_current, length(traj_err_f)),
            primal_error = traj_err_f,
            dual_residual = traj_res_f,
            real_primal_residual = traj_primal_res_f,
            real_dual_residual = traj_dual_res_f,
            objective = traj_opt_f,
            total_communication = traj_com_f,
            root_communication = traj_root_com_f,
            maxnodeiters = fill(maxnodeiters_f, length(traj_err_f)),
            max_communication = fill(maxcommunication_f, length(traj_err_f)),
            optimal_value = fill(opt_value, length(traj_err_f))
        ))
    end

    df_central = DataFrame(
        topology = [tp],
        iteration = [0],
        alg = ["Central"],
        lambda = [NaN],
        primal_error = [0.0],
        dual_residual = [NaN],
        real_primal_residual = [0.0],
        real_dual_residual = [NaN],
        objective = [opt_value],
        total_communication = [0.0],
        root_communication = [0.0],
        maxnodeiters = [0],
        max_communication = [0],
        optimal_value = [opt_value]
    )
   
    df = vcat(df_hADMMs..., df_nADMM, df_fADMMs..., df_central)
    push!(all_trajectories, df)
    println()
end

# Combine all trajectories and save to CSV
combined_trajectories = vcat(all_trajectories...)

csv_filename = joinpath("data", "disjoint-problem", string("balance_tree-D=", nD, "-N=", nN, ".csv"))
CSV.write(csv_filename, combined_trajectories)
println("\nAll trajectories saved to: $csv_filename")


