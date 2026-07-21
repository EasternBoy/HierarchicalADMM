import Pkg

Pkg.activate(".")
# Pkg.instantiate()

using Pkg, Plots, Graphs, GraphRecipes, NPZ, Statistics, JLD2
using LinearAlgebra, JuMP, Ipopt, SparseArrays
using Zygote
using ProximalAlgorithms
using ProximalOperators
using ProximalCore
using DifferentiationInterface: AutoZygote
using DataFrames, XLSX
using LaTeXStrings


include("TreeGenerationConsensus.jl")
include("hADMM.jl")
include("nADMM.jl")
include("fADMM.jl")
include("CentralizedSolver.jl")
include("ConvergeRateConsensus.jl")


const nN   = 8
const nD   = 3

node_iter  = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])
tt_com     = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])
max_com    = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])

network_config1 = [["1", "2"], ["1", "3"], ["1", "4"], ["1", "5"], ["2", "6"], ["3", "7"], ["4", "8"]]
network_config2 = [["1", "2"], ["2", "3"], ["2", "4"], ["2", "5"], ["2", "6"], ["2", "7"], ["2", "8"]]
network_config = (network_config1, network_config2)

countID = 0

fig = plot(tickfont = font(14), yticks = [1e2, 1, 1e-2, 1e-4, 1e-6], framestyle = :box, legendfont = font(13), size = (600, 300), legend=:bottomright, legendcolumns=2)


for (index, cfg) in enumerate(network_config)

    global countID = 0

    root = linknode(string(countID+=1))
    topo_gen!(root, cfg)
    assign!(root) 

    A = Vector{Matrix{Float64}}(undef, nD-1)
    B = Vector{Matrix{Float64}}(undef, nD-1)
    topo_matrix(root, nD, A, B)

    opt_sol, opt_value = get_CenVars(root)


    ######
    rate_best  = -Inf
    σ_best     = 0.
    ρ_best     = 0.
    t = 0.



    for σ in 0.1:0.01:1
        for ρ in 0.1:0.01:1
            rate, t = LMIsolver(A, B, nD, ρ, σ)
            rate = round(rate, digits=4)
            t    = round(t, digits=4)
            if rate > rate_best
                rate_best = rate
                σ_best = σ
                ρ_best = ρ
                println("Solving with ρ = $ρ_best, σ = $σ_best, rate = $rate, t = $t")
            end
        end
    end
    #####

    ## Hierarchical ADMM
    reset!(root)
    traj_err, traj_res, traj_opt, itr_num = hADMM(root, opt_sol; tol = 1e-4, λₕ = 1/ρ_best[1])
    total, max_num = tt_com_iter(root)

    opt_root = deepcopy(root)

    reset!(root)
    V_traj = hADMM_V(root, opt_root; tol = 1e-4, λₕ = 1/ρ_best[1], σ = σ_best[1])

    plot!(fig, 0:(length(V_traj)-1), V_traj ./ V_traj[1], yscale = :log10, label = string("Topology $index"), linewidth = 2)
    println("For topology $index, the best theoretical rate = $rate_best at rho = $ρ_best, sigma = $σ_best, t = $t")

    r = 1/(1+rate_best)
    if index == 1
        plot!(fig, 0:105, r.^(0:105), yscale = :log10, label = L"1/(1 + 0.097)^k", linewidth = 2, linestyle = :dot)
    else
        plot!(fig, 0:105, r.^(0:105), yscale = :log10, label = L"1/(1 + 0.023)^k", linewidth = 2, linestyle = :dot)
    end
end
display(fig)
savefig(fig,joinpath("media","figs","consensus_problem","V_trajectory_hADMM.pdf"))

