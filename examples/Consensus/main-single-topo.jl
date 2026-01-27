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


include("TreeGenerationConsensus.jl")
include("hADMM.jl")
include("nADMM.jl")
include("fADMM.jl")
include("CentralizedSolver.jl")
include("ConvergeRateConsensus.jl")


const nN   = 8
const nD   = 3

const tol  = 1e-4
const max_iter = 1000

global countID

node_iter  = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])
tt_com     = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])
max_com    = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])




global countID = 0

root = linknode(string(countID+=1))


# network_config = [["1", "2"], ["1", "3"], ["1", "4"], ["1", "5"],
#                               ["2", "6"], ["3", "7"], ["4", "8"]]
# ρ  = 0.21
# λₕ = 1/ρ
# σ  = 0.22
# rate_best = 0.1144                              

network_config = [["1", "2"], ["2", "3"], ["2", "4"], ["2", "5"],
                              ["2", "6"], ["2", "7"], ["2", "8"]]
ρ  = 0.1
λₕ = 1/ρ
σ  = 0.22
rate_best = 0.0208                            


topo_gen!(root, network_config)
assign!(root) 



A = Vector{Matrix{Float64}}(undef, nD-1)
B = Vector{Matrix{Float64}}(undef, nD-1)
topo_matrix(root, nD, A, B)

opt_sol, opt_value = get_CenVars(root)


# ######
# rate_best = [-Inf]
# σbest     = [0.]
# ρbest     = [0.]



# for σ in 0.1:0.01:1
#     for ρ in 0.1:0.01:1
#         rate, t = LMIsolver(A, B, nD, ρ, σ)
#         rate = round(rate, digits=4)
#         t    = round(t, digits=4)
#         # println("Solving with ρ = $ρ, σ = $σ, rate = $rate, t = $t")
#         if rate > rate_best[1]
#             rate_best[1] = rate
#             σbest[1] = σ
#             ρbest[1] = ρ
#             println("Solving with ρ = $ρbest, σ = $σbest, rate = $rate, t = $t")
#         end
#     end
# end
# #####

print_tree(root)


## Hierarchical ADMM
reset!(root)
traj_err, traj_res, traj_opt, itr_num = hADMM(root, opt_sol)
total, max_num = tt_com_iter(root)


opt_root = deepcopy(root)

reset!(root)
V_traj = hADMM_V(root, opt_root)

fig  = plot!(1:length(V_traj),V_traj, yscale = :log10, label = string("γ⋆=",rate_best), 
                                    tickfont = font(14), yticks = [1e2, 1, 1e-2, 1e-4, 1e-6], framestyle = :box, legendfont = font(14), size = (600,300), ylims = [1e-6, 2e2])
savefig(fig, joinpath("media","figs","consensus_problem",string("rate=",string(round(rate_best, digits = 3)),"-V_trajectory_hADMM.pdf")))


