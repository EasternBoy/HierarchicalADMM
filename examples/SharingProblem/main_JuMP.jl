import Pkg
using Pkg, Plots, LinearAlgebra, Graphs, GraphRecipes, Ipopt, JuMP, ProximalAlgorithms, MadNLP


Pkg.activate(".")
Pkg.instantiate()

include("hADMM_JuMP.jl")
include("getGlobalJuMP.jl")
include("nADMM_JuMP.jl")
include("fADMM_juMP.jl")


global step_arr = Int64[]
global tt_com   = 0
global var_map  = Dict{String, Vector{Vector{Int64}}}()
global tt_vars  = 0
global opt_cost = 0.

function com_cost!(transmitted_data::Vector{Float64}, nChannel::Int64)
    global tt_com
    tt_com += length(transmitted_data)
end


network_config = [["1", "2"], ["1", "3"], ["1", "4"], 
                  ["2", "5"], ["2", "6"], ["3", "7"], ["3", "8"], ["4", "9"], ["4", "10"]]

τ = Dict("2"=> 9., "3"=> 8., "4" => 11.)
a = Dict("5"=> 2., "6"=> 5., "7"=> 3., "8"=> 2., "9"=> 3., "10"=> 5.)
β = [28., 16., 20., 22., 18., 24., 15., 18.]
ϵ = 1.0
η = 5.0

global para = parameter(τ, a, β, ϵ, η)

const λₙ       = 1.
const λₕ       = 1.4
const λₛ       = 1.5
const tol      = 1e-3
const max_iter = 200

root = linknode("1")

topo_gen!(root, network_config)

setup_network!(root, para)

dict_result, opt_cen = get_GlobalVarsJuMP(root)


traj_res, traj_J = hADMM_JuMP(root; max_iter = max_iter, λ = λₕ)
hADMM_total, hADMM_max_num = tt_com_iter(root)

reset!(root)
nestedADMM!(root; tol = tol, max_iter = max_iter, λ = λₙ)
nADMM_total, nADMM_max_num = tt_com_iter(root)
println("J_nADMM = ", total_cost(vect_prime(root), root))

reset!(root)
traj_J_fADMM, traj_res_fADMM = flattenADMM(root; tol = tol, max_iter = max_iter, λ = λₛ)
fADMM_total, fADMM_max_num = tt_com_iter(root)
println("J_fADMM = ", total_cost(vect_prime(root), root))

println("nADMM total communication: $nADMM_total, maximum number of iteration: $nADMM_max_num")
println("hADMM total communication: $hADMM_total, maximum number of iteration: $hADMM_max_num")
println("fADMM total communication: $fADMM_total, maximum number of iteration: $fADMM_max_num")


fig1 = plot(framestyle = :box, yticks = [1e0, 1e-2, 1e-4, 1e-6], size = (600,350))
fig2 = plot(framestyle = :box, yticks = [1e0, 1e-1, 1e-2, 1e-3], size = (600,350))

plot!(fig1, 1:length(traj_J), abs.(traj_J .- opt_cen) ./ abs(traj_J[1] - opt_cen), yscale = :log10, xlimit = [1, length(traj_J)], grid = true, label = "hADMM", linewidth=2, tickfont = 16, framestyle = :box)

# plot!(fig1, 1:length(traj_J_fADMM), abs.(traj_J_fADMM .- opt_cen) ./ abs(traj_J[1] - opt_cen), yscale = :log10, xlimit = [1, max(length(traj_J), length(traj_J_fADMM))], grid = true, label = "fADMM", linewidth=2, tickfont = 16)

plot!(fig2, 1:length(traj_res), traj_res, yscale = :log10, xlimit = [1, max(length(traj_res), length(traj_res_fADMM))], grid = true, label = "hADMM", linewidth=2, tickfont = 16)
# plot!(fig2, 1:length(traj_res_fADMM), traj_res_fADMM, yscale = :log10, xlimit = [1, max(length(traj_res), length(traj_res_fADMM))], grid = true, label = "fADMM", linewidth=2, tickfont = 16)

savefig(fig1, joinpath("media","figs","sharing_problem","SP-Cost-Conver.pdf"))
savefig(fig2, joinpath("media","figs","sharing_problem","SP-Res-Conver.pdf"))
