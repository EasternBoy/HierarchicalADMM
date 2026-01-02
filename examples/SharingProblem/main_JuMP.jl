import Pkg
using Pkg, Plots, Graphs, GraphRecipes, Ipopt


Pkg.activate(".")
Pkg.instantiate()

include("hADMM_JuMP.jl")
include("getGlobalJuMP.jl")


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

tol      = 1e-3
max_iter = 100

root = linknode("1")

topo_gen!(root, network_config)

setup_network!(root, para)

dict_result, opt_cen = get_GlobalVarsJuMP(root)

traj_err = Float64[]
traj_res = Float64[]

fig1 = plot(framestyle = :box)
fig2 = plot(framestyle = :box)

for step in 1:max_iter
    ter = Float64[]

    Jdiff = total_cost(vect_prime(root), root) - opt_cen
    push!(traj_err, Jdiff)
    println("Step $step distant to optimal value $Jdiff")

    HADMM_JuMP!(root, ter)
    # HADMM_Prox!(root, ter)

    if maximum(ter) < tol
        println("Terminates at step $step")
        break
    end

    push!(traj_res, maximum(ter))
end

plot!(fig1, 1:length(traj_err), traj_err, yscale = :log10, xlimit = [1, length(traj_err)], grid = true, label = "")
plot!(fig2, 1:length(traj_res), traj_res, yscale = :log10, xlimit = [1, length(traj_res)], grid = true, label = "")

png(fig1, joinpath("code","HADMM-convergence","Figs","SP-cost-Conver"))
png(fig2, joinpath("code","HADMM-convergence","Figs","SP-Res-Conver"))