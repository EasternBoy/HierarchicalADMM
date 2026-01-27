import Pkg

Pkg.activate(".")
Pkg.instantiate()

using Pkg, Plots, Graphs, GraphRecipes, Ipopt


include("hADMM_Proximal.jl")
include("getGlobalProximal.jl")
include("NestedADMM.jl")


global step_arr = Int64[]
global tt_com   = 0
global var_map  = Dict{String, Vector{Vector{Int64}}}()
global tt_vars  = 0


network_config = [["1", "2"], ["1", "3"], ["1", "4"], 
                  ["2", "5"], ["2", "6"], ["3", "7"], ["3", "8"], ["4", "9"], ["4", "10"]]

τ = Dict("2"=> 9., "3"=> 8., "4" => 11.)
a = Dict("5"=> 2., "6"=> 5., "7"=> 3., "8"=> 2., "9"=> 3., "10"=> 5.)
β = [28., 16., 20., 22., 18., 24., 15., 18.]
ϵ = 1.0
η = 5.0

global para = parameter(τ, a, β, ϵ, η)
const λₙ    = 0.18
const λₕ    = 0.18
const tol   = 1e-4
const max_iter = 1000

root = linknode("1")

topo_gen!(root, network_config)

setup_network!(root, para)

dict_result, opt_cen = get_GlobalVarsProximal(root)

J_HADMM = Float64[]
Res_arr = Float64[]

for iteration in 1:max_iter
    ter = Float64[]

    cost = total_cost(vect_prime(root), root)
    push!(J_HADMM, cost)
    println("Step $iteration distant to optimal value $cost")

    HADMM_Prox!(root, ter)
    push!(Res_arr, maximum(ter))

    if maximum(ter) < tol
        println("Terminates at step $step")
        break
    end
end

hADMM_total, hADMM_max_num = tt_com_iter(root)

reset!(root)
nestedADMM!(root)
total, max_num = tt_com_iter(root)
println("nADMM total communication: $total, maximum number of iteration: $max_num")
println("hADMM total communication: $hADMM_total, maximum number of iteration: $hADMM_max_num")


fig1 = plot(framestyle = :box)
fig2 = plot(framestyle = :box)
plot!(fig1, 1:length(J_HADMM), (J_HADMM .- opt_cen)/abs(J_HADMM[1] - opt_cen) , yscale = :log10, 
                    xlimit = [1, length(J_HADMM)], grid = true, label = "",linewidth=2, yticks = [1e0, 1e-1, 1e-2, 1e-3], tickfont = 16)
plot!(fig2, 1:length(Res_arr), Res_arr, yscale = :log10, xlimit = [1, length(Res_arr)], grid = true, label = "",linewidth=2,tickfont = 16)

savefig(fig1, joinpath("media","figs","sharing_problem","SP-Cost-Conver.pdf"))
savefig(fig2, joinpath("media","figs","sharing_problem","SP-Res-Conver.pdf"))