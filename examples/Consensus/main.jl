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

const λₙ   = 1.
const λₕ   = 1.
const tol  = 1e-4
const max_iter = 1000

global countID

node_iter  = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])
tt_com     = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])
max_com    = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])

topo_arr = linknode[]


nTestTopo = 10

fontsize = 16
figPrime = plot(framestyle = :box, guidefont = font(16), tickfontsize = fontsize, xlabel = "Number of iteration in root node", yticks = [1, 0.1, 1e-2, 1e-3, 1e-4])
figRes = plot(framestyle = :box, guidefont = font(16), tickfontsize = fontsize, xlabel = "Number of iteration in root node", yticks = [1, 0.1, 1e-2, 1e-3, 1e-4])
# figJ = plot(framestyle = :box, guidefont = font(16), tickfontsize = fontsize, xlabel = "Number of iteration in root node", yticks = [1, 0.1, 1e-2, 1e-3, 1e-4])

global mat_data = zeros(1, 5)

@time for tp in 1:nTestTopo
# println("Solving topology $tp")


    global countID = 0
    global mat_data

    root = linknode(string(countID+=1))

    topo_gen!(root, nN-1, nD-1) # first layer and node are root
    assign!(root) 

    ids, levels, depths = congestion_stats(root)
    # println("Congestion node IDs: $ids")
    # println("Congestion levels: $levels")
    # println("Node depths: $depths")

    # @load "best_tree" root
    # @load "worst_tree" root

    A = Vector{Matrix{Float64}}(undef, nD-1)
    B = Vector{Matrix{Float64}}(undef, nD-1)
    topo_matrix(root, nD, A, B)


    rate_best = -Inf

    opt_sol, opt_value = get_CenVars(root)
    # print_tree(root)

    ## Hierarchical ADMM
    reset!(root)
    traj_err, traj_res, traj_opt, itr_num = hADMM(root, opt_sol)
    push!(topo_arr, deepcopy(root))
    total, max_num = tt_com_iter(root)

    # g = Graph(nN)
    # add_edge_graph!(root, g)
    # fig = graphplot(g, method=:tree)
    # annotate!(fig, 0.5, 0.5, text(string("Conv. rate =",rate_best), 12, :black))
    # display(fig)
    # png(fig, joinpath("code","HADMM-convergence","Figs",string("topo",tp,"-",rate_best)))
    begin
        rate_best = -Inf
        σbest = 0
        ρbest = 0

        for σ in 0.1:0.1:0.9
            for ρ in 0.1:0.1:1.0
                rate, t = LMIsolver(A, B, nD, ρ, σ)
                rate = round(rate, digits=4)
                t    = round(t, digits=4)
                # println("Solving with ρ = $ρ, σ = $σ, rate = $rate, t = $t")
                if rate > rate_best
                    rate_best = rate
                    σbest = σ
                    ρbest = ρ
                end
            end
        end

        if levels != []
            num_cong  = length(levels)
            max_level = maximum(levels)
            _, index  = findmax(depths)
            dep_max_level = levels[index]
        else
            num_cong  = 0
            max_level = 0
            dep_max_level = 0
        end

        # println(num_cong," ", max_level," ", dep_max_level," ", rate_best, " ", itr_num)

        row = Vector{Float64}([num_cong, max_level, dep_max_level,  rate_best, itr_num])
        println(row)
        println(second_largest_adjacency_eigenvalue(root))
        flag = true
        ndata = size(mat_data, 1)
        for i in 1:ndata
            if row == mat_data[i,:]
                flag = false
            end
        end

        if flag == true
            mat_data = vcat(mat_data, row')
        end
        # println("Best convergence rate of Topo $tp: $rate_best with ρ = $ρbest, σ = $σbest \n")
    end
end


# df = DataFrame(mat_data[2:end, :], ["#conges","max conges","deepest_conges","comput. rate","#iter"])  # auto-generates column names
# XLSX.writetable("data.xlsx", df)
# XLSX.openxlsx("converge_data_new.xlsx", mode="w") do xf
#     sheet = xf[1]
#     XLSX.writetable!(sheet, mat_data[2:end, :], "A1")
# end 