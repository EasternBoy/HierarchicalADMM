import Pkg
using Pkg, Plots, Graphs, GraphRecipes


Pkg.activate(".")
Pkg.instantiate()

include("myADMM_ProximalOp.jl")
include("getGlobal.jl")


global step_arr = Int64[]
global topolist = linknode[]
global nN   = 10
global nD   = 3
global tot_com = 0
global comlist = Int64[]



nTestTopo = 50

fig1 = plot(framestyle = :box)
fig2 = plot(framestyle = :box)


for tp in 1:nTestTopo
    global countID
    global step_arr
    global nN
    global nD
    global tot_com = 0

    local tol  = 1e-4
    local max_iter = 1000

    root = linknode(string(countID+=1))

    topo_gen!(root, nN-1, nD)
    assign_var!(root) #Only for disjoint problem

    dict_indepent = get_GlobalVars(root)
    dict_result   = Dict{String, Vector{Float64}}()
    assign_result!(root, dict_result, dict_indepent)

    traj_err = Float64[]
    traj_res = Float64[]
    for step in 1:max_iter
        ter = Float64[]
        # forward_hierarchicalADMM!(root)
        # backward_hierarchicalADMM!(root, ter)

        hierarchicalADMM!(root, ter)
        if maximum(ter) < tol
            step_arr = push!(step_arr, step)
            println("Topo $tp terminates at step $step")
            
            push!(topolist, root)
            push!(comlist,  tot_com)
            break
        end

        err = Float64[]
        get_err!(root, dict_result, err)
        push!(traj_res, maximum(ter))
        push!(traj_err, sum(err))
    end
    plot!(fig1, 1:length(traj_err), traj_err, yscale = :log10, grid = true, label = "")
    plot!(fig2, 1:length(traj_res), traj_res, yscale = :log10, grid = true, label = "")


    # print_tree(root)
    println()
    countID = -1 #reset counter ID
end
png(fig1, "DisjointProblem/Prime-Convergence")
png(fig2, "DisjointProblem/Residual-Convergence")

least_iter, least_index = findmin(step_arr)
most_iter, most_index = findmax(step_arr)

println("Fastest Convergence Topology $least_index after $least_iter steps")
print_tree(topolist[least_index])

println("Slowest Convergence Topology $most_index after $most_iter steps")
print_tree(topolist[most_index])

g = Graph(nN)
add_edge_graph!(topolist[least_index], g)
display(graphplot(g, method=:tree))

g2 = Graph(nN)
add_edge_graph!(topolist[most_index], g2)
display(graphplot(g2, method=:tree))

avrstep = sum(step_arr)/length(step_arr)
minstep = minimum(step_arr)
maxstep = maximum(step_arr)

avrcom = sum(comlist)/length(comlist)
mincom = minimum(comlist)
maxcom = maximum(comlist)

println("Number of steps: Min = $minstep, Average = $avrstep, Max = $maxstep")
println("Number of steps: Min = $mincom, Average = $avrcom, Max = $maxcom")