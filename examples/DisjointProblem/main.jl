import Pkg

Pkg.activate(".")
Pkg.instantiate()

using Pkg, Plots, Graphs, GraphRecipes, NPZ, Statistics
using LinearAlgebra, JuMP, Ipopt
using Printf
using DelimitedFiles
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

const nN   = 20
const nD   = 1

const λₙ   = 2e-3
const λₕ   = 2e-3
const tol  = 1e-5
const max_iter = 1000

global countID

node_iter  = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])
tt_com     = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])
max_com    = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])
root_com   = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])
obj_value  = Dict("nADMM" => Float64[], "fADMM" => Float64[], "hADMM" => Float64[])
opt_gap    = Dict("nADMM" => Float64[], "fADMM" => Float64[], "hADMM" => Float64[])
primal_res = Dict("nADMM" => Float64[], "fADMM" => Float64[], "hADMM" => Float64[])
dual_res   = Dict("nADMM" => Float64[], "fADMM" => Float64[], "hADMM" => Float64[])
cen_obj_value = Float64[]

topo_arr = linknode[]


nTestTopo = 10

fontsize = 16
figPrime = plot(framestyle = :box, guidefont = font(16), tickfontsize = fontsize, xlabel = "Number of iteration in root node", yticks = [1, 0.1, 1e-2, 1e-3, 1e-4])
figRes = plot(framestyle = :box, guidefont = font(16), tickfontsize = fontsize, xlabel = "Number of iteration in root node", yticks = [1, 0.1, 1e-2, 1e-3, 1e-4])
figJ = plot(framestyle = :box, guidefont = font(16), tickfontsize = fontsize, xlabel = "Number of iteration in root node", yticks = [1, 0.1, 1e-2, 1e-3, 1e-4])

function max_primal_residual(node::linknode)
    max_res = 0.0

    if node.children !== nothing
        for child in node.children
            res = node.prime[child.ID] - vect_prime(child)
            max_res = max(max_res, maximum(abs.(res)))
            max_res = max(max_res, max_primal_residual(child))
        end
    end

    return max_res
end

function max_dict_value(dict)
    isempty(dict) && return 0.0
    return maximum(values(dict))
end

function format_fixed(x::Real; digits = 6)
    return @sprintf("%.*f", digits, float(x))
end

function format_sci(x::Real)
    return @sprintf("%.3e", float(x))
end

function format_int(x::Integer)
    return string(x)
end

function make_stats_row(label::String, values; formatter = format_fixed)
    return [
        label,
        formatter(minimum(values)),
        formatter(mean(values)),
        formatter(maximum(values)),
    ]
end

function make_metric_stats_row(algorithm::String, metric::String, values; formatter = format_fixed)
    return [
        algorithm,
        metric,
        formatter(minimum(values)),
        formatter(mean(values)),
        formatter(maximum(values)),
    ]
end

function print_table(title::String, headers::Vector{String}, rows::Vector{Vector{String}})
    widths = [length(header) for header in headers]
    for row in rows
        for i in eachindex(row)
            widths[i] = max(widths[i], length(row[i]))
        end
    end

    border = "+" * join([repeat("-", width + 2) for width in widths], "+") * "+"
    println(title)
    println(border)
    println("| " * join([rpad(headers[i], widths[i]) for i in eachindex(headers)], " | ") * " |")
    println(border)
    for row in rows
        println("| " * join([rpad(row[i], widths[i]) for i in eachindex(row)], " | ") * " |")
    end
    println(border)
end

function collect_topology_rows!(node::linknode, rows::Vector{NTuple{3, String}})
    parent_id = node.parent === nothing ? "" : node.parent.ID
    push!(rows, (node.ID, parent_id, string(node.parent === nothing)))

    if node.children !== nothing
        for child in node.children
            collect_topology_rows!(child, rows)
        end
    end
end

function save_topology_csv(root::linknode, path::String)
    mkpath(dirname(path))

    rows = NTuple{3, String}[]
    collect_topology_rows!(root, rows)

    open(path, "w") do io
        println(io, "node_id,parent_id,is_root")
        for (node_id, parent_id, is_root) in rows
            println(io, string(node_id, ",", parent_id, ",", is_root))
        end
    end
end


for tp in 1:nTestTopo
    println("Solving topology $tp")

    global countID = 0
    k = rand(2:min(nN - 1, 5))
    println("k = ", k)

    root = linknode(string(countID+=1))

    topo_gen!(root, nN-1, nD-1, mode = :unbalanced, k = k)

    topo_csv_path = joinpath(
        "data",
        "disjoint-problem",
        "topologies",
        string("Topology-", tp, "-D=", nD, "-N=", nN, ".csv"),
    )
    save_topology_csv(root, topo_csv_path)
    println("Saved topology to ", topo_csv_path)

    g_topo = Graph(nN)
    add_edge_graph!(root, g_topo)
    fig_topo = graphplot(g_topo, method = :tree)
    annotate!(fig_topo, 0.5, 0.5, text(string("Topology ", tp), 12, :black))
    savefig(fig_topo, joinpath("media", "figs", "disjoint_problem",
        string("Topology-", tp, "-D=", nD, "-N=", nN, ".pdf")))

    #Only for disjoint problem
    assign!(root) 
    dict_result, opt_value = get_CenVars(root)
    push!(cen_obj_value, opt_value)

    ## Hierarchical ADMM
    reset!(root)
    traj_err, traj_res_h, traj_opt, h_dual_map = hADMM(root, dict_result)
    total_h, max_num_h = tt_com_iter(root)
    h_obj = total_cost(root)
    h_gap = abs(h_obj - opt_value) / max(abs(opt_value), eps(Float64))
    h_pres = max_primal_residual(root)
    h_dres = max_dict_value(h_dual_map)
    push!(topo_arr, deepcopy(root))

    push!(node_iter["hADMM"], root.iteration)
    push!(max_com["hADMM"],   max_num_h["com"])
    push!(tt_com["hADMM"],    total_h["com"])
    push!(root_com["hADMM"],  root.com_cost)
    push!(obj_value["hADMM"], h_obj)
    push!(opt_gap["hADMM"], h_gap)
    push!(primal_res["hADMM"], h_pres)
    push!(dual_res["hADMM"], h_dres)

    plot!(figPrime, 1:length(traj_err), traj_err, yscale = :log10, grid = true, label = "")
    plot!(figRes, 1:length(traj_res_h), traj_res_h, yscale = :log10, grid = true, label = "")
    plot!(figJ, 1:length(traj_opt), max.(abs.(traj_opt .- opt_value) ./ max(abs(opt_value), eps(Float64)), eps(Float64)), yscale = :log10, grid = true, label = "")

    ## Nested ADMM
    reset!(root)  #Reset variables
    n_dual_map = nestedADMM!(root)
    total_n, max_num_n = tt_com_iter(root)
    n_obj = total_cost(root)
    n_gap = abs(n_obj - opt_value) / max(abs(opt_value), eps(Float64))
    n_pres = max_primal_residual(root)
    n_dres = max_dict_value(n_dual_map)
    push!(node_iter["nADMM"], max_num_n["iter"])
    push!(max_com["nADMM"],   max_num_n["com"])
    push!(tt_com["nADMM"],    total_n["com"])
    push!(root_com["nADMM"],  root.com_cost)
    push!(obj_value["nADMM"], n_obj)
    push!(opt_gap["nADMM"], n_gap)
    push!(primal_res["nADMM"], n_pres)
    push!(dual_res["nADMM"], n_dres)

    ## Flatten ADMM
    reset!(root)  #Reset variables
    _, f_dual_map = flattenADMM(root)
    total_f, max_num_f = tt_com_iter(root)
    f_obj = total_cost(root)
    f_gap = abs(f_obj - opt_value) / max(abs(opt_value), eps(Float64))
    f_pres = max_primal_residual(root)
    f_dres = max_dict_value(f_dual_map)
    push!(node_iter["fADMM"], root.iteration)
    push!(max_com["fADMM"],   max_num_f["com"])
    push!(tt_com["fADMM"],    total_f["com"])
    push!(root_com["fADMM"],  root.com_cost)
    push!(obj_value["fADMM"], f_obj)
    push!(opt_gap["fADMM"], f_gap)
    push!(primal_res["fADMM"], f_pres)
    push!(dual_res["fADMM"], f_dres)

    topo_rows = [
        ["Centralized", format_fixed(opt_value), "-", "-", "-", "-", "-"],
        ["hADMM", format_fixed(h_obj), format_sci(h_gap), format_sci(h_pres), format_sci(h_dres), format_int(total_h["com"]), format_int(root_com["hADMM"][end])],
        ["nADMM", format_fixed(n_obj), format_sci(n_gap), format_sci(n_pres), format_sci(n_dres), format_int(total_n["com"]), format_int(root_com["nADMM"][end])],
        ["fADMM", format_fixed(f_obj), format_sci(f_gap), format_sci(f_pres), format_sci(f_dres), format_int(total_f["com"]), format_int(root_com["fADMM"][end])],
    ]
    print_table(
        "Topology $tp Summary",
        ["Method", "Objective", "Opt. Gap", "Primal Res.", "Dual Res.", "Total Com.", "Root Com."],
        topo_rows,
    )
    println()
end

# savefig(figPrime, joinpath("media","figs","disjoint_problem",string("DJ-Prime-Conver-D=",string(nD),"-N=",string(nN),".pdf")))
# savefig(figRes, joinpath("media","figs","disjoint_problem",string("DJ-Res-Conver-D=",string(nD),"-N=",string(nN),".pdf")))
# savefig(figJ, joinpath("media","figs","disjoint_problem",string("DJ-Cost-Conver-D=",string(nD),"-N=",string(nN),".pdf")))

medstep = Dict(key => round(mean(value)) for (key, value) in node_iter)

print_table(
    "Communication And Iteration Summary",
    ["Algorithm", "Metric", "Min", "Avg", "Max"],
    [
        make_metric_stats_row("nADMM", "Iterations", node_iter["nADMM"]; formatter = x -> format_fixed(x, digits = 1)),
        make_metric_stats_row("fADMM", "Iterations", node_iter["fADMM"]; formatter = x -> format_fixed(x, digits = 1)),
        make_metric_stats_row("hADMM", "Iterations", node_iter["hADMM"]; formatter = x -> format_fixed(x, digits = 1)),
        make_metric_stats_row("nADMM", "Max Com.", max_com["nADMM"]; formatter = x -> format_fixed(x, digits = 1)),
        make_metric_stats_row("fADMM", "Max Com.", max_com["fADMM"]; formatter = x -> format_fixed(x, digits = 1)),
        make_metric_stats_row("hADMM", "Max Com.", max_com["hADMM"]; formatter = x -> format_fixed(x, digits = 1)),
        make_metric_stats_row("nADMM", "Root Com.", root_com["nADMM"]; formatter = x -> format_fixed(x, digits = 1)),
        make_metric_stats_row("fADMM", "Root Com.", root_com["fADMM"]; formatter = x -> format_fixed(x, digits = 1)),
        make_metric_stats_row("hADMM", "Root Com.", root_com["hADMM"]; formatter = x -> format_fixed(x, digits = 1)),
        make_metric_stats_row("nADMM", "Total Com.", tt_com["nADMM"]; formatter = x -> format_fixed(x, digits = 1)),
        make_metric_stats_row("fADMM", "Total Com.", tt_com["fADMM"]; formatter = x -> format_fixed(x, digits = 1)),
        make_metric_stats_row("hADMM", "Total Com.", tt_com["hADMM"]; formatter = x -> format_fixed(x, digits = 1)),
    ],
)
println()

print_table(
    "Objective Summary",
    ["Method", "Min", "Avg", "Max"],
    [
        make_stats_row("Centralized", cen_obj_value),
        make_stats_row("nADMM", obj_value["nADMM"]),
        make_stats_row("fADMM", obj_value["fADMM"]),
        make_stats_row("hADMM", obj_value["hADMM"]),
    ],
)
println()

print_table(
    "Residual Summary",
    ["Algorithm", "Residual", "Min", "Avg", "Max"],
    [
        ["nADMM", "Primal", format_sci(minimum(primal_res["nADMM"])), format_sci(mean(primal_res["nADMM"])), format_sci(maximum(primal_res["nADMM"]))],
        ["nADMM", "Dual", format_sci(minimum(dual_res["nADMM"])), format_sci(mean(dual_res["nADMM"])), format_sci(maximum(dual_res["nADMM"]))],
        ["fADMM", "Primal", format_sci(minimum(primal_res["fADMM"])), format_sci(mean(primal_res["fADMM"])), format_sci(maximum(primal_res["fADMM"]))],
        ["fADMM", "Dual", format_sci(minimum(dual_res["fADMM"])), format_sci(mean(dual_res["fADMM"])), format_sci(maximum(dual_res["fADMM"]))],
        ["hADMM", "Primal", format_sci(minimum(primal_res["hADMM"])), format_sci(mean(primal_res["hADMM"])), format_sci(maximum(primal_res["hADMM"]))],
        ["hADMM", "Dual", format_sci(minimum(dual_res["hADMM"])), format_sci(mean(dual_res["hADMM"])), format_sci(maximum(dual_res["hADMM"]))],
    ],
)
println()

print_table(
    "Optimality Gap Summary",
    ["Algorithm", "Min", "Avg", "Max"],
    [
        make_stats_row("nADMM", opt_gap["nADMM"]; formatter = format_sci),
        make_stats_row("fADMM", opt_gap["fADMM"]; formatter = format_sci),
        make_stats_row("hADMM", opt_gap["hADMM"]; formatter = format_sci),
    ],
)

npzwrite(joinpath("data","disjoint-problem",string("Max-iter-D=",nD,"-N=",nN,"-h=",Int(round(medstep["hADMM"])),"-f=",Int(round(medstep["fADMM"])),".npz")), node_iter)
npzwrite(joinpath("data","disjoint-problem",string("Max-com-D=",nD,"-N=",nN,"-h=",Int(round(medstep["hADMM"])),"-f=", Int(round(medstep["fADMM"])),".npz")), max_com)
npzwrite(joinpath("data","disjoint-problem",string("Tot-com-D=",nD,"-N=",nN,"-h=",Int(round(medstep["hADMM"])),"-f=", Int(round(medstep["fADMM"])),".npz")), tt_com)

least_iter, least_index = findmin(node_iter["hADMM"])
most_iter, most_index = findmax(node_iter["hADMM"])

# println("Fastest Convergence Topology $least_index after $least_iter steps")
# print_tree(topo_arr[least_index])

# println("Slowest Convergence Topology $most_index after $most_iter steps")
# print_tree(topo_arr[most_index])

# g1 = Graph(nN)
# add_edge_graph!(topo_arr[least_index], g1)
# p1 = graphplot(g1, method = :tree, title = "Fastest Topology")
# savefig(p1, joinpath("media", "figs", "disjoint_problem", string("Fastest-Topology-D=", nD, "-N=", nN, ".pdf")))

# g2 = Graph(nN)
# add_edge_graph!(topo_arr[most_index], g2)
# p2 = graphplot(g2, method = :tree, title = "Slowest Topology")
# savefig(p2, joinpath("media", "figs", "disjoint_problem", string("Slowest-Topology-D=", nD, "-N=", nN, ".pdf")))
