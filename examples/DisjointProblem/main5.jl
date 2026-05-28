import Pkg

Pkg.activate(".")
Pkg.instantiate()

using Pkg, Plots, Graphs, GraphRecipes, NPZ, Statistics, CSV, DataFrames
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

const nN   = 30
const nD   = 7

const local_l = -2.0
const local_u = 2.0

const λn   = 1e-3
# const λf   = 1e-3
const λh = let values = Float64[], current_λ = 1e-5
    while current_λ < 1e-3
        push!(values, current_λ)
        current_λ *= 2
    end
    push!(values, 1e-3)
    values
end
const λf_values = copy(λh)
const tol  = 1e-4
const max_iter = 1000

global countID

node_iter  = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])
tt_com     = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])
max_com    = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])
root_com   = Dict("nADMM" => Int64[], "fADMM" => Int64[], "hADMM" => Int64[])
opt_gap    = Dict("nADMM" => Float64[], "fADMM" => Float64[], "hADMM" => Float64[])
primal_res = Dict("nADMM" => Float64[], "fADMM" => Float64[], "hADMM" => Float64[])
dual_res   = Dict("nADMM" => Float64[], "fADMM" => Float64[], "hADMM" => Float64[])
final_obj  = Dict("nADMM" => Float64[], "fADMM" => Float64[], "hADMM" => Float64[], "Central" => Float64[])

topo_arr = linknode[]


nTestTopo = 10

fontsize = 16
figPrime = plot(framestyle = :box, guidefont = font(16), tickfontsize = fontsize, xlabel = "Number of iteration in root node", yticks = [1, 0.1, 1e-2, 1e-3, 1e-4])
figRes = plot(framestyle = :box, guidefont = font(16), tickfontsize = fontsize, xlabel = "Number of iteration in root node", yticks = [1, 0.1, 1e-2, 1e-3, 1e-4])
figJ = plot(framestyle = :box, guidefont = font(16), tickfontsize = fontsize, xlabel = "Number of iteration in root node", yticks = [1, 0.1, 1e-2, 1e-3, 1e-4])

function print_last_residual(algorithm::String, primal::Float64, dual::Float64)
    println("$(algorithm) final residuals: primal = $(@sprintf("%.3e", primal)), dual = $(@sprintf("%.3e", dual))")
end

# Storage for all trajectories across topologies
all_trajectories = DataFrame[]

for tp in 1:nTestTopo
    println("Solving topology $tp")

    global countID = 0

    root = linknode(string(countID+=1))

    topo_gen!(root, nN-1, nD-1) 

    #Only for disjoint problem
    assign!(root) 
    dict_result, opt_value = get_CenVars(root)
    print_aggregation_constraints("Central solution", root; solution = dict_result)
    push!(final_obj["Central"], opt_value)
    push!(topo_arr, deepcopy(root))

    ## Hierarchical ADMM
    df_hADMMs = DataFrame[]
    for λh_current in λh
        println("  hADMM λ = $(@sprintf("%.3e", λh_current))")
        reset!(root)
        traj_err, traj_res, traj_opt, traj_com_h, traj_root_com_h, traj_primal_res_h, traj_dual_res_h = hADMM(root, dict_result, λ=λh_current, return_residuals = true)
        
        total, max_num = tt_com_iter(root)
        maxnodeiters_h = max_num["iter"]
        push!(node_iter["hADMM"], max_num["iter"])
        push!(max_com["hADMM"],   max_num["com"])
        push!(tt_com["hADMM"],    total["com"])
        push!(root_com["hADMM"],  root.com_cost)
        push!(opt_gap["hADMM"],   abs(traj_opt[end] - opt_value) / abs(opt_value) * 100)
        push!(primal_res["hADMM"], traj_primal_res_h[end])
        push!(dual_res["hADMM"],   traj_dual_res_h[end])
        push!(final_obj["hADMM"],  traj_opt[end])
        print_last_residual("hADMM λ=$(@sprintf("%.3e", λh_current))", traj_primal_res_h[end], traj_dual_res_h[end])
        print_aggregation_constraints("hADMM λ=$(@sprintf("%.3e", λh_current))", root)

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
    push!(node_iter["nADMM"], max_num["iter"])
    push!(max_com["nADMM"],   max_num["com"])
    push!(tt_com["nADMM"],    total["com"])
    push!(root_com["nADMM"],  root.com_cost)
    push!(opt_gap["nADMM"],   abs(traj_opt_n[end] - opt_value) / abs(opt_value) * 100)
    push!(primal_res["nADMM"], traj_primal_res_n[end])
    push!(dual_res["nADMM"],   traj_dual_res_n[end])
    push!(final_obj["nADMM"],  traj_opt_n[end])
    print_last_residual("nADMM", traj_primal_res_n[end], traj_dual_res_n[end])
    print_aggregation_constraints("nADMM", root)

    ## Flatten ADMM
    df_fADMMs = DataFrame[]
    for λf_current in λf_values
        println("  fADMM λ = $(@sprintf("%.3e", λf_current))")
    reset!(root)
    dict_prime_root_f, traj_err_f, traj_res_f, traj_opt_f, traj_com_f, traj_root_com_f, traj_primal_res_f, traj_dual_res_f = flattenADMM(root, tol=tol, λ=λf_current, max_iter=max_iter, dict_result=dict_result, return_residuals = true)
    
    total, max_num = tt_com_iter(root)
    maxnodeiters_f = max_num["iter"]
    push!(node_iter["fADMM"], max_num["iter"])
    push!(max_com["fADMM"],   max_num["com"])
    push!(tt_com["fADMM"],    total["com"])
    push!(root_com["fADMM"],  root.com_cost)
    push!(opt_gap["fADMM"],   abs(traj_opt_f[end] - opt_value) / abs(opt_value) * 100)
    push!(primal_res["fADMM"], traj_primal_res_f[end])
    push!(dual_res["fADMM"],   traj_dual_res_f[end])
    push!(final_obj["fADMM"],  traj_opt_f[end])
    print_last_residual("fADMM λ=$(@sprintf("%.3e", λf_current))", traj_primal_res_f[end], traj_dual_res_f[end])
    print_aggregation_constraints("fADMM λ=$(@sprintf("%.3e", λf_current))", root)

        plot!(figPrime, 0:(length(traj_err_f)-1), traj_err_f, yscale = :log10, grid = true, label = "")
        plot!(figRes, 0:(length(traj_res_f)-1), traj_res_f, yscale = :log10, grid = true, label = "")
        plot!(figJ, 0:(length(traj_opt_f)-1), abs.(traj_opt_f .- opt_value)/maximum(abs.(traj_opt_f .- opt_value)), yscale = :log10, grid = true, label = "")

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
            optimal_value = fill(opt_value, length(traj_err_f))
        ))
    end
    

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
        optimal_value = fill(opt_value, length(traj_err_n))
    )
    df = vcat(df_hADMMs..., df_nADMM, df_fADMMs...)
    push!(all_trajectories, df)
    println()
end

# Combine all trajectories and save to CSV
combined_trajectories = vcat(all_trajectories...)

csv_filename = joinpath("data", "disjoint-problem", string("trajectories-D=", nD, "-N=", nN, ".csv"))
CSV.write(csv_filename, combined_trajectories)
println("\nAll trajectories saved to: $csv_filename")

final_rows = combine(groupby(combined_trajectories, [:topology, :alg, :lambda])) do sdf 
    sdf[argmax(sdf.iteration), :]
end



savefig(figPrime, joinpath("media","figs","disjoint_problem",string("DJ-Prime-Conver-D=",string(nD),"-N=",string(nN),".pdf")))
savefig(figRes, joinpath("media","figs","disjoint_problem",string("DJ-Res-Conver-D=",string(nD),"-N=",string(nN),".pdf")))
savefig(figJ, joinpath("media","figs","disjoint_problem",string("DJ-Cost-Conver-D=",string(nD),"-N=",string(nN),".pdf")))

# medstep = Dict()
# for (key, value) in node_iter
#     medstep[key] = round(median(value))
#     min_value, min_index = findmin(value)
#     max_value, max_index = findmax(value)
#     println("$key Maximum number of iterations in a node (min) avg. (max): ($min_value) $(medstep[key]) ($max_value)")
# end
# println()
# for (key, value) in max_com
#     med = round(median(value))
#     min_value, min_index = findmin(value)
#     max_value, max_index = findmax(value)
#     println("$key Maximum number of scalar variables sent by a node (min) avg. (max): ($min_value) $med ($max_value)")
# end
# println()
# for (key, value) in tt_com
#     med = round(median(value))
#     min_value, min_index = findmin(value)
#     max_value, max_index = findmax(value)
#     println("$key Total number of scalar variables sent in network (min) avg. (max): ($min_value) $med ($max_value)")
# end

# # Function to print formatted summary statistics table
# function print_summary_table(metric_name::String, data_dict::Dict, format_func=x->string(round(Int, x)))
#     println("\n$metric_name")
#     println("+--------+$(repeat("-", 6))+$(repeat("-", 6))+$(repeat("-", 6))+")
#     println("| Metric | Min  | Mean | Max  |")
#     println("+--------+$(repeat("-", 6))+$(repeat("-", 6))+$(repeat("-", 6))+")
    
#     for method in ["hADMM", "fADMM", "nADMM"]
#         if haskey(data_dict, method) && !isempty(data_dict[method])
#             min_val = minimum(data_dict[method])
#             mean_val = mean(data_dict[method])
#             max_val = maximum(data_dict[method])
            
#             min_str = format_func(min_val)
#             mean_str = format_func(mean_val)
#             max_str = format_func(max_val)
            
#             println("| $(rpad(method, 6)) | $(rpad(min_str, 4)) | $(rpad(mean_str, 4)) | $(rpad(max_str, 4)) |")
#         end
#     end
#     println("+--------+$(repeat("-", 6))+$(repeat("-", 6))+$(repeat("-", 6))+")
# end

# function print_summary_table_scientific(metric_name::String, data_dict::Dict)
#     println("\n$metric_name")
#     println("+--------+$(repeat("-", 11))+$(repeat("-", 11))+$(repeat("-", 11))+")
#     println("| Metric | Min       | Mean      | Max       |")
#     println("+--------+$(repeat("-", 11))+$(repeat("-", 11))+$(repeat("-", 11))+")
    
#     for method in ["hADMM", "fADMM", "nADMM"]
#         if haskey(data_dict, method) && !isempty(data_dict[method])
#             min_val = minimum(data_dict[method])
#             mean_val = mean(data_dict[method])
#             max_val = maximum(data_dict[method])
            
#             min_str = @sprintf("%.3e", min_val)
#             mean_str = @sprintf("%.3e", mean_val)
#             max_str = @sprintf("%.3e", max_val)
            
#             println("| $(rpad(method, 6)) | $(rpad(min_str, 9)) | $(rpad(mean_str, 9)) | $(rpad(max_str, 9)) |")
#         end
#     end
#     println("+--------+$(repeat("-", 11))+$(repeat("-", 11))+$(repeat("-", 11))+")
# end

# function print_summary_table_decimal(metric_name::String, data_dict::Dict)
#     println("\n$metric_name")
#     println("+---------+$(repeat("-", 14))+$(repeat("-", 14))+$(repeat("-", 15))+")
#     println("| Metric  | Min          | Mean         | Max           |")
#     println("+---------+$(repeat("-", 14))+$(repeat("-", 14))+$(repeat("-", 15))+")
    
#     for method in ["Central", "hADMM", "fADMM", "nADMM"]
#         if haskey(data_dict, method) && !isempty(data_dict[method])
#             min_val = minimum(data_dict[method])
#             mean_val = mean(data_dict[method])
#             max_val = maximum(data_dict[method])
            
#             min_str = @sprintf("%.6f", min_val)
#             mean_str = @sprintf("%.6f", mean_val)
#             max_str = @sprintf("%.6f", max_val)
            
#             println("| $(rpad(method, 7)) | $(rpad(min_str, 12)) | $(rpad(mean_str, 12)) | $(rpad(max_str, 13)) |")
#         end
#     end
#     println("+---------+$(repeat("-", 14))+$(repeat("-", 14))+$(repeat("-", 15))+")
# end

# # Print summary statistics
# println("\n" * "="^80)
# println("SUMMARY STATISTICS")
# println("="^80)

# print_summary_table("Maximum Iterations per Node", node_iter)
# print_summary_table("Maximum Communication per Node", max_com)
# print_summary_table("Total Communication in Network", tt_com)
# print_summary_table("Root Node Communication", root_com)
# print_summary_table_scientific("Optimality Gap (%)", opt_gap)
# print_summary_table_scientific("Primal Residual", primal_res)
# print_summary_table_scientific("Dual Residual", dual_res)
# print_summary_table_decimal("Final Objective Value", final_obj)

# npzwrite(joinpath("data","disjoint-problem",string("Max-iter-D=",nD,"-N=",nN,"-h=",Int(round(medstep["hADMM"])),"-f=",Int(round(medstep["fADMM"])),".npz")), node_iter)
# npzwrite(joinpath("data","disjoint-problem",string("Max-com-D=",nD,"-N=",nN,"-h=",Int(round(medstep["hADMM"])),"-f=", Int(round(medstep["fADMM"])),".npz")), max_com)
# npzwrite(joinpath("data","disjoint-problem",string("Tot-com-D=",nD,"-N=",nN,"-h=",Int(round(medstep["hADMM"])),"-f=", Int(round(medstep["fADMM"])),".npz")), tt_com)

# # println("Fastest Convergence Topology $least_index after $least_iter steps")
# # print_tree(topo_arr[least_index])

# # println("Slowest Convergence Topology $most_index after $most_iter steps")
# # print_tree(topo_arr[most_index])

# # g1 = Graph(nN)
# # add_edge_graph!(topo_arr[least_index], g1)
# # display(graphplot(g1, method=:tree))

# # g2 = Graph(nN)
# # add_edge_graph!(topo_arr[most_index], g2)
# # display(graphplot(g2, method=:tree))
