import Pkg

Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CSV
using DataFrames
using NPZ
using Plots
using Printf
using Random               
using Statistics

const nD = 3
const nN = 20
const N_TOPOLOGIES_TO_PLOT = 1
const RNG_SEED = 2026

const DATA_DIR = joinpath(@__DIR__, "..", "..", "data", "disjoint-problem")
const FIG_DIR = joinpath(@__DIR__, "..", "..", "media", "figs", "disjoint_problem")
const TRAJECTORY_FILE = joinpath(DATA_DIR, "trajectories-D=$(nD)-N=$(nN).csv")
const DEFAULT_SELECTED_LAMBDAS = Dict(
    "hADMM" => 4.0e-04,
    "fADMM" => 4.0e-04,
)

const ALG_ORDER = ["hADMM", "fADMM"]
const SUMMARY_ALG_ORDER = ["hADMM", "fADMM", "nADMM"]
const OBJECTIVE_ALG_ORDER = ["Central", "hADMM", "fADMM", "nADMM"]
const ALG_LINESTYLE = Dict(
    "nADMM" => :dash,
    "hADMM" => :solid,
    "fADMM" => :dot,
)
const ALG_MARKER = Dict(
    "nADMM" => :none,
    "hADMM" => :none,
    "fADMM" => :none,
)
const TOPOLOGY_COLORS = [
    :blue,
    :red,
    :green,
    :orange,
    :purple,
    :brown,
    :cyan,
    :magenta,
    :black,
    :gray,
]

function sampled_topologies(df::DataFrame; n::Int = N_TOPOLOGIES_TO_PLOT, seed::Int = RNG_SEED)
    topologies = sort(unique(df.topology))
    rng = MersenneTwister(seed)
    n_sample = min(n, length(topologies))
    return sort(shuffle(rng, topologies)[1:n_sample])
end

function convergence_iterations(subdf::DataFrame)
    final_root_iter = maximum(subdf.iteration)

    if :maxnodeiters in propertynames(subdf)
        final_iter_count = subdf.maxnodeiters[end]
    else
        final_iter_count = final_root_iter
    end

    return subdf.iteration .* final_iter_count ./ final_root_iter
end

function lambda_label(lambda_value::Real)
    return @sprintf("%.1e", lambda_value)
end

function lambda_file_label(lambda_value::Real)
    return replace(lambda_label(lambda_value), "." => "p", "+" => "")
end

function lambda_matches(lambda_value, selected_lambda::Real)
    return isapprox(Float64(lambda_value), Float64(selected_lambda); rtol = 1e-8, atol = eps(Float64))
end

function parse_lambda_arg(arg::String)
    parts = split(arg, "=", limit = 2)
    length(parts) == 2 || return nothing

    key = lowercase(strip(parts[1]))
    value = parse(Float64, strip(parts[2]))
    if key in ["--hadmm-lambda", "hadmm", "hadmm-lambda"]
        return "hADMM" => value
    elseif key in ["--fadmm-lambda", "fadmm", "fadmm-lambda"]
        return "fADMM" => value
    end
    return nothing
end

function prompt_lambda(alg::String, default_value::Real)
    print("Enter $alg lambda [default $(lambda_label(default_value))]: ")
    input = strip(readline(stdin))
    isempty(input) && return Float64(default_value)
    return parse(Float64, input)
end

function selected_lambdas_from_user()
    selected = Dict{String, Float64}(alg => Float64(DEFAULT_SELECTED_LAMBDAS[alg]) for alg in keys(DEFAULT_SELECTED_LAMBDAS))

    for arg in ARGS
        parsed = parse_lambda_arg(arg)
        parsed === nothing && continue
        selected[first(parsed)] = last(parsed)
    end

    for alg in ALG_ORDER
        has_arg = any(
            startswith(lowercase(arg), lowercase(alg)) ||
            startswith(lowercase(arg), "--" * lowercase(alg) * "-lambda")
            for arg in ARGS
        )
        if !has_arg
            selected[alg] = prompt_lambda(alg, selected[alg])
        end
    end

    return selected
end

function finite_metric_values(values)
    numeric_values = Float64[]
    for value in skipmissing(values)
        value_float = Float64(value)
        if isfinite(value_float)
            push!(numeric_values, value_float)
        end
    end
    return numeric_values
end

function final_metric_rows(df::DataFrame)
    final_rows = combine(groupby(df, [:topology, :alg, :lambda])) do sdf
        sdf[argmax(sdf.iteration), :]
    end
    final_rows.objective_gap = abs.(final_rows.objective .- final_rows.optimal_value) ./ abs.(final_rows.optimal_value) .* 100
    return final_rows
end

function metric_dict(final_rows::DataFrame, metric_col::Symbol; algorithms = SUMMARY_ALG_ORDER)
    data = Dict{String, Vector{Float64}}()
    if !(metric_col in propertynames(final_rows))
        return data
    end

    for alg in algorithms
        values = final_rows[final_rows.alg .== alg, metric_col]
        metric_values = finite_metric_values(values)
        if !isempty(metric_values)
            data[alg] = metric_values
        end
    end

    return data
end

function print_summary_table(metric_name::String, data_dict::Dict, format_func = x -> string(round(Int, x)))
    println("\n$metric_name")
    println("+--------+$(repeat("-", 6))+$(repeat("-", 6))+$(repeat("-", 6))+")
    println("| Metric | Min  | Mean | Max  |")
    println("+--------+$(repeat("-", 6))+$(repeat("-", 6))+$(repeat("-", 6))+")

    for method in SUMMARY_ALG_ORDER
        if haskey(data_dict, method) && !isempty(data_dict[method])
            min_val = minimum(data_dict[method])
            mean_val = mean(data_dict[method])
            max_val = maximum(data_dict[method])

            min_str = format_func(min_val)
            mean_str = format_func(mean_val)
            max_str = format_func(max_val)

            println("| $(rpad(method, 6)) | $(rpad(min_str, 4)) | $(rpad(mean_str, 4)) | $(rpad(max_str, 4)) |")
        end
    end
    println("+--------+$(repeat("-", 6))+$(repeat("-", 6))+$(repeat("-", 6))+")
end

function print_summary_table_scientific(metric_name::String, data_dict::Dict)
    println("\n$metric_name")
    println("+--------+$(repeat("-", 11))+$(repeat("-", 11))+$(repeat("-", 11))+")
    println("| Metric | Min       | Mean      | Max       |")
    println("+--------+$(repeat("-", 11))+$(repeat("-", 11))+$(repeat("-", 11))+")

    for method in SUMMARY_ALG_ORDER
        if haskey(data_dict, method) && !isempty(data_dict[method])
            min_val = minimum(data_dict[method])
            mean_val = mean(data_dict[method])
            max_val = maximum(data_dict[method])

            min_str = @sprintf("%.3e", min_val)
            mean_str = @sprintf("%.3e", mean_val)
            max_str = @sprintf("%.3e", max_val)

            println("| $(rpad(method, 6)) | $(rpad(min_str, 9)) | $(rpad(mean_str, 9)) | $(rpad(max_str, 9)) |")
        end
    end
    println("+--------+$(repeat("-", 11))+$(repeat("-", 11))+$(repeat("-", 11))+")
end

function print_summary_table_decimal(metric_name::String, data_dict::Dict)
    println("\n$metric_name")
    println("+---------+$(repeat("-", 14))+$(repeat("-", 14))+$(repeat("-", 15))+")
    println("| Metric  | Min          | Mean         | Max           |")
    println("+---------+$(repeat("-", 14))+$(repeat("-", 14))+$(repeat("-", 15))+")

    for method in OBJECTIVE_ALG_ORDER
        if haskey(data_dict, method) && !isempty(data_dict[method])
            min_val = minimum(data_dict[method])
            mean_val = mean(data_dict[method])
            max_val = maximum(data_dict[method])

            min_str = @sprintf("%.6f", min_val)
            mean_str = @sprintf("%.6f", mean_val)
            max_str = @sprintf("%.6f", max_val)

            println("| $(rpad(method, 7)) | $(rpad(min_str, 12)) | $(rpad(mean_str, 12)) | $(rpad(max_str, 13)) |")
        end
    end
    println("+---------+$(repeat("-", 14))+$(repeat("-", 14))+$(repeat("-", 15))+")
end

function save_metric_npz(node_iter::Dict, max_com::Dict, tt_com::Dict)
    if !(haskey(node_iter, "hADMM") && haskey(node_iter, "fADMM"))
        return
    end

    h_step = Int(round(median(node_iter["hADMM"])))
    f_step = Int(round(median(node_iter["fADMM"])))

    npzwrite(joinpath(DATA_DIR, "Max-iter-D=$(nD)-N=$(nN)-h=$(h_step)-f=$(f_step).npz"), node_iter)
    if !isempty(max_com)
        npzwrite(joinpath(DATA_DIR, "Max-com-D=$(nD)-N=$(nN)-h=$(h_step)-f=$(f_step).npz"), max_com)
    end
    npzwrite(joinpath(DATA_DIR, "Tot-com-D=$(nD)-N=$(nN)-h=$(h_step)-f=$(f_step).npz"), tt_com)
end

function print_metric_summary(df::DataFrame)
    final_rows = final_metric_rows(df)

    node_iter = metric_dict(final_rows, :maxnodeiters)
    max_com = metric_dict(final_rows, :max_communication)
    tt_com = metric_dict(final_rows, :total_communication)
    root_com = metric_dict(final_rows, :root_communication)
    opt_gap = metric_dict(final_rows, :objective_gap)
    primal_res = metric_dict(final_rows, :real_primal_residual)
    dual_res = metric_dict(final_rows, :real_dual_residual)
    final_obj = metric_dict(final_rows, :objective; algorithms = OBJECTIVE_ALG_ORDER)
    if !haskey(final_obj, "Central") && :optimal_value in propertynames(final_rows)
        central_by_topology = combine(groupby(final_rows, :topology), :optimal_value => first => :objective)
        final_obj["Central"] = finite_metric_values(central_by_topology.objective)
    end

    println("\n" * "="^80)
    println("SUMMARY STATISTICS")
    println("="^80)

    print_summary_table("Maximum Iterations per Node", node_iter)
    print_summary_table("Maximum Communication per Node", max_com)
    print_summary_table("Total Communication in Network", tt_com)
    print_summary_table("Root Node Communication", root_com)
    print_summary_table_scientific("Optimality Gap (%)", opt_gap)
    print_summary_table_scientific("Primal Residual", primal_res)
    print_summary_table_scientific("Dual Residual", dual_res)
    print_summary_table_decimal("Final Objective Value", final_obj)

    save_metric_npz(node_iter, max_com, tt_com)
end

function validate_selected_lambdas(df::DataFrame, selected_lambdas::Dict{String, Float64})
    for alg in keys(selected_lambdas)
        has_alg = any(row -> row.alg == alg, eachrow(df))
        has_alg || continue

        selected_lambda = selected_lambdas[alg]
        has_selected_lambda = any(row -> row.alg == alg && lambda_matches(row.lambda, selected_lambda), eachrow(df))
        if !has_selected_lambda
            available = sort(unique(collect(skipmissing(filter(row -> row.alg == alg, df).lambda))))
            error("No $alg rows found for lambda=$(lambda_label(selected_lambda)). Available $alg lambda values: $available")
        end
    end
end

function filter_selected_lambdas(df::DataFrame, selected_lambdas::Dict{String, Float64})
    if !(:lambda in propertynames(df))
        df.lambda = fill(NaN, nrow(df))
    end

    validate_selected_lambdas(df, selected_lambdas)

    filtered_df = filter(df) do row
        !haskey(selected_lambdas, row.alg) || lambda_matches(row.lambda, selected_lambdas[row.alg])
    end

    if nrow(filtered_df) == 0
        available = sort(unique(collect(skipmissing(df.lambda))))
        error("No rows found for selected lambdas $selected_lambdas. Available lambda values: $available")
    end

    return filtered_df
end

function lambda_file_suffix(selected_lambdas::Dict{String, Float64})
    h_label = lambda_file_label(selected_lambdas["hADMM"])
    f_label = lambda_file_label(selected_lambdas["fADMM"])
    return "-lambda-h=$(h_label)-f=$(f_label)"
end

function add_algorithm_legend!(fig)
    for alg in ALG_ORDER
        plot!(
            fig,
            [NaN],
            [NaN],
            color = :black,
            linestyle = ALG_LINESTYLE[alg],
            linewidth = 2.5,
            # marker = ALG_MARKER[alg],
            # markersize = alg == "hADMM" ? 6 : 0,
            label = alg,
        )
    end
end

function plot_metric_by_topology(df::DataFrame, topologies, metric_col::Symbol, ylabel::String, output_name::String; yscale = :identity)
    palette = TOPOLOGY_COLORS[1:min(length(topologies), length(TOPOLOGY_COLORS))]
    topo_color = Dict(topo => palette[i] for (i, topo) in enumerate(topologies))

    fig = plot(
        framestyle = :box,
        guidefont = font(16),
        tickfontsize = 14,
        legendfontsize = 12,
        xlabel = "Iteration",
        ylabel = ylabel,
        yscale = yscale,
        grid = true,
        size = (800, 600),
        legend = :topright,
    )

    for topo in topologies
        for alg in ALG_ORDER
            subdf = filter(row -> row.topology == topo && row.alg == alg, df)

            if nrow(subdf) == 0
                continue
            end

            sort!(subdf, :iteration)
            x_values = convergence_iterations(subdf)
            y_values = subdf[!, metric_col]
            if yscale == :log10
                y_values = max.(y_values, eps(Float64))
            end

            plot!(
                fig,
                x_values,
                y_values,
                color = topo_color[topo],
                linestyle = ALG_LINESTYLE[alg],
                linewidth = 2.5,
                marker = ALG_MARKER[alg],
                # markersize = alg == "hADMM" ? 4 : 0,
                # markeralpha = alg == "hADMM" ? 0.9 : 0.0,
                alpha = 1.0,
                label = "",
            )
        end
    end

    add_algorithm_legend!(fig)
    savefig(fig, joinpath(FIG_DIR, output_name))
    return fig
end

function plot_gap_vs_communication(df::DataFrame, topologies, output_name::String)
    palette = TOPOLOGY_COLORS[1:min(length(topologies), length(TOPOLOGY_COLORS))]
    topo_color = Dict(topo => palette[i] for (i, topo) in enumerate(topologies))

    fig = plot(
        framestyle = :box,
        guidefont = font(16),
        tickfontsize = 14,
        legendfontsize = 12,
        xlabel = "Total Communication",
        ylabel = "Optimality Gap (%)",
        # xscale = :log10,
        yscale = :log10,
        grid = true,
        size = (800, 600),
        legend = :topright,
    )

    for topo in topologies
        for alg in ALG_ORDER
            subdf = filter(row -> row.topology == topo && row.alg == alg, df)

            if nrow(subdf) == 0
                continue
            end

            sort!(subdf, :iteration)
            x_values = max.(subdf.total_communication, eps(Float64))
            y_values = max.(subdf.objective_gap, eps(Float64))

            plot!(
                fig,
                x_values,
                y_values,
                color = topo_color[topo],
                linestyle = ALG_LINESTYLE[alg],
                linewidth = 2.5,
                marker = ALG_MARKER[alg],
                # markersize = alg == "hADMM" ? 4 : 0,
                # markeralpha = alg == "hADMM" ? 0.9 : 0.0,
                alpha = 1.0,
                label = "",
            )
        end
    end

    add_algorithm_legend!(fig)
    savefig(fig, joinpath(FIG_DIR, output_name))
    return fig
end

function main()
    if !isfile(TRAJECTORY_FILE)
        error("Trajectory CSV not found: $TRAJECTORY_FILE")
    end

    mkpath(FIG_DIR)

    df = CSV.read(TRAJECTORY_FILE, DataFrame)
    selected_lambdas = selected_lambdas_from_user()
    df = filter_selected_lambdas(df, selected_lambdas)
    print_metric_summary(df)

    df.objective_gap = abs.(df.objective .- df.optimal_value) ./ abs.(df.optimal_value) .* 100

    topologies = sampled_topologies(df)
    println("Plotting topologies: ", topologies)
    for alg in ALG_ORDER
        println("Selected $alg lambda: ", lambda_label(selected_lambdas[alg]))
    end

    lambda_suffix = lambda_file_suffix(selected_lambdas)

    fig_gap = plot_metric_by_topology(
        df,
        topologies,
        :objective_gap,
        "Optimality Gap (%)",
        "OptGap-vs-Iteration-RandomTopologies-D=$(nD)-N=$(nN)$(lambda_suffix).pdf";
        yscale = :log10,
    )

    fig_comm = plot_metric_by_topology(
        df,
        topologies,
        :total_communication,
        "Total Communication",
        "TotalCommunication-vs-Iteration-RandomTopologies-D=$(nD)-N=$(nN)$(lambda_suffix).pdf";
        # yscale = :log10,
    )

    fig_gap_comm = plot_gap_vs_communication(
        df,
        topologies,
        "OptGap-vs-TotalCommunication-RandomTopologies-D=$(nD)-N=$(nN)$(lambda_suffix).pdf",
    )

    display(fig_gap)
    display(fig_comm)
    display(fig_gap_comm)

    println("Figures saved to: ", FIG_DIR)
end

main()
