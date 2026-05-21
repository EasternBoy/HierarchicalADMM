import Pkg

Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CSV
using DataFrames
using Plots
using Printf
using Random

const nD = 5
const nN = 40
const N_TOPOLOGIES_TO_PLOT = 2
const RNG_SEED = 2026

const DATA_DIR = joinpath(@__DIR__, "..", "..", "data", "disjoint-problem")
const FIG_DIR = joinpath(@__DIR__, "..", "..", "media", "figs", "disjoint_problem")
const TRAJECTORY_FILE = joinpath(DATA_DIR, "trajectories-D=$(nD)-N=$(nN).csv")
const SELECTED_LAMBDA = 2.56e-4

const ALG_ORDER = ["hADMM", "fADMM"]
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

function filter_selected_lambda(df::DataFrame, selected_lambda::Real)
    if !(:lambda in propertynames(df))
        df.lambda = fill(selected_lambda, nrow(df))
    end

    has_hadmm = any(row -> row.alg == "hADMM", eachrow(df))
    has_selected_hadmm = any(row -> row.alg == "hADMM" && lambda_matches(row.lambda, selected_lambda), eachrow(df))
    if has_hadmm && !has_selected_hadmm
        available = sort(unique(collect(skipmissing(filter(row -> row.alg == "hADMM", df).lambda))))
        error("No hADMM rows found for lambda=$(lambda_label(selected_lambda)). Available hADMM lambda values: $available")
    end

    filtered_df = filter(df) do row
        row.alg != "hADMM" || lambda_matches(row.lambda, selected_lambda)
    end

    if nrow(filtered_df) == 0
        available = sort(unique(collect(skipmissing(df.lambda))))
        error("No rows found for lambda=$(lambda_label(selected_lambda)). Available lambda values: $available")
    end

    return filtered_df
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
        legend = :bottomleft,
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
    df = filter_selected_lambda(df, SELECTED_LAMBDA)
    df.objective_gap = abs.(df.objective .- df.optimal_value) ./ abs.(df.optimal_value) .* 100

    topologies = sampled_topologies(df)
    println("Plotting topologies: ", topologies)
    println("Selected hADMM lambda: ", lambda_label(SELECTED_LAMBDA))

    lambda_suffix = "-lambda=$(lambda_file_label(SELECTED_LAMBDA))"

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
