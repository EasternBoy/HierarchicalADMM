import Pkg

Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CSV
using DataFrames
using Plots
using Printf
using Random

const nD = 3
const nN = 20
const N_TOPOLOGIES_TO_PLOT = 1
const RNG_SEED = 2026

const DATA_DIR = joinpath(@__DIR__, "..", "..", "data", "disjoint-problem")
const FIG_DIR = joinpath(@__DIR__, "..", "..", "media", "figs", "disjoint_problem")
const TRAJECTORY_FILE = joinpath(DATA_DIR, "trajectories-D=$(nD)-N=$(nN).csv")

const TARGET_ALG = "hADMM"
const LAMBDA_COLORS = [
    "#0057B8", # blue
    "#D7191C", # red
    "#000000", # black
    "#FFD700", # yellow
    "#1A9641", # green
    "#F28E2B", # orange
    "#E7298A", # pink
    "#00BFC4", # cyan
    "#AA00FF", # magenta
    "#8C510A", # brown
]
const LAMBDA_LINESTYLES = [:solid, :dash, :dot, :dashdot, :dashdotdot]

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

function available_lambdas(df::DataFrame)
    if !(:lambda in propertynames(df))
        return [NaN]
    end

    return sort(unique(collect(skipmissing(df.lambda))))
end

function lambda_color(lambda_index::Int)
    return LAMBDA_COLORS[mod1(lambda_index, length(LAMBDA_COLORS))]
end

function lambda_linestyle(lambda_index::Int)
    linestyle_index = cld(lambda_index, length(LAMBDA_COLORS))
    return LAMBDA_LINESTYLES[mod1(linestyle_index, length(LAMBDA_LINESTYLES))]
end

function lambda_label(lambda_value::Real)
    return "lambda=$(@sprintf("%.1e", lambda_value))"
end

function plot_metric_by_topology(df::DataFrame, topologies, metric_col::Symbol, ylabel::String, output_name::String; yscale = :identity)
    lambdas = available_lambdas(filter(row -> row.alg == TARGET_ALG, df))
    topo_figs = []

    for topo in topologies
        topo_fig = plot(
            framestyle = :box,
            guidefont = font(16),
            tickfontsize = 14,
            legendfontsize = 10,
            xlabel = "Iteration",
            ylabel = ylabel,
            title = "Topology $topo",
            yscale = yscale,
            grid = true,
            legend = :bottomright,
        )

        for (lambda_index, lambda_value) in enumerate(lambdas)
            subdf = filter(row -> row.topology == topo && row.alg == TARGET_ALG && row.lambda == lambda_value, df)

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
                topo_fig,
                x_values,
                y_values,
                color = lambda_color(lambda_index),
                linestyle = lambda_linestyle(lambda_index),
                linewidth = 2.5,
                alpha = 1.0,
                label = lambda_label(lambda_value),
            )
        end

        push!(topo_figs, topo_fig)
    end

    fig = plot(
        topo_figs...,
        layout = (length(topo_figs), 1),
        size = (800, max(600, 350 * length(topo_figs))),
    )
    savefig(fig, joinpath(FIG_DIR, output_name))
    return fig
end

# function plot_gap_vs_communication(df::DataFrame, topologies, output_name::String)
#     lambdas = available_lambdas(filter(row -> row.alg == TARGET_ALG, df))
#     topo_figs = []

#     for topo in topologies
#         topo_fig = plot(
#             framestyle = :box,
#             guidefont = font(16),
#             tickfontsize = 14,
#             legendfontsize = 10,
#             xlabel = "Total Communication",
#             ylabel = "Optimality Gap (%)",
#             title = "Topology $topo",
#             # xscale = :log10,
#             yscale = :log10,
#             grid = true,
#             legend = :bottomright,
#         )

#         for (lambda_index, lambda_value) in enumerate(lambdas)
#             subdf = filter(row -> row.topology == topo && row.alg == TARGET_ALG && row.lambda == lambda_value, df)

#             if nrow(subdf) == 0
#                 continue
#             end

#             sort!(subdf, :iteration)
#             x_values = max.(subdf.total_communication, eps(Float64))
#             y_values = max.(subdf.objective_gap, eps(Float64))

#             plot!(
#                 topo_fig,
#                 x_values,
#                 y_values,
#                 color = lambda_color(lambda_index),
#                 linestyle = lambda_linestyle(lambda_index),
#                 linewidth = 2.5,
#                 alpha = 1.0,
#                 label = lambda_label(lambda_value),
#             )
#         end

#         push!(topo_figs, topo_fig)
#     end

#     fig = plot(
#         topo_figs...,
#         layout = (length(topo_figs), 1),
#         size = (800, max(600, 350 * length(topo_figs))),
#     )
#     savefig(fig, joinpath(FIG_DIR, output_name))
#     return fig
# end

function main()
    if !isfile(TRAJECTORY_FILE)
        error("Trajectory CSV not found: $TRAJECTORY_FILE")
    end

    mkpath(FIG_DIR)

    df = CSV.read(TRAJECTORY_FILE, DataFrame)
    if !(:lambda in propertynames(df))
        df.lambda = fill(1e-3, nrow(df))
    end
    df.objective_gap = abs.(df.objective .- df.optimal_value) ./ abs.(df.optimal_value) .* 100

    topologies = sampled_topologies(df)
    println("Plotting topologies: ", topologies)

    fig_gap = plot_metric_by_topology(
        df,
        topologies,
        :objective_gap,
        "Optimality Gap (%)",
        "OptGap-vs-Iteration-RandomTopologies-D=$(nD)-N=$(nN).pdf";
        yscale = :log10,
    )

    fig_comm = plot_metric_by_topology(
        df,
        topologies,
        :total_communication,
        "Total Communication",
        "TotalCommunication-vs-Iteration-RandomTopologies-D=$(nD)-N=$(nN).pdf";
        # yscale = :log10,
    )

    # fig_gap_comm = plot_gap_vs_communication(
    #     df,
    #     topologies,
    #     "OptGap-vs-TotalCommunication-RandomTopologies-D=$(nD)-N=$(nN).pdf",
    # )

    display(fig_gap)
    display(fig_comm)
    # display(fig_gap_comm)

    println("Figures saved to: ", FIG_DIR)
end

main()
