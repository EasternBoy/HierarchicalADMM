using CSV
using DataFrames
using Plots
using Statistics

const nN = 30
const nD = 5

const DATA_DIR = joinpath(@__DIR__, "..", "..", "data", "disjoint-problem")
const FIG_DIR = joinpath(@__DIR__, "..", "..", "media", "figs", "disjoint_problem")
const TRAJECTORY_FILE = joinpath(DATA_DIR, "trajectories-D=$(nD)-N=$(nN).csv")
const EXTREME_TRAJECTORY_FILE = joinpath(DATA_DIR, "trajectories-extreme-D=$(nD)-N=$(nN).csv")

mkpath(FIG_DIR)

all_df = CSV.read(TRAJECTORY_FILE, DataFrame)
df = isfile(EXTREME_TRAJECTORY_FILE) ? CSV.read(EXTREME_TRAJECTORY_FILE, DataFrame) : all_df

all_df.objective_gap = abs.(all_df.objective .- all_df.optimal_value) ./ abs.(all_df.optimal_value) .* 100
df.objective_gap = abs.(df.objective .- df.optimal_value) ./ abs.(df.optimal_value) .* 100

final_rows = combine(groupby(all_df, [:topology, :alg])) do sdf
    sdf[argmax(sdf.iteration), :]
end

topology_scores = combine(groupby(final_rows, :topology),
    :objective_gap => mean => :mean_objective_gap,
    :total_communication => mean => :mean_total_communication,
)

worst_gap_topo = topology_scores[argmax(topology_scores.mean_objective_gap), :topology]
max_comm_topo = topology_scores[argmax(topology_scores.mean_total_communication), :topology]

extreme_cases = [
    (
        label = "Worst Optimality Gap",
        topology = worst_gap_topo,
        y_col = :objective_gap,
        y_label = "Optimality Gap (%)",
        y_scale = :log10,
        filename = "WorstOptGap-Comparison-D=$(nD)-N=$(nN).pdf",
    ),
    (
        label = "Max Communication",
        topology = max_comm_topo,
        y_col = :total_communication,
        y_label = "Total Communication",
        y_scale = :log10,
        filename = "MaxCommunication-Comparison-D=$(nD)-N=$(nN).pdf",
    ),
]

alg_order = ["hADMM", "fADMM"]

function convergence_iterations(subdf::DataFrame)
    final_root_iter = maximum(subdf.iteration)

    if :maxnodeiters in propertynames(subdf)
        final_iter_count = subdf.maxnodeiters[end]
    else
        final_iter_count = final_root_iter
    end

    return subdf.iteration .* final_iter_count ./ final_root_iter
end

function plot_gap_vs_communication(topo_df::DataFrame, case_label::String, topo, filename::String)
    fig = plot(
        framestyle = :box,
        guidefont = font(16),
        tickfontsize = 14,
        legendfontsize = 12,
        xlabel = "Total Communication",
        ylabel = "Optimality Gap (%)",
        xscale = :log10,
        yscale = :log10,
        title = "$(case_label): Topology $(topo)",
        grid = true,
    )

    for alg in alg_order
        alg_df = filter(row -> row.alg == alg, topo_df)

        if nrow(alg_df) == 0
            continue
        end

        sort!(alg_df, :iteration)
        x_values = max.(alg_df.total_communication, eps(Float64))
        y_values = max.(alg_df.objective_gap, eps(Float64))

        plot!(
            fig,
            x_values,
            y_values,
            linewidth = 2,
            # marker = :circle,
            # markersize = 3,
            label = alg,
        )
    end

    savefig(fig, joinpath(FIG_DIR, filename))
    display(fig)
end

for case in extreme_cases
    topo_df = filter(row -> row.topology == case.topology, df)

    fig = plot(
        framestyle = :box,
        guidefont = font(16),
        tickfontsize = 14,
        legendfontsize = 12,
        xlabel = "Convergence Iteration Count",
        ylabel = case.y_label,
        yscale = case.y_scale,
        title = "$(case.label): Topology $(case.topology)",
        grid = true,
    )

    for alg in alg_order
        alg_df = filter(row -> row.alg == alg, topo_df)

        if nrow(alg_df) == 0
            continue
        end

        sort!(alg_df, :iteration)
        x_values = convergence_iterations(alg_df)
        y_values = alg_df[!, case.y_col]
        if case.y_scale == :log10
            y_values = max.(y_values, eps(Float64))
        end

        plot!(
            fig,
            x_values,
            y_values,
            linewidth = 2,
            # marker = :circle,
            # markersize = 3,
            label = alg,
        )
    end

    savefig(fig, joinpath(FIG_DIR, case.filename))
    display(fig)

    gap_comm_filename = replace(case.filename, "Comparison" => "OptGap-vs-Communication")
    plot_gap_vs_communication(topo_df, case.label, case.topology, gap_comm_filename)
end

println("Worst optimality gap topology: ", worst_gap_topo)
println("Max communication topology: ", max_comm_topo)
println("\nTopology scores:")
show(topology_scores, allrows = true, allcols = true)
println()
println("Figures saved to: ", FIG_DIR)
