using CSV
using DataFrames
using Plots
using Statistics

const nN = 30
const nD = 3

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
        filename = "WorstOptGap-DualAxis-D=$(nD)-N=$(nN).pdf",
        gap_comm_filename = "WorstOptGap-OptGap-vs-Communication-D=$(nD)-N=$(nN).pdf",
    ),
    (
        label = "Max Communication",
        topology = max_comm_topo,
        filename = "MaxCommunication-DualAxis-D=$(nD)-N=$(nN).pdf",
        gap_comm_filename = "MaxCommunication-OptGap-vs-Communication-D=$(nD)-N=$(nN).pdf",
    ),
]

alg_order = ["fADMM", "hADMM"]
alg_color = Dict("hADMM" => :blue, "fADMM" => :red)

function convergence_iterations(subdf::DataFrame)
    final_root_iter = maximum(subdf.iteration)

    if :maxnodeiters in propertynames(subdf)
        final_iter_count = subdf.maxnodeiters[end]
    else
        final_iter_count = final_root_iter
    end

    return subdf.iteration .* final_iter_count ./ final_root_iter
end

function plot_dual_axis_case(topo_df::DataFrame, case_label::String, topo, filename::String)
    fig = plot(
        framestyle = :box,
        guidefont = font(16),
        tickfontsize = 14,
        legendfontsize = 12,
        xlabel = "Iteration",
        ylabel = "Optimality Gap (%)",
        yscale = :log10,
        title = "$(case_label): Topology $(topo)",
        grid = true,
        legend = :topright,
    )
    fig_comm = twinx(fig)
    plot!(
        fig_comm,
        ylabel = "Total Communication",
        yscale = :log10,
        tickfontsize = 14,
        guidefont = font(16),
        legend = :bottomright,
    )

    for alg in alg_order
        alg_df = filter(row -> row.alg == alg, topo_df)

        if nrow(alg_df) == 0
            continue
        end

        sort!(alg_df, :iteration)
        x_values = convergence_iterations(alg_df)
        gap_values = max.(alg_df.objective_gap, eps(Float64))
        communication_values = max.(alg_df.total_communication, eps(Float64))

        plot!(
            fig,
            x_values,
            gap_values,
            color = alg_color[alg],
            linestyle = :solid,
            linewidth = 2,
            label = alg,
        )

        plot!(
            fig_comm,
            x_values,
            communication_values,
            color = alg_color[alg],
            linestyle = :dash,
            linewidth = 2,
            label = "",
        )
    end

    savefig(fig, joinpath(FIG_DIR, filename))
    display(fig)
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
        legend = :topright,
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
            color = alg_color[alg],
            linewidth = 2,
            label = alg,
        )
    end

    savefig(fig, joinpath(FIG_DIR, filename))
    display(fig)
end

for case in extreme_cases
    topo_df = filter(row -> row.topology == case.topology, df)

    plot_dual_axis_case(topo_df, case.label, case.topology, case.filename)
    plot_gap_vs_communication(topo_df, case.label, case.topology, case.gap_comm_filename)
end

println("Worst optimality gap topology: ", worst_gap_topo)
println("Max communication topology: ", max_comm_topo)
println("\nTopology scores:")
# show(topology_scores, allrows = true, allcols = true)
println()
println("Figures saved to: ", FIG_DIR)
