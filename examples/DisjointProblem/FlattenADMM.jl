function collect_nodes!(node::linknode, flatten_nodes)
    if node.children !== nothing
        for child in node.children
            push!(flatten_nodes, child.ID => child)
            collect_nodes!(child, flatten_nodes)
        end
    elseif node.parent !== nothing
        push!(flatten_nodes, node.ID => node)
    end
end

function flatten_solution_snapshot(root::linknode)
    snapshot = Dict()
    collect_solution_snapshot!(root, snapshot)
    return snapshot
end

function collect_solution_snapshot!(node::linknode, snapshot::Dict)
    snapshot[node.ID] = vect_prime(node)
    if node.children !== nothing
        for child in node.children
            collect_solution_snapshot!(child, snapshot)
        end
    end
end

function flattenADMM(root::linknode; tol = tol, λ = λf, max_iter = max_iter, dict_result = nothing, return_residuals = false)
    traj_err = Float64[]
    traj_res = Float64[]
    traj_primal_res = Float64[]
    traj_dual_res = Float64[]
    traj_opt = Float64[]
    traj_com = Float64[]
    traj_root_com = Float64[]

    push!(traj_opt, total_cost(root))
    push!(traj_res, NaN)
    push!(traj_primal_res, NaN)
    push!(traj_dual_res, NaN)
    push!(traj_com, 0.0)
    push!(traj_root_com, 0.0)
    if dict_result !== nothing
        initial_err = Float64[]
        get_err!(root, dict_result, initial_err)
        push!(traj_err, sum(initial_err))
    end

    for iteration in 1:max_iter
        ter = Float64[]
        prev_parent_primes = Dict()
        snapshot_edge_parent_primes!(root, prev_parent_primes)

        hierarchicalADMM!(root, ter; λ = λ)

        push!(traj_opt, total_cost(root))
        push!(traj_res, maximum(ter))
        push!(traj_primal_res, max_primal_residual(root))
        push!(traj_dual_res, max_dual_residual(root, prev_parent_primes))

        total, _ = tt_com_iter(root)
        push!(traj_com, total["com"])
        push!(traj_root_com, root.com_cost)

        if dict_result !== nothing
            err = Float64[]
            get_err!(root, dict_result, err)
            push!(traj_err, sum(err))
        end

        if maximum(ter) < tol
            println("fADMM converged after $iteration iterations in root")
            break
        end
    end

    dict_prime_root = flatten_solution_snapshot(root)

    if return_residuals
        return dict_prime_root, traj_err, traj_res, traj_opt, traj_com, traj_root_com, traj_primal_res, traj_dual_res
    end
    return dict_prime_root, traj_err, traj_res, traj_opt, traj_com, traj_root_com
end
