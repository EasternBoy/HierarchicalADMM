function nestedADMM!(node::linknode, top_query = 0.; tol = tol, max_iter = max_iter, dict_result = nothing, traj_err = nothing, traj_res = nothing, traj_opt = nothing, traj_com = nothing, traj_root_com = nothing, traj_primal_res = nothing, traj_dual_res = nothing)
    if node.parent !== nothing
        com_cost!(node.parent, [sum(vect_prime(node))], 1)
    end

    if node.parent !== nothing
        proxAlg!(node; λ = λn)
        return
    end

    for iteration in 1:max_iter
        ter = Float64[]
        prev_parent_primes = Dict()
        if traj_primal_res !== nothing && traj_dual_res !== nothing
            snapshot_edge_parent_primes!(node, prev_parent_primes)
        end

        hierarchicalADMM!(node, ter; λ = λn)

        if dict_result !== nothing && traj_err !== nothing && traj_res !== nothing && traj_opt !== nothing
            push!(traj_opt, total_cost(node))
            push!(traj_res, maximum(ter))
            if traj_primal_res !== nothing
                push!(traj_primal_res, max_primal_residual(node))
            end
            if traj_dual_res !== nothing
                push!(traj_dual_res, max_dual_residual(node, prev_parent_primes))
            end

            err = Float64[]
            get_err!(node, dict_result, err)
            push!(traj_err, sum(err))

            if traj_com !== nothing
                total, _ = tt_com_iter(node)
                push!(traj_com, total["com"])
            end
            if traj_root_com !== nothing
                push!(traj_root_com, node.com_cost)
            end
        end

        if maximum(ter) < tol
            println("nADMM converged after $iteration iterations in root")
            break
        end
    end
end
