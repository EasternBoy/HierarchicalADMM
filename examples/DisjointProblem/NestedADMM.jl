function as_scalar_query(query)
    return query isa Vector ? query[1] : query
end

function update_leaf_nested!(node::linknode, top_query; λ = λn)
    para = node.cost_func.para
    func = node.cost_func.val
    w = node.cost_func.w
    target = as_scalar_query(top_query)
    x0 = vect_prime(node)
    n = node.nV

    opti = JuMP.Model(Ipopt.Optimizer)
    set_silent(opti)

    x = @variable(opti, [i = 1:n], start = x0[i])
    slack = @variable(opti, [i = 1:n])

    @constraint(opti, slack .>= x)
    @constraint(opti, slack .>= -x)
    @constraint(opti, local_l <= sum(x))
    @constraint(opti, sum(x) <= local_u)

    @objective(opti, Min, func(x; para = para) + w * sum(slack) + (sum(x) - target)^2 / (2λ))

    JuMP.optimize!(opti)

    node.prime[node.ID] = JuMP.value.(x)
end

function update_internal_nested!(node::linknode, top_query; λ = λn)
    para = node.cost_func.para
    func = node.cost_func.val
    w = node.cost_func.w
    x0 = vect_prime(node)
    n = node.nV

    opti = JuMP.Model(Ipopt.Optimizer)
    set_silent(opti)

    x = @variable(opti, [i = 1:n], start = x0[i])
    slack = @variable(opti, [i = 1:n])

    @constraint(opti, slack .>= x)
    @constraint(opti, slack .>= -x)
    @constraint(opti, local_l <= sum(x))
    @constraint(opti, sum(x) <= local_u)

    penalty = zero(AffExpr)
    if node.parent !== nothing
        target = as_scalar_query(top_query)
        penalty += (sum(x) - target)^2
    end

    for (idx, child) in enumerate(node.children)
        child_target = sum(vect_prime(child)) - node.dual[child.ID][1]
        penalty += (x[idx] - child_target)^2
    end

    @objective(opti, Min, func(x; para = para) + w * sum(slack) + penalty / (2λ))

    JuMP.optimize!(opti)

    for (idx, child) in enumerate(node.children)
        node.prime[child.ID] = [JuMP.value(x[idx])]
    end
end

function record_nested_root_trajectory!(
    node::linknode,
    dict_result,
    traj_err,
    traj_res,
    traj_opt,
    traj_com,
    traj_root_com,
    traj_primal_res,
    traj_dual_res,
    primal_res,
    dual_res,
)
    if dict_result !== nothing && traj_err !== nothing && traj_res !== nothing && traj_opt !== nothing
        push!(traj_opt, total_cost(node))
        push!(traj_res, primal_res)

        if traj_primal_res !== nothing
            push!(traj_primal_res, max_primal_residual(node))
        end
        if traj_dual_res !== nothing
            push!(traj_dual_res, dual_res)
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
end

function nestedADMM!(
    node::linknode,
    top_query = 0.;
    tol = tol,
    max_iter = max_iter,
    dict_result = nothing,
    traj_err = nothing,
    traj_res = nothing,
    traj_opt = nothing,
    traj_com = nothing,
    traj_root_com = nothing,
    traj_primal_res = nothing,
    traj_dual_res = nothing,
)
    if node.children === nothing
        node.iteration += 1
        update_leaf_nested!(node, top_query; λ = λn)
        return
    end

    for iteration in 1:max_iter
        node.iteration += 1

        prev_parent_primes = Dict()
        if node.parent === nothing
            snapshot_edge_parent_primes!(node, prev_parent_primes)
        end

        update_internal_nested!(node, top_query; λ = λn)

        ter = Float64[]
        for child in node.children
            q_to_child = node.prime[child.ID] + node.dual[child.ID]
            com_cost!(node, q_to_child, 1)

            nestedADMM!(child, q_to_child; tol = tol, max_iter = max_iter)

            child_prime = vect_prime(child)
            res = edge_residual(node, child)
            node.dual[child.ID] += res

            com_cost!(child, child_prime, 1)
            push!(ter, maximum(abs.(res)))
        end

        primal_res = max_primal_residual(node)
        dual_res = 0.0
        if node.parent === nothing
            dual_res = max_dual_residual(node, prev_parent_primes)
            record_nested_root_trajectory!(
                node,
                dict_result,
                traj_err,
                traj_res,
                traj_opt,
                traj_com,
                traj_root_com,
                traj_primal_res,
                traj_dual_res,
                primal_res,
                dual_res,
            )
        end

        if primal_res < tol
            if node.parent === nothing
                println("nADMM converged after $iteration iterations in root")
            end
            break
        end
    end
end
