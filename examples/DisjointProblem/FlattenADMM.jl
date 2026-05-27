function collect_nodes!(node::linknode, flatten_nodes)
    push!(flatten_nodes, node.ID => node)
    if node.children !== nothing
        for child in node.children
            collect_nodes!(child, flatten_nodes)
        end
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

function get_path(node::linknode)
    path = [node]
    pointer = node

    while pointer.parent !== nothing
        pointer = pointer.parent
        push!(path, pointer)
    end
    return reverse(path)
end

function getPath!(node::linknode, allPaths::Dict)
    allPaths[node.ID] = get_path(node)
    if node.children !== nothing
        for child in node.children
            getPath!(child, allPaths)
        end
    end
end

function write_flat_solution_to_tree!(node::linknode, dict_value::Dict)
    value = dict_value[node.ID]
    if node.children === nothing
        node.prime[node.ID] = value
    else
        for (idx, child) in enumerate(node.children)
            node.prime[child.ID] = [value[idx]]
        end
    end

    if node.children !== nothing
        for child in node.children
            write_flat_solution_to_tree!(child, dict_value)
        end
    end
end

function update_root_flat!(root::linknode, dict_global::Dict, dict_local::Dict, dict_dual::Dict; λ = λf)
    nodes = Dict()
    collect_nodes!(root, nodes)

    opti = JuMP.Model(Ipopt.Optimizer)
    set_silent(opti)

    z = Dict()
    for (id, node) in nodes
        start_value = dict_global[id]
        z[id] = @variable(opti, [i = 1:node.nV], start = start_value[i], base_name = string("z", id))
        @constraint(opti, local_l <= sum(z[id]))
        @constraint(opti, sum(z[id]) <= local_u)
    end

    for (_, node) in nodes
        if node.children !== nothing
            for (idx, child) in enumerate(node.children)
                @constraint(opti, z[node.ID][idx] == sum(z[child.ID]))
            end
        end
    end

    @objective(
        opti,
        Min,
        sum(
            sum((z[id][i] - (dict_local[id][i] - dict_dual[id][i]))^2 for i in 1:nodes[id].nV)
            for id in keys(nodes)
        ) / (2λ)
    )

    JuMP.optimize!(opti)

    for id in keys(nodes)
        dict_global[id] = JuMP.value.(z[id])
    end
end

function update_node_flat(node::linknode, query::Vector{Float64}, λ::Float64)
    para = node.cost_func.para
    func = node.cost_func.val
    w = node.cost_func.w
    x0 = vect_prime(node)
    n = length(query)

    opti = JuMP.Model(Ipopt.Optimizer)
    set_silent(opti)

    x = @variable(opti, [i = 1:n], start = x0[i])
    slack = @variable(opti, [i = 1:n])

    @constraint(opti, slack .>= x)
    @constraint(opti, slack .>= -x)
    @constraint(opti, local_l <= sum(x))
    @constraint(opti, sum(x) <= local_u)

    @objective(opti, Min, func(x; para = para) + w * sum(slack) + sum((x[i] - query[i])^2 for i in 1:n) / (2λ))

    JuMP.optimize!(opti)

    return JuMP.value.(x)
end

function max_flat_primal_residual(dict_global::Dict, dict_local::Dict)
    if isempty(dict_global)
        return 0.0
    end
    return maximum(maximum(abs.(dict_global[id] - dict_local[id])) for id in keys(dict_global))
end

function max_flat_dual_residual(dict_global::Dict, prev_global::Dict)
    if isempty(dict_global)
        return 0.0
    end
    return maximum(maximum(abs.(dict_global[id] - prev_global[id])) for id in keys(dict_global))
end

function flattenADMM(root::linknode; tol = tol, λ = λf, max_iter = max_iter, dict_result = nothing, return_residuals = false)
    nodes = Dict()
    collect_nodes!(root, nodes)

    paths = Dict()
    getPath!(root, paths)

    dict_global = Dict()
    dict_local = Dict()
    dict_dual = Dict()
    for (id, node) in nodes
        initial_value = vect_prime(node)
        dict_global[id] = copy(initial_value)
        dict_local[id] = copy(initial_value)
        dict_dual[id] = zeros(node.nV)
    end

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
        root.iteration += 1

        prev_global = Dict(id => copy(value) for (id, value) in dict_global)
        update_root_flat!(root, dict_global, dict_local, dict_dual; λ = λ)

        for (id, node) in nodes
            node.iteration += node === root ? 0 : 1

            query = dict_global[id] + dict_dual[id]
            if node !== root
                com_cost!(paths[id], query, 1)
            end

            dict_local[id] = update_node_flat(node, query, λ)

            if node !== root
                com_cost!(reverse(paths[id]), dict_local[id], 1)
            end
        end

        for id in keys(nodes)
            dict_dual[id] += dict_global[id] - dict_local[id]
        end

        write_flat_solution_to_tree!(root, dict_local)

        split_res = max_flat_primal_residual(dict_global, dict_local)
        dual_res = max_flat_dual_residual(dict_global, prev_global)
        aggregation_res = max_primal_residual(root)
        push!(traj_res, aggregation_res)
        push!(traj_primal_res, aggregation_res)
        push!(traj_dual_res, dual_res)
        push!(traj_opt, total_cost(root))

        total, _ = tt_com_iter(root)
        push!(traj_com, total["com"])
        push!(traj_root_com, root.com_cost)

        if dict_result !== nothing
            err = Float64[]
            get_err!(root, dict_result, err)
            push!(traj_err, sum(err))
        end

        if max(split_res, aggregation_res) < tol
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
