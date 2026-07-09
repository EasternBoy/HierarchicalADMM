function collect_nodes!(node::linknode, flatten_nodes::Dict)
    if node.parent !== nothing
        push!(flatten_nodes, node.ID => node)
    end

    if node.children !== nothing
        for child in node.children
            collect_nodes!(child, flatten_nodes)
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
    if node.parent !== nothing
        allPaths[node.ID] = get_path(node)
    end

    if node.children !== nothing
        for child in node.children
            getPath!(child, allPaths)
        end
    end
end


function get_position(path::Vector{linknode})
    start_index = 1

    for i in 1:length(path)-1
        for child in path[i].children
            if child !== path[i + 1]
                start_index += child.nV
            else
                break
            end
        end
    end

    return start_index
end


function update_root_flat!(root::linknode, dict_prime_root::Dict, dict_prime_child::Dict,
                           dict_dual::Dict, index::Dict, query::Dict, x0::Vector{Float64};
                           λ = λₛ)
    opti = JuMP.Model(Ipopt.Optimizer)
    set_silent(opti)

    @variable(opti, x[i = 1:root.nV], base_name = "x_root", start = x0[i])
    @constraint(opti, x .>= 1e-5)

    for (key, value) in dict_prime_child
        query[key] = value - dict_dual[key]
    end

    func = root.cost_func.val
    para = root.cost_func.para
    penalty = sum(dot(x[index[key]] - query[key], x[index[key]] - query[key]) for key in keys(query))

    @objective(opti, Min, func(x; para = para) + 1 / (2λ) * penalty)
    JuMP.optimize!(opti)

    solution = JuMP.value.(x)
    copyto!(x0, solution)

    for key in keys(dict_prime_root)
        dict_prime_root[key] = solution[index[key]]
    end
end


function update_node_flat!(node::linknode, query::Vector{Float64}, x0::Vector{Float64}; λ = λₛ)
    opti = JuMP.Model(Ipopt.Optimizer)
    set_silent(opti)

    @variable(opti, x[i = 1:node.nV], base_name = string("x", node.ID), start = x0[i])
    @constraint(opti, x .>= 1e-5)

    para = node.cost_func.para
    J = 1 / (2λ) * dot(x - query, x - query)

    if node.children === nothing
        @variable(opti, sl[i = 1:node.nV] .>= 0, base_name = string("sl", node.ID))
        @constraint(opti, sl .>= 1 ./ x .- 1 / para[1])
        J += sum(sl)
    else
        η  = para[1]
        τ  = para[2]
        nc = para[3]
        horizon = Int64(node.nV / nc)

        @variable(opti, t[i = 1:horizon] .>= 0, base_name = string("t", node.ID))
        for i in 1:horizon
            @constraint(opti, t[i] >= sum([x[(j - 1) * horizon + i] for j in 1:nc]) - τ)
        end
        J += η * sum(t)
    end

    @objective(opti, Min, J)
    JuMP.optimize!(opti)

    solution = JuMP.value.(x)
    copyto!(x0, solution)

    return solution
end


function assign_node!(node::linknode, dict_prime::Dict)
    if node.children !== nothing
        for child in node.children
            node.prime[child.ID] = copy(dict_prime[child.ID])
            assign_node!(child, dict_prime)
        end
    else
        node.prime[node.ID] = copy(dict_prime[node.ID])
    end
end


function flattenADMM(root::linknode; tol = tol, λ = λₛ, max_iter = max_iter)
    dict_prime_child = Dict()
    dict_prime_root  = Dict()
    dict_dual        = Dict()
    traj_J_fADMM     = Float64[]
    traj_res_fADMM   = Float64[]

    flatten_tree = Dict()
    collect_nodes!(root, flatten_tree)

    path = Dict()
    getPath!(root, path)

    index = Dict()
    local_x0 = Dict()

    for (key, value) in flatten_tree
        dict_prime_root[key]  = 2ones(value.nV)
        dict_prime_child[key] = 2ones(value.nV)
        dict_dual[key]        = 2ones(value.nV)
        pos                  = get_position(path[key])
        index[key]           = pos:(pos + value.nV - 1)
        local_x0[key]        = copy(dict_prime_child[key])
    end

    query_root = Dict(key => similar(value) for (key, value) in dict_prime_child)
    reverse_path = Dict(key => reverse(value) for (key, value) in path)
    root_x0 = 2ones(root.nV)
    dict_child_root_old = deepcopy(dict_prime_child)

    for iteration in 1:max_iter
        root.iteration += 1

        update_root_flat!(root, dict_prime_root, dict_prime_child, dict_dual,
                          index, query_root, root_x0; λ = λ)

        max_residual = 0.0
        for (key, value) in flatten_tree
            value.iteration += 1

            query = dict_prime_root[key] + dict_dual[key]
            com_cost!(path[key], query, 1)

            dict_prime_child[key] = update_node_flat!(value, query, local_x0[key]; λ = λ)

            prime_res = dict_prime_root[key] - dict_prime_child[key]
            dual_res  = (dict_prime_child[key] - dict_child_root_old[key]) / λ
            dict_dual[key] += prime_res
            max_residual = max(max_residual, norm(prime_res, Inf), norm(dual_res, Inf))

            com_cost!(reverse_path[key], dict_prime_child[key], 1)
        end

        dict_child_root_old = deepcopy(dict_prime_child)
        assign_node!(root, dict_prime_child)
        push!(traj_J_fADMM, total_cost(vect_prime(root), root))
        push!(traj_res_fADMM, max_residual)

        if max_residual < tol
            println("fADMM converged after $iteration iterations in root")
            return traj_J_fADMM, traj_res_fADMM
        end
    end

    return traj_J_fADMM, traj_res_fADMM
end
