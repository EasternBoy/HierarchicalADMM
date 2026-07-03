function collect_nodes!(node::linknode, flatten_nodes)
    if node.parent !== nothing
        push!(flatten_nodes, node.ID => node)
    end

    if node.children !== nothing
        for child in node.children
            collect_nodes!(child, flatten_nodes)
        end
    end
end


function update_root_flat!(root::linknode, dict_prime_root::Dict, dict_prime_child::Dict, dict_dual::Dict, pos::Dict, nV::Dict; λ = λₛ)
    para = root.cost_func.para
    func = root.cost_func.val

    query = Dict()
    for (key, value) in dict_prime_child   
        query[key] = value - dict_dual[key] 
    end

    index = Dict()
    for (key, value) in pos 
        index[key] = value:(value + nV[key] - 1)
    end

    f = ProximalAlgorithms.AutoDifferentiable(
        v -> func(v; para = para) + (1/(2λ))*sum(dot(v[index[key]] - query[key], v[index[key]] - query[key]) for (key, value) in query),
        AutoZygote()
    )

    g = ProximalOperators.NormL1(root.cost_func.w)

    x0 = ones(root.nV)

    solution, _ = root.solver(f = f, g = g, x0 = x0)

    for (key, _) in dict_prime_root
        dict_prime_root[key] = solution[index[key]]
    end
end


function update_leaf_flat(node::linknode, query::Vector{Float64}, λ = λₛ)
    para = node.cost_func.para
    func = node.cost_func.val

    f = ProximalAlgorithms.AutoDifferentiable(
        v -> func(v; para = para) + (1/(2λ))*dot(v - query, v - query),
        AutoZygote()
    )

    g = ProximalOperators.NormL1(node.cost_func.w)

    x0 = ones(node.nV)
    solution, iterations = node.solver(f = f, g = g, x0 = x0)

    return solution
end

function get_path(node::linknode)
    path    = [node]
    pointer = node

    while pointer.parent !== nothing #path to root
        pointer = pointer.parent
        push!(path, pointer)
    end
    return reverse(path)
end

function getPath!(node::linknode, allPaths::Dict)
    if node.parent !== nothing
        path = get_path(node)
        allPaths[node.ID] = path
    end

    if node.children !== nothing
        for child in node.children
            getPath!(child, allPaths)
        end
    end
end

function get_postion(path::Vector{linknode})

    path_length = length(path)
    start_index = 1

    # println(path_length)

    for i in 1:path_length-1
        # println("i = $i, node $(path[i].ID) to node $(path[i+1].ID)")
        for child in path[i].children
            if child !== path[i+1]
                start_index += child.nV
            else
                break
            end
        end
    end

    return start_index
end



function flattenADMM(root::linknode; tol = tol, λ = λₛ, max_iter = max_iter)

    # Flatten the tree structure into a list of nodes
    dict_prime_child = Dict()
    dict_prime_root  = Dict()
    dict_dual        = Dict()
    
    flatten_tree = Dict()
    collect_nodes!(root, flatten_tree)

    path = Dict()
    getPath!(root, path)

    pos = Dict()
    nV  = Dict()

    for (key, value) in flatten_tree
        dict_prime_root[key]  = ones(value.nV)
        dict_prime_child[key] = ones(value.nV)
        dict_dual[key]        = ones(value.nV)
        pos[key]              = get_postion(path[key])
        nV[key]               = value.nV
    end


    dict_child_root_old = deepcopy(dict_prime_child)

    # Initialize the query vector
    for iteration in 1:max_iter

        root.iteration += 1 #Add 1 iteration in root

        update_root_flat!(root, dict_prime_root, dict_prime_child, dict_dual, pos, nV; λ = λ)

        ter = Float64[]
        # Iterate through the nodes to solve the optimization problem
        for (key, value) in flatten_tree
            value.iteration += 1 #Add 1 iteration in a child

            query = dict_prime_root[key] + dict_dual[key]

            com_cost!(path[key], query, 1) #Communication from root to child
            dict_prime_child[key] = update_leaf_flat(value, query, λ)

            prime_res = dict_prime_root[key]   - dict_prime_child[key]
            dual_res  = (dict_prime_child[key] - dict_child_root_old[key])/λ
            dict_dual[key] += prime_res

            # Check stopping criteria based on the residuals
            if prime_stop
                push!(ter, norm(prime_res, Inf))
            else
                push!(ter, max(norm(prime_res, Inf), norm(dual_res, Inf)))
            end

            com_cost!(reverse(path[key]), dict_prime_child[key], 1) #Communication from child to root to compute dual variable located in root
        end

        dict_child_root_old = copy(dict_prime_child)

        if maximum(ter) < tol
            println("fADMM converged after $iteration iterations in root")
            break
        end
    end

    assign_node!(root, dict_prime_child)

    return dict_prime_root
end


function assign_node!(node, dict_prime)
    if node.children !==  nothing
        for child in node.children
            node.prime[child.ID] = copy(dict_prime[child.ID])
            assign_node!(child, dict_prime)
        end
    else
        node.prime[node.ID] = copy(dict_prime[node.ID])
    end
end
