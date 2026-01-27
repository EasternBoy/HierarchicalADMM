using LinearAlgebra: Diagonal, eigen, Symmetric

mutable struct CostFunc
    val::Function
    grad::Function
    para
    λ::Float64
    q::Vector{Float64}
    w::Float64
    
    function CostFunc(val, grad, para)
        return new(val, grad, para, 0.1, [0.], 2.)
    end
end


mutable struct linknode
    ID::String
    nV::Int64
    prime::Vector{Float64}
    dual::Dict{Any, Any}
    children::Union{Vector{linknode}, Nothing}
    parent::Union{linknode, Nothing}
    cost_func::CostFunc
    com_cost::Int64
    iteration::Int64
    solver

    function linknode(ID::String)
        obj = new(ID)
        obj.nV = 1
        obj.prime = [1.]
        obj.dual  = Dict()
        obj.children  = nothing
        obj.parent    = nothing
        obj.com_cost  = 0
        obj.iteration = 0
        obj.solver    = ProximalAlgorithms.PANOC(maxit = 1000, tol = 1e-8, verbose = false)
        return obj
    end
end

function set_relative!(parent::linknode, child::linknode)
    if parent.children === nothing
        parent.children = linknode[]
    end
    push!(parent.children, child)
    child.parent = parent 
end

function set_relative!(parent::linknode, children::Vector{linknode})
    parent.children = children
    for i in 1:length(children) 
        children[i].parent = parent 
    end
end

function _collect_congestion!(node::linknode, depth::Int, ids::Vector{String}, levels::Vector{Int}, depths::Vector{Int})
    if node.children === nothing
        return
    end

    if depth > 0 && length(node.children) > 1
        push!(ids, node.ID)
        push!(levels, length(node.children) - 1)
        push!(depths, depth)
    end

    for child in node.children
        _collect_congestion!(child, depth + 1, ids, levels, depths)
    end
end

function congestion_stats(root::linknode)
    ids = String[]
    levels = Int[]
    depths = Int[]
    _collect_congestion!(root, 0, ids, levels, depths)
    return ids, levels, depths
end

function _collect_all_nodes!(node::linknode, nodes::Vector{linknode})
    push!(nodes, node)
    if node.children !== nothing
        for child in node.children
            _collect_all_nodes!(child, nodes)
        end
    end
end

function adjacency_matrix(root::linknode; normalized::Bool = false)
    nodes = linknode[]
    _collect_all_nodes!(root, nodes)

    id_to_idx = Dict{String, Int}()
    for (idx, node) in enumerate(nodes)
        id_to_idx[node.ID] = idx
    end

    n = length(nodes)
    adj = zeros(Float64, n, n)
    for node in nodes
        i = id_to_idx[node.ID]
        if node.children !== nothing
            for child in node.children
                j = id_to_idx[child.ID]
                adj[i, j] = 1.0
                adj[j, i] = 1.0
            end
        end
    end

    if !normalized
        return adj, id_to_idx
    end

    degrees = sum(adj, dims = 2)
    if any(degrees .== 0.0)
        return adj, id_to_idx
    end

    inv_sqrt_deg = Diagonal(vec(1 ./ sqrt.(degrees)))
    normalized_adj = Symmetric(inv_sqrt_deg * adj * inv_sqrt_deg)

    return Matrix(normalized_adj), id_to_idx
end

function second_largest_adjacency_eigenvalue(root::linknode; normalized::Bool = true)
    adj, _ = adjacency_matrix(root; normalized = normalized)
    n = size(adj, 1)
    if n < 2
        return 0.0
    end
    vals = eigen(Symmetric(adj)).values
    sorted_vals = sort(vals, rev = true)
    return sorted_vals[2]
end

function access_nodeID!(node::linknode, ID::String, result::Vector{linknode})
    if node.ID == ID
        push!(result, node)
    else
        if node.children !== nothing
            for child in node.children
                access_nodeID!(child, ID, result)
            end
        end
    end
end

function add_edge_graph!(parent::linknode, g)
    if parent.children !== nothing
        for child in parent.children
            pID = parse(Int64, parent.ID)
            cID = parse(Int64, child.ID)
            add_edge!(g, pID, cID)
        end

        for child in parent.children
            add_edge_graph!(child, g)
        end
    end
end

function vect_dual(node::linknode)
    result = Float64[]
    for child in node.children
        result = [result; node.dual[child.ID]]
    end
    return result
end

function vect_prime(node::linknode)
    result = Float64[]
    if node.children === nothing #leaf: only element in its prime dict
        result = node.prime[node.ID]
    else #middle nodes: 
        for child in node.children
            append!(result, node.prime[child.ID])
        end
    end
    return result
end

function vect_child(node::linknode)
    result = Float64[]
    
    for child in node.children
        result = [result; vect_prime(child)]
    end

    return result
end

function com_cost!(node::linknode, transmitted_data::Union{Float64, Vector{Float64}}, nChannel::Int64)
    if typeof(transmitted_data) == Float64
        node.com_cost += nChannel
    else
        node.com_cost += length(transmitted_data)*nChannel
    end
end

function com_cost!(path::Vector{linknode}, transmitted_data::Union{Float64, Vector{Float64}}, nChannel::Int64)
    n = length(path)-1
    if typeof(transmitted_data) == Float64
        for i in 1:n #Root (end of the path) only received
            path[i].com_cost += nChannel
        end
    else
        for i in 1:n
            path[i].com_cost += length(transmitted_data)*nChannel
        end
    end
end


function tt_com_iter(root::linknode)
    total   = Dict("com" => 0)
    max_num = Dict("com" => root.com_cost, "iter" => root.iteration)
    
    com_iter!(root, total, max_num)

    return total, max_num
end

function com_iter!(node::linknode, total::Dict, max_num::Dict)
    total["com"]  += node.com_cost

    if node.com_cost > max_num["com"]
        max_num["com"] = node.com_cost
    end

    if node.iteration > max_num["iter"]
        max_num["iter"] = node.iteration
    end

    if node.children !== nothing
        for child in node.children
            com_iter!(child, total, max_num)
        end
    end
end

function get_node!(node::linknode, ID::String, result::Vector{linknode})

    if node.ID == ID
        push!(result, node)
    else
        if node.children !== nothing
            for child in node.children
                get_node!(child, ID, result)
            end
        end
    end
end