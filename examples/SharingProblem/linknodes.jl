using LinearAlgebra, JuMP, Optim
using Zygote
using ProximalAlgorithms
using ProximalOperators
using DifferentiationInterface: AutoZygote


mutable struct CostFunc
    val::Function
    grad::Function
    para
    λ::Float64
    q::Vector{Float64}
    w::Float64

    
    function CostFunc(val, grad, para)
        return new(val, grad, para, 0., [0.], 2.)
    end
end


struct parameter
    τ; a; β; ϵ; η
end


mutable struct linknode
    ID::String
    nV::Int64
    prime::Dict{Any, Any}
    dual::Dict{Any, Any}
    children::Union{Vector{linknode}, Nothing}
    parent::Union{linknode, Nothing}
    cost_func::CostFunc
    com_cost::Int64
    iteration::Int64
    solver

    function linknode(ID::String)
        obj = new(ID)
        obj.nV = 0
        obj.prime = Dict()
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

function _collect_congestion!(node::linknode, depth::Int, levels::Vector{Int}, depths::Vector{Int})
    if node.children === nothing
        return
    end

    if depth > 0 && length(node.children) > 1
        push!(levels, length(node.children) - 1)
        push!(depths, depth)
    end

    for child in node.children
        _collect_congestion!(child, depth + 1, levels, depths)
    end
end

function congestion_stats(root::linknode)
    levels = Int[]
    depths = Int[]
    _collect_congestion!(root, 0, levels, depths)
    return length(levels), levels, depths
end

#Vectorize prime dict of a node in the same order as its children
function vect_prime(node::linknode)
    result = Float64[]
    if node.children === nothing #leaf: only element in its prime dict
        result = node.prime[node.ID]
    else #middle nodes: 
        for child in node.children
            result = [result; node.prime[child.ID]]
        end
    end
    return result
end

function vect_dual(node::linknode)
    result = Float64[]
    for child in node.children
        result = [result; node.dual[child.ID]]
    end
    return result
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

function add_edge_graph!(parent::linknode, g)
    if parent.children !== nothing
        for child in parent.children
            pID = parse(Int64, parent.ID) + 1
            cID = parse(Int64, child.ID) + 1
            add_edge!(g, pID, cID)
        end

        for child in parent.children
            add_edge_graph!(child, g)
        end
    end
end


function com_cost!(node::linknode, transmitted_data::Union{Float64, Vector{Float64}}, nChannel::Int64)
    if typeof(transmitted_data) == Float64
        node.com_cost += nChannel
    else
        node.com_cost += length(transmitted_data)*nChannel
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


function reset!(node::linknode)
    for (x,y) in node.prime
         node.prime[x] = 2ones(length(y))  
    end

    for (x,y) in node.dual
        node.dual[x] = 2ones(length(y))  
    end

    node.com_cost = 0
    node.iteration = 0

    if node.children !== nothing
        for child in node.children
            reset!(child)
        end
    end
end
