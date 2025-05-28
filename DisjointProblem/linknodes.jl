using LinearAlgebra, Optim, JuMP
using Zygote
using ProximalOperators
using ProximalAlgorithms
using ProximalCore
using DifferentiationInterface: AutoZygote

function cost_func(x, p)
    return dot(x .- p, x .- p)^2
end

mutable struct linknode
    ID::String
    nV::Int64
    prime::Dict{Any, Any}
    dual::Dict{Any, Any}
    children::Union{Vector{linknode}, Nothing}
    parent::Union{linknode, Nothing}
    prox
    cost_fnc::Function

    function linknode(ID::String)
        obj = new(ID)
        obj.nV = 0
        obj.prime = Dict()
        obj.dual  = Dict()
        obj.children = nothing
        obj.parent   = nothing
        obj.prox     = ProximalAlgorithms.PANOC(maxit = 500, tol = 1e-6, verbose = false)
        return obj
    end
end

function set_relative!(parent::linknode, children::Vector{linknode})
    parent.children = children
    for i in 1:length(children) 
        children[i].parent = parent 
    end
end

function get_node(node::linknode, ID::String)
    if node.ID == ID
        return node
    else
        if node.children !== nothing
            for child in node.children
                get_node(node, ID)
            end
        else
            throw(ErrorException("your ID does not exist!!!"))
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