using LinearAlgebra, Optim, JuMP

function costfunction(x, para::Float64)
    return dot(x.-para, x.-para)^2
end

function costfunction(x::JuMP.Containers.SparseAxisArray, para::Float64, nc::Int64)
    return sum(dot(x[i,:] .- para, x[i,:] .- para) for i in 1:nc)^2
end

mutable struct linknode
    ID::String
    nV::Int64
    prime::Dict{Any, Any}
    dual::Dict{Any, Any}
    children::Union{Vector{linknode}, Nothing}
    parent::Union{linknode, Nothing}
    # costfunction::Function

    function linknode(ID::String)
        obj = new(ID)
        obj.nV = 0
        obj.prime = Dict()
        obj.dual  = Dict()
        obj.children = nothing
        obj.parent = nothing
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