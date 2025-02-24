using LinearAlgebra, Optim, JuMP, JLD2

function costfunction(x::Vector{VariableRef}, para::Float64)
    return (x.-para)'*(x.-para)
end

mutable struct linknode
    ID::String
    nV::Int64
    prime::Dict{Any, Any}
    dual::Dict{Any, Any}
    children::Union{Vector{linknode}, Nothing}
    parent::Union{linknode, Nothing}
    costfunction::Function

    function linknode(ID::String)
        obj = new(ID)
        obj.nV = 0
        obj.prime = Dict()
        obj.dual  = Dict()
        obj.children = nothing
        obj.parent = nothing
        obj.costfunction = costfunction
        return obj
    end
end

function set_relative!(parent::linknode, children::Vector{linknode})
    parent.children = children
    for i in 1:length(children) 
        children[i].parent = parent 
    end
end