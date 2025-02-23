mutable struct linknode
    ID::String
    nV::Int64
    dict::Dict{Any, Any}
    children::Union{Vector{linknode}, Nothing}
    parent::Union{linknode, Nothing}

    function linknode(ID::String)
        obj = new(ID)
        obj.nV = 0
        obj.dict = Dict()
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