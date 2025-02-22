mutable struct linknode
    ID::String
    children::Union{Vector{linknode}, Nothing}
    parent::Union{linknode, Nothing}
    dict::Dict{String, Vector{linknode}}

    function linknode(ID::String)
        return  obj = new(ID)
    end
end

function setrelative!(parent::linknode, children::Vector{linknode})
    parent.children = children
    for i in 1:length(children) children[i].parent = parent end
    return nothing
end

child1 = linknode("1")
child2 = linknode("2")
root   = linknode("0")

setrelative!(root, [child1, child2])
