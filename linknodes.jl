mutable struct linknode
    ID::Int64
    children
    parent::linknode
    dict::Dict

    function linknode(ID::Int64)
        return  obj = new(ID)
    end
end

function setrelative!(parent::linknode, children::Vector{linknode})
    parent.children = children
    for i in 1:length(children) children[i].parent = parent end
    return nothing
end

child1 = linknode(1)
child1.children = Nothing
child2 = linknode(2)
child2.children = Nothing
root   = linknode(0)

setrelative!(root, [child1, child2])
