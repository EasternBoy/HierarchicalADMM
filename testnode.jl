mutable struct Node
    name::String
    parent::Union{Node,Nothing}
    children::Vector{Node}
    X::Dict{Node, Vector{Float64}}
    n::Int
end

function Node(name, C::Vector{Node}, N::Int)
    if !isempty(C)
        n = sum(c.n for c in C)
    else
        n = N
    end
    self = Node(name, nothing, C, Dict(c => zeros(c.n) for c in C), n)
    for c in C
        c.parent = self
    end
    self
end

function Node(name, N::Int)
    Node(name, nothing, Node[], Dict(), N)
end

tree =
Node("root", [
    Node("C1", 2),
    Node("C2", [
        Node("L21", 1),
        Node("L22", 2)
    ], 0)
], 0)

N1 = Node("C1", 2)
N21 = Node("L21", 1)
N22 = Node("L22", 2)
N2 = Node("C2", [N21, N22], 0)
root = Node("root", [N1, N2], 0)

isleaf(n::Node) = isempty(n.children)
function printchildren(n::Node)
    if isleaf(n)
        println("This is a leaf node.")
    else
        for c in n.children
            print(c.name, ",")
            println()
        end
    end
end

function printmyparentvector(n::Node)
    if isnothing(n.parent)
        println("I'm the root node.")
    else
        println(n.parent.X[n])
    end
end