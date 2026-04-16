include("../../src/linknodes.jl")
using Random: shuffle!

# Define cost function and its gradient for each type of node
## Leaf nodes
function cost_func_leaf(x; para)
    return 1/2*dot(x .- para[1], x .- para[1])^2
end

function grad_cost_leaf(x; para)
    return 2*(x .- para[1])*dot(x .- para[1], x .- para[1])
end

## Parent nodes
function cost_func_parent(x; para)
    return 1/2*dot(x .- para[1], x .- para[1])^2
end

function grad_cost_parent(x; para)
    return 2*(x .- para[1])*dot(x .- para[1], x .- para[1])
end

## Root
function cost_func_root(x; para)
    return 1/2*dot(x .- para[1], x .- para[1])^2
end

function grad_cost_root(x; para)
    return 2*(x .- para[1])*dot(x .- para[1], x .- para[1])
end


function choose_num_children(nN:: Int, nL::Int, mode::Symbol, k::Int)
    if nN <= 0
        return 0
    elseif mode == :balanced && nL <= 1
        return nN
    end

    if mode == :balanced 
        return min(k, nN)
    elseif mode == :unbalanced
        if nN == 1
            return 1
        end
        return rand(2:min(k, nN))
    else
        error("Invalid mode. Please choose either :balanced or :unbalanced.")
    end
end

function allocate_subtree_sizes(n_remain::Int, num_child::Int, mode::Symbol, k::Int)
    alloc = zeros(Int, num_child)

    if n_remain <= 0 || num_child == 0
        return alloc
    end

    if mode == :balanced
        base = div(n_remain, num_child)
        extra = rem(n_remain, num_child)

        for i in 1:num_child
            alloc[i] = base
        end
        for i in 1:extra
            alloc[i] += 1
        end

    elseif mode == :unbalanced
        deep_idx = rand(1:num_child)

        max_side = min(num_child - 1, n_remain ÷ k)
        side_budget = max_side > 0 ? rand(0:max_side) : 0

        side_candidates = [i for i in 1:num_child if i != deep_idx]
        shuffle!(side_candidates)

        for i in 1:side_budget
            alloc[side_candidates[i]] = 1
        end

        alloc[deep_idx] = n_remain - side_budget
    else
        error("Unknown mode: $mode")
    end

    return alloc
end

function topo_gen!(node::linknode, nN::Int, nL::Int; mode::Symbol = :balanced, k::Int = 2, depth::Int = 1)
    global countID

    if mode == :balanced
        if nN <= 0 || nL <= 0
            return
        end
    elseif mode == :unbalanced
        if nN <= 0
            return
        end
    else
        error("Unknown mode: $mode")
    end

    num_child = choose_num_children(nN, nL, mode, k)
    children = [linknode(string(countID += 1)) for _ in 1:num_child]
    set_relative!(node, children)

    n_remain = nN - num_child
    alloc = allocate_subtree_sizes(n_remain, num_child, mode, k)

    for i in 1:num_child
        if alloc[i] > 0
            if mode == :balanced
                topo_gen!(children[i], alloc[i], nL - 1; mode = mode, k = k, depth = depth + 1)
            else
                topo_gen!(children[i], alloc[i], nL; mode = mode, k = k, depth = depth + 1)
            end
        end
    end
end

# for specail casese #variables = #dual = #prime (no local variables)
function assign!(node::linknode)
    node.com_cost = 0
    node.iteration = 0
    if node.children === nothing #leaf
        node.nV = 1                       #Set dimension of variable
        push!(node.prime, node.ID => ones(node.nV))        #Keep its prime variable
        push!(node.parent.prime, node.ID => ones(node.nV)) #Initiate a prime variable in its parent
        push!(node.parent.dual,  node.ID => ones(node.nV)) #Initiate a dual variable in its parent

        node.cost_func = CostFunc(cost_func_leaf, grad_cost_leaf, (parse(Float64, node.ID))) #cost + its grad + parameters
    else
        for child in node.children
            assign!(child) #go to next layer
            push!(node.prime, child.ID => ones(child.nV)) #when next layers are initiated, push a prime variable
            push!(node.dual,  child.ID => ones(child.nV)) #when next layers are initiated, push a dual variable
        end
        for (x,y) in node.prime
            node.nV += length(y) #calculate total number of variables
        end
        if node.parent !== nothing #parent
            push!(node.parent.prime, node.ID => ones(node.nV)) #initiate a prime variable in its parent
            push!(node.parent.dual,  node.ID => ones(node.nV)) #initiate a dual variable in its parent

            node.cost_func = CostFunc(cost_func_parent, grad_cost_parent, (parse(Float64, node.ID))) #cost + its grad + parameters
        else #root
            node.cost_func = CostFunc(cost_func_root, grad_cost_root, (parse(Float64, node.ID)))     #cost + its grad + parameters
        end
    end
end


function print_tree(node::linknode, depth=0)
    print("  "^depth * "Node $(node.ID): ")  # Indent based on depth
    print_dict(node)
    println()
    if node.children !== nothing
        for child in node.children
            print_tree(child, depth + 1)
        end
    end
end

function print_dict(node::linknode)
    if node.children !== nothing
        for (x,y) in node.prime
            r = round.(y, digits = 3)
            print(x*"->$r, ")
        end
    else
        r = round.(node.prime[node.ID], digits = 3)
        print(r)
    end
end

function reset!(node::linknode)
    for (x,y) in node.prime
         node.prime[x] = ones(length(y))  
    end

    for (x,y) in node.dual
        node.dual[x] = ones(length(y))  
    end

    node.com_cost = 0
    node.iteration = 0

    if node.children !== nothing
        for child in node.children
            reset!(child)
        end
    end
end
