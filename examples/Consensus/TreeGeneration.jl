include("../../src/linknodes.jl")

# Define cost function and its gradient for each type of node
## Leaf nodes
function cost_func_leaf(x; para)
    return 1/2*dot(x .- para[1], x .- para[1])
end

function grad_cost_leaf(x; para)
    return x .- para[1]
end

## Parent nodes
function cost_func_parent(x; para)
    return 1/2*dot(x .- para[1], x .- para[1])
end

function grad_cost_parent(x; para)
    return x .- para[1]
end

## Root
function cost_func_root(x; para)
    return 1/2*dot(x .- para[1], x .- para[1])
end

function grad_cost_root(x; para)
    return x .- para[1]
end

function topo_gen!(node::linknode, nN::Int64, nL::Int64, depth=1)
    global countID

    if nL == 1
        num_child = nN
    else
        # nc = Int(round(nN/nL))
        nc = nN - nL
        num_child = rand(1:max(nc,1))
    end

    children = [linknode(string(countID+=1)) for i in 1:num_child]
    set_relative!(node, children)

    res_alloc = nN - num_child

    arr_alloc = Vector{Int64}(zeros(num_child))

    if res_alloc > 0
        d = rand(1:num_child)
        arr_alloc[d] = rand(min(nL-1,res_alloc):max(nL-1, res_alloc)) #make sure reaching the deepest level
        res_alloc -= arr_alloc[d]

        for i in 1:num_child
            if i !== d      
                arr_alloc[i] = rand(1:max(1,res_alloc))
                res_alloc   -= arr_alloc[i]
            end

            if res_alloc <= 0
                break
            end
        end

        if res_alloc > 0
            arr_alloc[rand(1:end)] += res_alloc
        end

        for i in 1:num_child
            if nL - 1 > 0 && arr_alloc[i] > 0
                topo_gen!(children[i], arr_alloc[i], nL-1, depth+1)
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