include("linknodes.jl")

# Define cost function and its gradient for each type of node
# Leaf nodes
function cost_func_leaf(x; para)
    dim = length(x)
    return sum(1/min(x[t], para[1]) - 1/para[1] for t in 1:dim)
end

function grad_cost_leaf(x; para)
    dim = length(x)
    grad = zeros(dim)

    for i in 1:dim
        if x[i] >= para[1]
            grad[i] = 0.
        else
            grad[i] = -1/x[i]^2
        end
    end

    return grad
end

## Parent nodes
function cost_func_parent(x; para)
    η  = para[1]
    τ  = para[2]
    nc = para[3]

    horizon = Int64(length(x)/nc)

    return η*sum(max(0., sum([x[(j-1)*horizon + i] for j in 1:nc]) - τ) for i in 1:horizon)
end

function grad_cost_parent(x; para)
    dim = length(x)

    η  = para[1]
    τ  = para[2]
    nc = para[3]

    horizon = Int64(length(x)/nc)

    grad = zeros(dim)
    for i in 1:horizon
        if sum([x[(j-1)*horizon + i] for j in 1:nc]) - τ <= 0
            for j in 1:nc
                grad[(j-1)*horizon + i] = 0.
            end
        else
            for j in 1:nc
                grad[(j-1)*horizon + i] = η
            end
        end
    end

    return grad
end

## Root
function cost_func_root(x; para)
    ϵ = para[1]
    β = para[2]

    nβ = length(β)
    nx = length(x)
    
    nv   = Int64(nx/nβ)

    sx = sum(x[((i-1)*nβ + 1):(i*nβ)] for i in 1:nv)
    return ϵ*dot(sx - β, sx - β)
end


function grad_cost_root(x; para)
    ϵ = para[1]
    β = para[2]

    nβ  = length(β)
    dim = length(x)
    nv  = Int64(dim/nβ)

    sx  = ϵ*sum(x[((i-1)*nβ + 1):(i*nβ)] for i in 1:nv)

    return kron(ones(nv), sx - β)
end

function topo_gen!(node::linknode, node_config::Vector{Vector{String}})
    nL = length(node_config)

    for i in 1:nL
        par   = node_config[i][1]
        ID    = node_config[i][2]
        child = linknode(string(ID))

        parent = linknode[]
        get_node!(node, par, parent)
        set_relative!(parent[1], child)
    end
end

function setup_network!(root::linknode, para::parameter)
    assign!(root, para) #Only for disjoint/sharing problem
    init_var_map(root)
    set_order(root)
end


# for specail casese #variables = #dual = #prime (no local variables)
function assign!(node::linknode, para::parameter)
    if node.children === nothing #leaf
        node.nV = length(para.β)                           #Set dimension of variable
        push!(node.prime, node.ID => 2ones(node.nV))        #Keep its prime variable
        push!(node.parent.prime, node.ID => 2ones(node.nV)) #Initiate a prime variable in its parent
        push!(node.parent.dual,  node.ID => 2ones(node.nV)) #Initiate a dual variable in its parent

        node.cost_func = CostFunc(cost_func_leaf, grad_cost_leaf, (para.a[node.ID], 0.)) #cost + its grad + parameters
    else
        for child in node.children
            assign!(child, para) #go to next layer
            push!(node.prime, child.ID => 2ones(child.nV)) #when next layers are initiated, push a prime variable
            push!(node.dual,  child.ID => 2ones(child.nV)) #when next layers are initiated, push a dual variable
        end
        for (x,y) in node.prime
            node.nV += length(y) #calculate total number of variables
        end
        if node.parent !== nothing #parent
            push!(node.parent.prime, node.ID => 2ones(node.nV)) #initiate a prime variable in its parent
            push!(node.parent.dual,  node.ID => 2ones(node.nV)) #initiate a dual variable in its parent

            node.cost_func = CostFunc(cost_func_parent, grad_cost_parent, (para.η, para.τ[node.ID], length(node.children))) #cost + its grad + parameters
        else #root
            node.cost_func = CostFunc(cost_func_root, grad_cost_root, (para.ϵ, para.β)) #cost + its grad + parameters
        end
    end
end

function init_var_map(node::linknode; dict=var_map)
    dict[node.ID] = Vector{Int64}[]
    if node.children !== nothing
        for child in node.children
            init_var_map(child)
        end
    end
end

function set_order(node::linknode; dict=var_map)
    global tt_vars

    if node.children === nothing
        dict[node.ID] = push!(dict[node.ID], [tt_vars+1, tt_vars+node.nV])
        tt_vars += node.nV
    else
        for child in node.children
            set_order(child)
            for pair in dict[child.ID]
                dict[node.ID] = push!(dict[node.ID], pair)
            end
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

function reset_var!(node::linknode)
    for (x,y) in node.prime
         node.prime[x] = zeros(length(y))  
    end

    for (x,y) in node.dual
        node.dual[x] = zeros(length(y))  
    end
    if node.children !== nothing
        for child in node.children
            reset_var!(child)
        end
    end
end