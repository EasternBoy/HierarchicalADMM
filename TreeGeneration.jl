include("linknodes.jl")

global countID = -1
function topo_gen!(node::linknode, nN::Int64, nL::Int64, depth=1)
    global countID

    if nN - nL >= 0
        num_child = rand(1:nN-nL+1)
    else
        num_child = rand(1:nN)
    end

    if nL == 1
        num_child = nN
    end

    children = [linknode(string(countID+=1)) for i in 1:num_child]
    set_relative!(node, children)

    res_alloc = nN - num_child

    arr_alloc = Vector{Int64}(zeros(num_child))

    if res_alloc > 0
        for i in 1:num_child
            if i == 1       
                arr_alloc[i] = rand(min(nL-1,res_alloc):res_alloc) #make sure
            else
                arr_alloc[i] = rand(1:res_alloc)
            end

            res_alloc -= arr_alloc[i]
            if res_alloc <= 0
                break
            end
        end

        if res_alloc >= 1
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
function assign_var(node::linknode)
    if node.children === nothing #leaf
        node.nV = 1
        push!(node.prime, node.ID => zeros(node.nV))        #keep its prime variable
        push!(node.parent.prime, node.ID => zeros(node.nV)) #initiate a prime variable in its parent
        push!(node.parent.dual,  node.ID => zeros(node.nV)) #initiate a dual variable in its parent
    else
        for child in node.children
            assign_var(child) #go to next layer
            push!(node.prime, child.ID => zeros(child.nV)) #when next layers are initiated, push a prime variable
            push!(node.dual,  child.ID => zeros(child.nV)) #when next layers are initiated, push a dual variable
        end
        for (x,y) in node.prime
            node.nV += length(y) #calculate total number of variables
        end
        if node.parent !== nothing #not leaf node
            push!(node.parent.prime, node.ID => zeros(node.nV)) #initiate a prime variable in its parent
            push!(node.parent.dual,  node.ID => zeros(node.nV)) #initiate a dual variable in its parent
        end
    end
end


function print_tree(node::linknode, depth=0)
    println("  "^depth * "Node $(node.ID)")  # Indent based on depth
    if node.children !== nothing
        for child in node.children
            print_tree(child, depth + 1)
        end
    end
end



root=linknode(string(countID+=1))
topo_gen!(root, 15, 4)
assign_var(root)

print_tree(root)
println(root.prime)
println(root.dual)