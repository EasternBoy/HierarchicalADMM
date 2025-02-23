include("linknodes.jl")

#Forward genneration to create a tree topology
# function topo_gen!(parent::linknode, nN::Int64, nL::Int64, depth=1)

#     if flag == false
#         alloc = rand(1:nN-nL+1)
#         nc    = alloc
#     else
#         alloc = rand(1:nN-nL+1)
#     end
        

#     children = [linknode(string(depth)*string(i)) for i in 1:alloc]
#     set_relative!(parent, children)

#     if nL-1 > 0
#         for i in 1:nc
#             if nN - alloc > 0
#                 temp = topo_gen!(children[i], nN-alloc, nL-1, depth+1)
#                 alloc += temp
#             end
#         end
#     else #reach deepest level

#     end

#     return alloc
# end
global countID = -1
function topo_gen!(node::linknode, nN::Int64, nL::Int64, depth=1)
    global countID

    if nN - nL >= 0
        num_child = rand(1:nN-nL+1)
    else
        num_child = rand(1:nN)
    end

    children = [linknode(string(countID+=1)) for i in 1:num_child]
    set_relative!(node, children)

    res_alloc = nN - num_child

    arr_alloc = Vector{Int64}(zeros(num_child))

    if res_alloc != 0
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


root=linknode(string(countID+=1))

topo_gen!(root, 10, 4)

function assign_var(node::linknode)
    if node.children === nothing
        return 0
    else
        
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

print_tree(root)