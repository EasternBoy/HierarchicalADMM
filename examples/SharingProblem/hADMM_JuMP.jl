include("TreeGeneration.jl")

function prox!(node::linknode; λ = 0.25)  

    opti = JuMP.Model(Ipopt.Optimizer)
    set_silent(opti)

    x0 = ones(node.nV)
    @variable(opti, x[i = 1:node.nV], base_name = string("x",node.ID), start = x0[i])
    @constraint(opti, x .>= 0)

    func = node.cost_func.val
    para = node.cost_func.para
    J = 0
    q = Float64[]

    
    if node.children === nothing #Leaf nodes
        q = node.parent.prime[node.ID] + node.parent.dual[node.ID] #query
        
        com_cost!(node.parent,q, 1)

        J += func(x; para = para) + 1/(2λ)*(x - q)'*(x - q)
    elseif node.parent === nothing #Root node
        nc = length(node.children)
        
        for i in 1:nc
            q  = [q; vect_prime(node.children[i]) - node.dual[node.children[i].ID]]
        end
        # Received prime from all children

        J += func(x; para = para) + 1/(2λ)*dot(x - q, x - q)
    else #Middle node
        nc = length(node.children)

        par  = node.parent.prime[node.ID] + node.parent.dual[node.ID]

        com_cost!(node.parent, par, 1) #Received prime + dual from parent

        ch_pr = Float64[]
        for i in 1:nc
            ch_pr = [ch_pr;  vect_prime(node.children[i])]
        end
        # Received prime from all children

        q = [q; par - vect_dual(node) + ch_pr]

        η  = para[1]
        τ  = para[2]
        nc = para[3]
        horizon = Int64(node.nV/nc)
        t = @variable(opti, [i=1:horizon], base_name = string("t",node.ID))
        @constraint(opti, t .>= 0)
        for i in 1:horizon
            @constraint(opti, t[i] >= sum([x[(j-1)*horizon + i] for j in 1:nc]) - τ)
        end
        J += η*sum(t) + (1/λ)*dot(x - q/2, x - q/2)
    end

    @objective(opti, Min, J)
    JuMP.optimize!(opti)
    x = JuMP.value.(x)

    #Update prime variable
    if node.children === nothing #leaf node has only one
        node.prime[node.ID] = x
    else
        index = 1
        for i in 1:nc
            ID  = node.children[i].ID
            nVi = node.children[i].nV
            node.prime[ID] = x[index:(index+nVi-1)]
            index += nVi
        end
    end
end

# function forward_hierarchicalADMM!(root::linknode; max_nlayer = 10)
#     #Update prime 
#     VecLink = [root]

#     for _ in 1:max_nlayer #maximum layer is 10
#         if !isempty(VecLink)
#             newVecLink = linknode[]
#             for node in VecLink
#                 prox!(node)
#                 if node.children !== nothing
#                     for child in node.children
#                         push!(newVecLink, child)
#                     end
#                 end
#             end
#             VecLink = newVecLink
#         end
#     end
# end


# function backward_hierarchicalADMM!(root::linknode, ter::Vector{Float64};  max_nlayer = 10)
#     VecLink = [root]

#     for _ in 1:max_nlayer #maximum layer is 10
#         if !isempty(VecLink)
#             newVecLink = linknode[]
#             max_abs = 0
#             for node in VecLink
#                 if node.children !== nothing
#                     for child in node.children
#                         res = node.prime[child.ID] - vect_prime(child)
#                         node.dual[child.ID] = node.dual[child.ID] + res
#                         max_abs = (max_abs > maximum(abs.(res))) ? max_abs : maximum(abs.(res))
#                         push!(ter, max_abs)
#                         if child.children !== nothing
#                             push!(newVecLink, child)
#                         end
#                     end
#                 end
#             end
#             VecLink = newVecLink
#         end
#     end
# end


function HADMM_JuMP!(node::linknode, ter::Vector{Float64})
    #Update prime
    prox!(node)

    if node.children !== nothing
        for child in node.children
            HADMM_JuMP!(child, ter)

            #Update dual
            prime_child = vect_prime(child)
            res = child.parent.prime[child.ID] -  prime_child
            child.parent.dual[child.ID] += res

            # Receive a prime variable of one child for updating dual variable
            com_cost!(child, prime_child, 1)

            #Take the maximum residual error for stopping criteria
            push!(ter, maximum(abs.(res)))
        end
    end
end