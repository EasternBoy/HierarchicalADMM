using JuMP
import Clarabel
import LinearAlgebra
import MathOptInterface as MOI

function LMIsolver(A, B, nD, ρ = 1., σ = 0.5, α = 1/2, l = 1/2)
    
    model = Model(Clarabel.Optimizer)
    set_silent(model) 
    @variable(model, γ >= 0.)
    @variable(model, 1. >= t >= 0.)

    for d in 1:nD-1
        @constraint(model,  t >= σ*opnorm(B[d])^2/(2*α))
    end    

    s  = [size(B[d], 1) for d in 1:nD-1]
    ss = sum(s)
    Is = diagm(ones(ss))

    #------------------ Calculate Ξ11 ------------------
    Λ1 = B[1]
    for d in 2:nD-1
        Λ1 = blockdiag_dense(Λ1, B[d])
    end

    Λ2 = A[2]
    for d in 3:nD-1
        Λ2 = blockdiag_dense(Λ2, A[d])
    end

    Λ2 = [Λ2  zeros(size(Λ2, 1), size(B[nD-1], 2))]
    Λ2 = [zeros(size(B[1],1), size(Λ2,2)); Λ2]

    Λ = Λ1 + Λ2
    Ξ11 = Λ*Λ'

    # for row in eachrow(Ξ11)
    #        println(row)
    # end
    # println(minimum(eigvals(Ξ11)))
    
    Ξ11 = 2(1-t)/l*Ξ11

    #------------------ Calculate Ξ22------------------
    Ξ22 = (2t*α/opnorm(B[1])^2)*diagm(ones(s[1]))
    for d in 2:nD-1
        Ξ22 = blockdiag_dense(Ξ22, (2t*α/opnorm(B[d])^2)*diagm(ones(s[d])))
    end

    #------------------ Calculate Ξ33------------------
    Ξ33 = ρ*Is

    #------------------ Calculate Ξ44------------------
    Ξ44 = zeros(s[1], s[1])
    for d in 2:nD-1
        Ξ44 = blockdiag_dense(Ξ44, A[d]*A[d]')
    end
    Ξ44 = (1-t)/l*Ξ44 + (1/ρ + σ/ρ^2)*Is

    #------------------ Calculate Ξ41 ------------------
    Ξ411 = zeros(s[1], s[1])
    for d in 2:nD-1
        Ξ411 = blockdiag_dense(Ξ411, -1/l*A[d]*A[d]')
    end

    Ξ412 = -1/l*A[2]*B[1]'
    for d in 2:nD-2
        Ξ412 = blockdiag_dense(Ξ412, -1/l*A[d+1]*B[d]')
    end
    Ξ412 = [zeros(s[1], size(Ξ412,2)); Ξ412]
    Ξ412 = [Ξ412 zeros(size(Ξ412,1), s[nD-1])]
    Ξ41  = 2(1-t)*(Ξ411 + Ξ412)

    #------------------ Calculate Ξ42------------------
    Ξ42  = -σ/ρ*Is

    #------------------ Calculate Ξ43------------------
    Ξ43  = -Is

    Zs = zeros(ss,ss)
    Θ  = [1/ρ*Is Zs         Zs Zs;
          Zs     (ρ + σ)*Is Zs Zs;
          Zs     Zs         Zs Zs;
          Zs     Zs         Zs Zs]

    ##------------------ LMIsolver ------------------
    blkM = [Ξ11  Zs    Zs    Ξ41';
            Zs   Ξ22   Zs    Ξ42';
            Zs   Zs    Ξ33   Ξ43';
            Ξ41  Ξ42   Ξ43   Ξ44]
                 
    @constraint(model, blkM - γ*Θ in PSDCone())

    J = γ
    @objective(model, Max, J)
    optimize!(model)

    return JuMP.value.(γ), JuMP.value.(t)
end

function topo_matrix(root::linknode, nD, A, B)
    for d in 1:nD-1
        A[d], B[d] = A_B_d(root, d)
    end
end

function A_B_d(root::linknode, d::Int64)
    Ad = nothing
    Bd = nothing

    SLnode_arr = linknode[]

    get_same_level_nodes(root, SLnode_arr, d)

    for node in SLnode_arr
        if node.children !== nothing
            if Ad === nothing
                Ad = Adi(node)
            else
                Ad = blockdiag(Ad, Adi(node))
            end

            # if Bd === nothing
            #     Bd = Bdi(node)
            # else
            #     Bd = blockdiag(Bd, Bdi(node))
            # end 
            
            for child in node.children
                if child.children !== nothing
                    if Bd === nothing
                        Bd = - spdiagm(ones(child.nV))
                    else
                        Bd = blockdiag(Bd, - spdiagm(ones(child.nV)))
                    end
                end
                
                if child.children === nothing
                    if Bd === nothing
                        Bd = - spdiagm(ones(child.nV))
                    else
                        Bd = blockdiag(Bd, - spdiagm(ones(child.nV)))
                    end
                end
            end
        end
    end

    for node in SLnode_arr
        if node.children === nothing
            Ad = [Ad spzeros(size(Ad,1), node.nV)]
        end
    end

    return Matrix(Ad), Matrix(Bd)
end

function Adi(node::linknode)
    return spdiagm(ones(node.nV))
end

# function Bdi(node::linknode)
#     Bdi = nothing
#     for child in node.children
#         if Bdi === nothing
#             Bdi = - spdiagm(ones(child.nV))
#         else
#             Bdi = blockdiag(Bdi, - spdiagm(ones(child.nV)))
#         end
#     end
#     return Bdi
# end

function get_same_level_nodes(node::linknode, node_arr::Vector{linknode}, target::Int64, level = 1)
    if level == target
        push!(node_arr, node)
        return 0
    end
    if node.children !== nothing
        for child in node.children
            get_same_level_nodes(child, node_arr, target, level+1)
        end
    end
end

function blockdiag_dense(mats::AbstractMatrix...)
    total_rows = sum(size(M,1) for M in mats)
    total_cols = sum(size(M,2) for M in mats)
    out = zeros(eltype(mats[1]), total_rows, total_cols)

    r, c = 1, 1
    for M in mats
        rows, cols = size(M)
        out[r:r+rows-1, c:c+cols-1] .= M
        r += rows
        c += cols
    end
    return out
end