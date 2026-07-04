if nD == 3 && nN == 10
    const λₙ = 1.2e-2
    const λₛ = 1.5e-2
    const λₕ = 1e-2
elseif nD == 3 && nN == 20
    const λₙ = 2e-3
    const λₛ = 2e-3
    const λₕ = 2e-3
elseif nD == 5 && nN == 20 
    const λₙ = 1e-3
    const λₛ = 1.5e-3
    const λₕ = 1.2e-3
end

const opt_gap  = 0.01/100
const Niter    = (nD == 3) ? 40 : 60 # Number of iterations should run in mode == 2
const tol      = 1e-4
const max_iter = 1000



function monotone_func(arr)
    res = copy(arr)
    N   = length(arr)
    for i in 1:N
        if i > 1
            if arr[i] < res[i-1]
                res[i] = arr[i]
            else
                res[i] = res[i-1]
            end
        end
    end
    return res
end