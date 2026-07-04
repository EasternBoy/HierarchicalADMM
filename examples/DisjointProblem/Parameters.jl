if nD == 3 && nN == 10
    const λₙ = 1.2e-2
    const λₛ = 1.5e-2
    const λₕ = 1e-2
elseif nD == 3 && nN == 20
    const λₙ = 2e-3
    const λₛ = 2e-3
    const λₕ = 2e-3
elseif nD == 5 && nN == 20 
    const λₙ = 2e-3
    const λₛ = 2e-3
    const λₕ = 1.5e-3
end

const opt_gap  = 0.01/100
const Niter    = 100 # Number of iterations should run in mode == 2
const tol      = 1e-4
const max_iter = 1000