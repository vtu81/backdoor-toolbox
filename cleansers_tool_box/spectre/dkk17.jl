using TensorToolbox
using LinearMaps
using KrylovKit

include("util.jl")

function cov_Tail(T, d, ε, τ)
    #N = (d * log(d/ε/τ))^6 / ε^2
    #12 * exp(-T) + 3ε/(d*log(N/τ))
    if T <= 10 * log(1/ε)
        return 1
    end
    return 3ε/(T * log(T))^2
end

function Q(G, P)
    # Computes the mean of x'Px - tr(P) for x ~ N(0, I)

    #χ = Chisq(1)
    #return mean((sum(v*rand(χ) for v in c) - tr(P))^2 for _ in 1:10000

    #P = (P + P') ./ 2
    #Z = cholesky(G.Σ)
    #M = Z.U*P*Z.L
    #U = eigvecs(M)'
    #c = diag(U*M*U')
    #return 2*norm(c)^2 + sum(c*c') - 2*tr(P)*sum(c) + tr(P)^2

    return 2norm(P)^2
end

function cov_estimation_filter(S′, ε, τ=0.1; limit=nothing, method=:krylov)
    d, n = size(S′)
    C = 10
    C′ = 0
    Σ′ = S′*S′' ./ n
    # Σ′ = cov(S′', corrected=false)
    Σ′ += 1e-8 * I
    G′ = MvNormal(Symmetric(Σ′))
    # invsqrtΣ′ = Symmetric(Σ′)^(-1/2)
    invsqrtΣ′ = sqrt(inv(Symmetric(Σ′)))
    Y = invsqrtΣ′ * S′
    xinvΣ′x = [y'y for y in eachcol(Y)]
    mask = xinvΣ′x .>= C*d*log(n/τ)
    if any(mask)
        println("early filter")
        if limit == nothing
            return .! mask
        else
            return .! mask .| k_lowest_ind(xinvΣ′x, max(0, n - limit))
        end
    end
    if method == :arpack
        Z = mapcols(y->kron(y, y), Y)
        Id♭ = ♭(Matrix(I, d, d))
        TS′ = Symmetric(-Id♭ * Id♭' + Z * Z' ./ n)
        (λ,), v = eigs(TS′; nev=1)
    else
        Z = LinearMap(v->krtv(Y, Y, v), v->tkrtv(Y', Y', v), d^2, n)
        Id♭ = LinearMap(reshape(♭(Matrix(I, d, d)), :, 1))
        TS′ = -Id♭ * Id♭' + Z * Z' / n
        (λ,), (v,) = eigsolve(TS′, ones(d^2), issymmetric=true)
    end
    if λ <= (1 + C*ε*log(1/ε)^2)*Q(collect(invsqrtΣ′) * G′, ♯(v)) / 2
        return G′
    end
    V = Symmetric(♯(v) + ♯(v)')/2
    ps = [1/√2 * (y'V*y - tr(V)) for y in eachcol(Y)]
    μ = median(ps)
    diffs = abs.(ps .- μ)
    for (i, diff) in enumerate(sort(diffs))
        shift = 3
        if diff < shift
            continue
        end
        T = diff - shift
        if T <= C′
            continue
        end
        if i/n >= cov_Tail(T, d, ε, τ)
            if limit == nothing
                return diffs .<= T
            else
                return (diffs .<= T) .| k_lowest_ind(diffs, max(0, n - limit))
            end
        end
    end
end

function cov_estimation_iterate(S′, ε, τ=0.1, k=nothing; iters=nothing, limit=nothing)
    _, n = size(S′)
    idxs = 1:n
    i = 0
    if limit != nothing
        orig_limit = limit
        p = Progress(limit, 1)
    end
    while true
        if iters != nothing && i >= iters
            break
        end
        if k == nothing
            S′k = S′
        else
            S′k, _ = pca(S′, k)
        end
        select = cov_estimation_filter(S′k, ε, τ, limit=limit)
        if select isa MvNormal
            println("Terminating early $(i) success...")
            break
        end
        if select == nothing
            println("Terminating early $(i) fail...")
            break
        end
        if limit != nothing
            limit -= length(select) - sum(select)
            @assert limit >= 0
            update!(p, orig_limit - limit)
        end
        S′ = S′[:, select]
        idxs = idxs[select]
        i += 1
        if limit == 0
            break
        end
    end
    select = falses(n)
    for i in idxs
        select[i] = 1
    end
    return select
end

function rcov(S′, ε, τ=0.1, k=nothing; iters=nothing, limit=nothing)
    select = cov_estimation_iterate(S′, ε, τ, k; iters=iters, limit=limit)
    selected = S′[:, select]
    return selected*selected'
end

function rpca(S′, ε, τ=0.1, k=100; iters=nothing, limit=nothing)
    d, n = size(S′)
    perm = randperm(n)
    S′paired = S′[:, perm][:, 1:div(n, 2)] - S′[:, perm][:, div(n, 2)+1:end-n%2]
    if limit != nothing
        # TODO: Is this correction right?
        limit = round(Int, limit - limit^2/2/n)
    end
    selected = cov_estimation_iterate(S′paired, ε, τ, iters=iters, limit=limit)
    _, U = pca(S′paired[:, selected], k)
    S′selected = U*S′paired[:, selected]
    return U * S′, U, cov(S′paired[:, selected]'U', corrected=false) ./ 2
end

mean_Tail(T, d, ε, δ, τ, ν=1) = 8exp(-T^2/(2ν)) + 8ε/(T^2*log(d*log(d/(ε*τ))))

function mean_estimation_filter(S′, ε, τ=0.1, ν=1; limit=nothing)
    d, n = size(S′)
    μ, Σ = mean(S′, dims=2), cov(S′', corrected=false)
    (λ,), v = eigs(Σ; nev=1)
    if λ - 1 <= ε * log(1/ε)
        return
    end
    δ = 3sqrt(ε * (λ - 1))
    λmags = abs.((S′ .- μ)'v)[:]
    for (i, mag) in enumerate(sort(λmags))
        if mag < δ
            continue
        end
        T = mag - δ
        if (n - i)/n > mean_Tail(T, d, ε, δ, τ, ν)
            if limit == nothing
                return λmags .<= mag
            else
                return (λmags .<= mag) .| k_lowest_ind(λmags, max(0, n - limit))
            end
        end
    end
end

function mean_estimation_iterate(A, ε, τ=0.1, ν=1; iters=nothing, limit=nothing)
    d, n = size(A)
    idxs = 1:n
    i = 0
    if limit != nothing
        orig_limit = limit
        p = Progress(limit, 1)
    end
    while true
        if iters != nothing && i >= iters
            break
        end
        select = mean_estimation_filter(A, ε, τ, ν, limit=limit)
        if select == nothing
            println("Terminating early $(i)...")
            break
        end
        if limit != nothing
            limit -= length(select) - sum(select)
            @assert limit >= 0
            update!(p, orig_limit - limit)
        end
        A = A[:, select]
        idxs = idxs[select]
        i += 1
    end
    select = falses(n)
    for i in idxs
        select[i] = 1
    end
    return select
end

function rmean(A, ε, τ=0.1, ν=1; iters=nothing, limit=nothing)
    select = mean_estimation_iterate(A, ε, τ, ν; iters=iters, limit=limit)
    return mean(A[:, select], dims=2)
end
