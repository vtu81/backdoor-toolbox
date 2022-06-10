using LinearAlgebra
using Statistics
using SliceMap
using Arpack
using ProgressMeter
using Random
using Distributions
# import Seaborn
import Pandas

function pca(A, k)
    @assert k <= minimum(size(A))
    A_centered = A .- mean(A; dims=2)
    if k == minimum(size(A))
        U, = svd(A_centered)
    else
        (U,), = svds(A_centered; nsv=k)
    end
    return U' * A_centered, U'
end

# TODO: Use a Cholesky factorization
function svd_pow(A, p)
    LinearAlgebra.checksquare(A)
    U, S, V = svd(A)
    @assert all(S .> 0)
    return U * diagm(0=>S .^ p) * V'
end

function k_lowest_ind(A, k)
    @assert 0 <= k
    @assert k <= length(A)
    if k == 0
        return falses(size(A))
    end
    for (i, cut) in enumerate(sort(A[:]))
        if i >= k
            return A .<= cut
        end
    end
    @assert false
end

function step_vec(n, k)
    v = falses(n)
    v[1:k] .= 1
    return v
end

function sb_pairplot(A, clean=5000)
    d, n = size(A)
    df = Pandas.DataFrame(A')
    df["poison"] = .! step_vec(n, clean)
    Seaborn.pairplot(df, diag_kind="kde", hue="poison")
end

function ♭(A::AbstractMatrix)
    LinearAlgebra.checksquare(A)
    return A[:]
end

function ♯(v::AbstractArray)
    n = length(v)
    m = isqrt(n)
    @assert m*m == n
    return reshape(v, (m, m))
end
