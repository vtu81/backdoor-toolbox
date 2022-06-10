using Clustering
using Random

function ind_to_indexes(c)
    a = similar(c, Int)
    count = 1
    @inbounds for i in eachindex(c)
        a[count] = i
        count += (c[i] != zero(eltype(c)))
    end
    return resize!(a, count-1)
end

function sample_ind_wo_replacement(c, k)
    a = falses(size(c))
    idxs = ind_to_indexes(c)
    perm = randperm(length(idxs))
    a[idxs[perm[1:k]]] .= 1
    return a
end

function kmeans_filter1(reps, eps)
    n = size(reps)[2]
    kmass = kmeans(reps, 2).assignments
    c1, c2 = kmass .== 1, kmass .== 2
    c1c, c1p, c2c, c2p = (
        sum(c1[1:end-eps]),
        sum(c1[end-eps+1:end]),
        sum(c2[1:end-eps]),
        sum(c2[end-eps+1:end])
    )
    if c1p/(c1p + c1c) > c2p/(c2p + c2c)
        good, bad = c1, c2
    else
        good, bad = c2, c1
    end
    return sample(ind_to_indexes(good))
end

function kmeans_filter2(reps, eps, k=64, limit=1.5)
    reps_pca = pca(reps, k)[1]
    to_remove = Set{Int}()
    n = 0
    while length(to_remove) < round(limit*eps)
        i = kmeans_filter1(reps_pca, eps)
        push!(to_remove, i)
        n += 1
        if n > 10000
            break
        end
    end
    ind = trues(size(reps)[2])
    for i in to_remove
        ind[i] = 0
    end
    return ind
end
