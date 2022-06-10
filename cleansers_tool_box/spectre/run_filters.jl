using Printf
using NPZ
include("util.jl")
include("kmeans_filters.jl")
include("quantum_filters.jl")

log_file = open("run_filters.log", "a")

for name in ARGS
    target_label = parse(Int, split(split(name, "/")[2], "-")[1][end:end])
    @show target_label
    poison_indices = npzread("output/$(name)/poison_indices.npy")'
    reps = npzread("output/$(name)/reps.npy")'
    @show size(reps)
    n = size(reps)[2]
    eps = parse(Int, split(name, "-")[2][:])
    # eps = parse(Int, match(r"[0-9]+$", name).match)
    if eps <= 0
        eps = round(Int, 0.1 * n)
    end
    if eps > 0.33 * n
        eps = round(Int, 0.33 * n)
    end
    if n < 500 # shrink the poison budget if the class is too small; otherwise the score could be terribly high (which is bad for target class identification).
        if eps > 0.1 * n
            eps = round(Int, 0.1 * n)
        end
    end
    @show eps
    removed = round(Int, 1.5 * eps)
    @show removed

    # @printf("%s: Running PCA filter\n", name)
    # reps_pca, U = pca(reps, 1)
    # pca_poison_ind = k_lowest_ind(-abs.(mean(reps_pca[1, :]) .- reps_pca[1, :]), round(Int, 1.5*eps))
    # poison_removed = sum(pca_poison_ind[end-eps+1:end])
    # clean_removed = removed - poison_removed
    # @show poison_removed, clean_removed
    # @printf(log_file, "%s-pca: %d, %d\n", name, poison_removed, clean_removed)
    # npzwrite("output/$(name)/mask-pca-target.npy", pca_poison_ind)


    # @printf("%s: Running kmeans filter\n", name)
    # kmeans_poison_ind = .! kmeans_filter2(reps, eps)
    # poison_removed = sum(kmeans_poison_ind[end-eps+1:end])
    # clean_removed = removed - poison_removed
    # @show poison_removed, clean_removed
    # @printf(log_file, "%s-kmeans: %d, %d\n", name, poison_removed, clean_removed)
    # npzwrite("output/$(name)/mask-kmeans-target.npy", kmeans_poison_ind)

    @printf("%s: Running quantum filter\n", name)
    quantum_poison_ind, opnorm = rcov_auto_quantum_filter(reps, eps, poison_indices)
    quantum_poison_ind = .! quantum_poison_ind
    poison_removed = sum(quantum_poison_ind[poison_indices])
    clean_removed = removed - poison_removed
    @show poison_removed, clean_removed
    
    @printf("\n<Overall Performance Evaluation>\n")
    @printf("Elimination Rate = %d/%d = %.6f\n", poison_removed, eps, poison_removed / eps)
    @printf("Sacrifice Rate = %d/%d = %.6f\n", clean_removed, n - eps, poison_removed / (n - eps))
    @printf(log_file, "%s-quantum: %d, %d\n", name, poison_removed, clean_removed)
    
    npzwrite("output/$(name)/mask-rcov-target.npy", quantum_poison_ind)
    npzwrite("output/$(name)/opnorm.npy", opnorm)
end
