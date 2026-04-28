using Test

@test [5, 3, 4, 1, 2] = bfs_from_hierarchy([HIERARCHY_t(1, 2, 0.1, 2), HIERARCHY_t(3, 4, 0.2, 3)], 5)

#=
_compute_stability([CONDENSED_t(4, 3, 0.3, 2),
    CONDENSED_t(3, 1, 0.1, 1),
    CONDENSED_t(3, 2, 0.2, 1)])
Dict{Int64, Float64} with 2 entries:
  4 => NaN
  3 => 0.3

max_lambdas([CONDENSED_t(3, 1, 0.1, 1),
        CONDENSED_t(3, 2, 0.4, 1), 
        CONDENSED_t(4, 3, 0.2, 2),
        CONDENSED_t(4, 5, 0.6, 1)])
    4-element Vector{Float64}:
    0.0
    0.0
    0.4
    0.6

recurse_leaf_dfs([CONDENSED_t(5, 3, 0.0, 2),
        CONDENSED_t(5, 4, 0.0, 1),
        CONDENSED_t(3, 1, 0.0, 1),
        CONDENSED_t(3, 2, 0.0, 1)], 5)
    2-element Vector{Vector}:
 [[1], [2]]
 [4]

traverse_upwards([CONDENSED_t(3, 1, 0.1, 1),
        CONDENSED_t(4, 3, 0.2, 2),
        CONDENSED_t(5, 4, 0.5, 3)], 100.0, 1, true)
    3
=#


