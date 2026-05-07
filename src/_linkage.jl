struct MST_edge_t
    current_node::Int64
    next_node::Int64
    distance::Float64
end

mutable struct UnionFind
    parent::Vector{Int}
    size::Vector{Int}
end

function UnionFind(n)
    parent = collect(0:n-1)
    size = ones(Int, n)
    return UnionFind(parent, size)
end

function find(U::UnionFind, x)
    while U.parent[x + 1] != x
        U.parent[x + 1] = parent[parent[x + 1]]
        x = U.parent[x + 1]
    end
    return x
end

function union(U::UnionFind, x, y)
    x_root = find(U, x)
    y_root = find(U, y)

    if x_root == y_root
        return x_root
    end

    if U.size[x_root +1] < U.size[y_root + 1]
        x_root, y_root = y_root, x_root
    end

    U.parent[y_root + 1]= x_root
    U.size[x_root] += U.size[y_root]

    return x_root
end

#=
function dist(dist_metric, )

end
=#


function mst_from_mutual_reachability(mutual_reachability::Matrix{Float64})
    n_samples = size(mutual_reachability, 1)

    mst = Vector{MST_edge_t}(undef, n_samples - 1)

    current_labels = collect(0:n_samples-1)
    current_node = 0

    min_reachability = fill(Inf, n_samples)

    for i in 1:(n_samples - 1)
        # filter labels
        mask = current_labels .!= current_node
        current_labels = current_labels[mask]

        left = min_reachability[mask]
        right = mutual_reachability[current_node+1, current_labels .+ 1]

        min_reachability = min.(left, right)

        new_node_index = argmin(min_reachability)
        new_node = current_labels[new_node_index]

        mst[i] = MST_edge_t(
            current_node,
            new_node,
            min_reachability[new_node_index]
        )

        current_node = new_node
    end

    return mst
end


function mst_from_data_matrix(
    raw_data::Matrix{Float64},
    core_distances::Vector{Float64},
    dist_metric,
    alpha::Float64 = 1.0)

    n_samples = size(raw_data, 1)
    num_features = size(raw_data, 2)

    mst = Vector{MST_edge_t}(undef, n_samples - 1)

    in_tree = zeros(UInt8, n_samples)
    min_reachability = fill(Inf, n_samples)
    current_sources = ones(Int64, n_samples)

    current_node = 0

    for i in 1:(n_samples - 1)
        in_tree[current_node + 1] = 1

        current_node_core_dist = core_distances[current_node + 1]

        new_reachability = Inf
        source_node = 0
        new_node = 0

        for j in 0:(n_samples - 1)
            if in_tree[j + 1] == 1
                continue
            end

            next_node_min_reach = min_reachability[j + 1]
            next_node_source = current_sources[j + 1]

            pair_distance = dist(dist_metric,
                view(raw_data, current_node + 1, :),
                view(raw_data, j + 1, :),
                num_features
            )

            pair_distance /= alpha

            next_node_core_dist = core_distances[j + 1]

            mutual_reachability_distance = max(
                current_node_core_dist,
                next_node_core_dist,
                pair_distance
            )

            if mutual_reachability_distance < next_node_min_reach
                min_reachability[j + 1] = mutual_reachability_distance
                current_sources[j + 1] = current_node

                if mutual_reachability_distance < new_reachability
                    new_reachability = mutual_reachability_distance
                    source_node = current_node
                    new_node = j
                end

            elseif next_node_min_reach < new_reachability
                new_reachability = next_node_min_reach
                source_node = next_node_source
                new_node = j
            end
        end

        mst[i] = MST_edge_t(source_node, new_node, new_reachability)
        current_node = new_node
    end

    return mst
end





function make_single_linkage(mst::Vector{MST_edge_t})
    n_samples = length(mst) + 1

    single_linkage = Vector{HIERARCHY_t}(undef, n_samples - 1)

    U = UnionFind(n_samples)

    for i in 1:(n_samples - 1)
        current_node = mst[i].current_node
        next_node = mst[i].next_node
        distance = mst[i].distance

        current_node_cluster = fast_find(U, current_node)
        next_node_cluster = fast_find(U, next_node)

        single_linkage[i] = HIERARCHY_t(
            current_node_cluster,
            next_node_cluster,
            distance,
            U.size[current_node_cluster+1] + U.size[next_node_cluster+1]
        )

        union(U, current_node_cluster, next_node_cluster)
    end

    return single_linkage

end