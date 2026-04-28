struct CONDENSED_t
    parent::Int
    child::Int
    value::Float64
    cluster_size::Int
end

struct HIERARCHY_t
    left_node::Int
    right_node::Int
    value::Float64
    cluster_size::Int
end
#=
struct HIERARCHY_dtype
    left_node::Int
    right_node::Int
    value::Float64
    cluster_size::Int
end

struct CONDENSED_dtype
    parent::Int
    child::Int
    value::Float64
    cluster_size::Int
end
=#
const INFTY::Float64 = Inf
const NOISE::Int = -1

function tree_to_labels(
    single_linkage_tree::Vector{HIERARCHY_t},
    min_cluster_size::Int=10,
    cluster_selection_method::String="eom",
    allow_single_cluster::Bool=false,
    cluster_selection_epsilon::Float64=0.0,
    max_cluster_size=nothing) ::Tuple{Vector{Int}, Vector{Float64}}

    condensed_tree::Vector{CONDENSED_t}
    labels::Array{Int}
    probabilities::Array{Float64}

    labels, probabilities = _get_clusters(
        condensed_tree,
        _compute_stability(condensed_tree),
        cluster_selection_method,
        allow_single_cluster,
        cluster_selection_epsilon,
        max_cluster_size,
    )
    return (labels, probabilities)
end

function bfs_from_hierarchy(
    hierarchy::Vector{HIERARCHY_t},
    bfs_root::Int)

    n_samples = length(hierarchy) + 1
    process_queue = [bfs_root]
    result = Int[]

    while !isempty(process_queue)
        append!(result, process_queue)

        process_queue = [x - n_samples for x in process_queue if x > n_samples]

        if !isempty(process_queue)
            next_queue = Int[]
            for node in process_queue
                h = hierarchy[node]
                append!(next_queue, [h.left_node, h.right_node])
            end
            process_queue = next_queue
        end
    end
    return result
end

function _condense_tree(
    hierarchy::Vector{HIERARCHY_t},
    min_cluster_size::Int=10)

    root::Int = 2 * length(hierarchy)
    n_samples::Int = length(hierarchy) + 1
    next_label::Int = n_samples + 1

    result_list,
    node_list = bfs_from_hierarchy(hierarchy, root)

    relabel = zeros(Int, root + 1) #Inizializza a 0, non a empty rispetto a np.empty
    relabel[root] = n_samples

    ignore = falses(length(node_list))

    for node in node_list
        if ignore[node+1] || node < n_samples  
            continue
        end

        children = hierarchy[node - n_samples + 1]
        left = children.left_node
        right = children.right_node
        distance = children.value

        if  distance > 0
            lambda_value = 1.0 / distance
        else 
            lambda_value = INFTY
        end

        if left >= n_samples
            left_count = hierarchy[left - n_samples + 1].cluster_size
        else
            left_count = 1
        end

        if right >= n_samples
            right_count = hierarchy[right - n_samples + 1].cluster_size 
        else    
            right_count = 1
        end
        
        if left_count >= min_cluster_size && right_count >= min_cluster_size
            relabel[left + 1] = next_label
            next_label += 1
            append!(result_list, CONDENSED_t(relabel[node+1], relabel[left + 1], lambda_value, left_count))
            

            relabel[right + 1] = next_label
            next_label += 1
            append!(result_list, CONDENSED_t(relabel[node+1], relabel[right + 1] , lambda_value, right_count))
            

        elseif left_count < min_cluster_size && right_count < min_cluster_size
            for sub_node in bfs_from_hierarchy(hierarchy, left)
                if sub_node < n_samples
                    append!(result_list, CONDENSED_t(relabel[node + 1], sub_node, lambda_value, 1))
                end
                ignore[sub_node] = true
            end

            for sub_node in bfs_from_hierarchy(hierarchy, right)
                if sub_node < n_samples
                    append!(result_list, CONDENSED_t(relabel[node + 1], sub_node, lambda_value, 1))
                end
                ignore[sub_node] = true
            end

        elseif left_count < min_cluster_size
            relabel[right + 1] = relabel[node + 1]
            for sub_node in bfs_from_hierarchy(hierarchy, left)
                if sub_node < n_samples
                    append!(result_list, CONDENSED_t(relabel[node + 1], sub_node, lambda_value, 1))
                end
                ignore[sub_node] = true
            end

        else
            relabel[left + 1] = relabel[node + 1]
            for sub_node in bfs_from_hierarchy(hierarchy, right)
                if sub_node < n_samples
                    append!(result_list, CONDENSED_t(relabel[node], sub_node, lambda_value, 1))
                end
                ignore[sub_node] = true
            end
        end
    end

    return result_list
end

function _compute_stability(condensed_tree::Vector{CONDENSED_t})

    parents = [n.parent for n in condensed_tree]

    largest_child = maximum(n.child for n in condensed_tree)
    smallest_cluster = minimum(parents)
    num_clusters = maximum(parents) - smallest_cluster + 1

    largest_child = max(largest_child, smallest_cluster)

    births = fill(NaN, largest_child + 1)

    for idx in 1:length(condensed_tree)
        condensed_node = condensed_tree[idx]
        births[condensed_node.child] = condensed_node.value
    end

    births[smallest_cluster] = 0.0

    result = zeros(Float64, num_clusters)

    for idx in 1:length(condensed_tree)
        condensed_node = condensed_tree[idx]
        parent = condensed_node.parent
        lambda_val = condensed_node.value
        cluster_size = condensed_node.cluster_size
        
        result_index = parent - smallest_cluster + 1
        result[result_index] += (lambda_val - births[parent]) * cluster_size
    end
    
    stability_dict = Dict{Int, Float64}()

    for idx in 1:num_clusters
        stability_dict[idx + smallest_cluster - 1] = result[idx]
    end

    return stability_dict
end

function bfs_from_cluster_tree(
    condensed_tree::Vector{CONDENSED_t},
    bfs_root::Int)
    result,
    process_queue::Vector{Int} = bfs_root::Array{Int}
    children = [n.child for n in condensed_tree]
    parents = [n.parent for n in condensed_tree]

    while !isempty(process_queue)
        push!(result, process_queue)
        process_queue = children[in.(parents, process_queue)]
    end
    
    return result
end

function max_lambdas(condensed_tree::Vector{CONDENSED_t})
    parent,
    parents = [n.parent for n in condensed_tree]
    largest_parent = maximum(parents)
    deaths = zeros(Float64, largest_parent + 1)
    current_parent = condensed_tree[1].parent
    max_lambda = condensed_tree[1].value

    for idx in 1:length(condensed_tree)
        parent = condensed_tree[idx].parent
        lambda_val = condensed_tree[idx].value

        if parent == current_parent
            max_lambda = max(max_lambda, lambda_val)
        else
            deaths[current_parent] = max_lambda
            current_parent = parent
            max_lambda = lambda_val
        end
    end
    deaths[current_parent] = max_lambda
    return deaths
end

function labelling_at_cut(
    linkage::Vector{HIERARCHY_t},
    cut::Float64,
    min_cluster_size::Int)

    root = 2 * length(linkage)
    n_samples = div(root, 2) + 1

    union_find = TreeUnionFind(root + 1)
    result = zeros(Int, n_samples)

    cluster = n_samples

    for node in linkage
        if node.value < cut
            union(union_find, node.left_node, cluster)
            union(union_find, node.right_node, cluster)
        end
        cluster += 1
    end

    cluster_size = zeros(Int, cluster)

    for n in n_samples
        cluster = find(union_find, n)
        cluster_size[cluster] += 1
        result[n] = cluster
    end

    cluster_label_map = Dict{Int, Int}(-1 => NOISE)
    unique_labels = unique(result)

    cluster_label = 0
    for cluster in unique_labels
        if cluster_size[cluster] < min_cluster_size
            cluster_label_map[cluster] = NOISE
        else
            cluster_label_map[cluster] = cluster_label
            cluster_label += 1
        end
    end

    for n in n_samples
        result[n] = cluster_label_map[result[n]]
    end

    return result
end

function _do_labelling(
    condensed_tree::Vector{CONDENSED_t},
    clusters::Set{Int},
    cluster_label_map::Dict{Int,Int},
    allow_single_cluster::Int,
    cluster_selection_epsilon::Float64
)
    child_array = [n.child for n in condensed_tree]
    parent_array = [n.parent for n in condensed_tree]
    lambda_array = [n.value for n in condensed_tree]

    root_cluster = minimum(parent_array)
    result = Vector{Int}(undef, root_cluster)

    union_find = TreeUnionFind(maximum(parent_array) + 1)

    for n in condensed_tree
        child = child_array[n]
        parent = parent_array[n]
        if !(child in clusters)
            union(union_find, parent, child)
        end
    end

    for n in root_cluster
        cluster = find(union_find, n)
        label = NOISE

        if cluster != root_cluster
            label = cluster_label_map[cluster]

        elseif length(clusters) == 1 && allow_single_cluster
            parent_lambda = lambda_array[findall(==(n), child_array)]
                if  cluster_selection_epsilon != 0.0
                    threshold = 1 / cluster_selection_epsilon
                else
                maximum(lambda_array[parent_array .== cluster])
                end
            if !isempty(parent_lambda) && parent_lambda[1] ? threshold
                label = cluster_label_map[cluster]
            end
        end

        result[n+1] = label
    end

    return result
end

function get_probabilities(
    condensed_tree::Vector{CONDENSED_t},
    cluster_map::Dict{Int,Int},
    labels::Vector{Int}
)
    child_array = [n.child for n in condensed_tree]
    parent_array = [n.parent for n in condensed_tree]
    lambda_array = [n.value for n in condensed_tree]

    result = zeros(Float64, length(labels))
    deaths = max_lambdas(condensed_tree)

    root_cluster = minimum(parent_array)

    for n in condensed_tree
        point = child_array[n]
        if point >= root_cluster
            continue
        end

        cluster_num = labels[point]
        if cluster_num == -1
            continue
        end

        cluster = cluster_map[cluster_num]
        max_lambda = deaths[cluster]

        if max_lambda == 0.0 || isinf(lambda_array[n])
            result[point+1] = 1.0
        else
            lambda_val = min(lambda_array[n], max_lambda)
            result[point+1] = lambda_val / max_lambda
        end
    end

    return result
end

function recurse_leaf_dfs(cluster_tree::Vector{CONDENSED_t}, current_node::Int)
    children = [n.child for n in cluster_tree if n.parent == current_node]

    if isempty(children)
        return [current_node]
    else
        return vcat([recurse_leaf_dfs(cluster_tree, child) for child in children])
    end
end

function get_cluster_tree_leaves(cluster_tree::Vector{CONDENSED_t})
    if isempty(cluster_tree) 
        return []
    end
    root = minimum(n.parent for n in cluster_tree)
    return recurse_leaf_dfs(cluster_tree, root)
end

function traverse_upwards(
    cluster_tree::Vector{CONDENSED_t},
    cluster_selection_epsilon::Float64,
    leaf::Int,
    allow_single_cluster::Bool
)
    root = minimum(n.parent for n in cluster_tree)

    parent = first(n.parent for n in cluster_tree if n.child == leaf)

    if parent == root
        if allow_single_cluster
            return parent
        else
            return leaf
        end
    end

    parent_val = first(n.value for n in cluster_tree if n.child == parent)
    parent_eps = 1 / parent_val

    if parent_eps > cluster_selection_epsilon
        return parent
    else
        return traverse_upwards(
            cluster_tree,
            cluster_selection_epsilon,
            parent,
            allow_single_cluster)
    end
end

function epsilon_search(
    leaves::Set{Int},
    cluster_tree::Vector{CONDENSED_t},
    cluster_selection_epsilon::Float64,
    allow_single_cluster::Bool
)
    selected_clusters = []
    processed = []

    for leaf in leaves
        leaf_nodes = children == leaf

        eps = 1 / vals[1]

        if eps < cluster_selection_epsilon
            if !(leaf in processed)
                ec = traverse_upwards(cluster_tree, cluster_selection_epsilon,
                    leaf, allow_single_cluster)
                push!(selected_clusters, ec)

                for sub_node in bfs_from_cluster_tree(cluster_tree, ec)
                    if sub_node != ec
                        push!(processed, sub_node)
                    end
                end
            end
        else
            push!(selected_clusters, leaf)
        end
    end

    return Set(selected_clusters)
end

function _get_clusters(
    condensed_tree::Vector{CONDENSED_t},
    stability::Dict{Int,Float64};
    cluster_selection_method::String="eom",
    allow_single_cluster::Bool=false,
    cluster_selection_epsilon::Float64=0.0,
    max_cluster_size=nothing
)
    nodes_list = sort(collect(keys(stability)), rev=true)

    if !allow_single_cluster
        nodes_list = nodes_list[1:end-1]
    end

    cluster_tree = [n for n in condensed_tree if n.cluster_size > 1]

    is_cluster = Dict(n => true for n in nodes_list)

    n_samples = maximum(n.child for n in condensed_tree if n.cluster_size == 1) + 1

    if max_cluster_size === nothing
        max_cluster_size = n_samples + 1
    end

    #cluster_sizes = Dict(n.child => n.cluster_size for n in cluster_tree)

    if allow_single_cluster
        root = nodes_list[end]
        cluster_sizes[root] = sum(n.cluster_size for n in cluster_tree if n.parent == root)
    end

    if cluster_selection_method == "eom"
        for node in nodes_list
            children = [n.child for n in cluster_tree if n.parent == node]
            subtree_stability = sum(stability[ch] for ch in children)

            if subtree_stability > stability[node] || cluster_sizes[node] > max_cluster_size
                is_cluster[node] = false
                stability[node] = subtree_stability
            else
                for sub in bfs_from_cluster_tree(cluster_tree, node)
                    if sub != node
                        is_cluster[sub] = false
                    end
                end
            end
        end

        if cluster_selection_epsilon != 0.0 && !isempty(cluster_tree)
            eom_clusters = [c for (c,v) in is_cluster if v]

            selected_clusters = []
                if length(eom_clusters) == 1 &&
                   eom_clusters[1] == minimum(n.parent for n in cluster_tree)
                    if allow_single_cluster 
                        selected_clusters = eom_clusters
                else
                    epsilon_search(Set(eom_clusters), cluster_tree,
                                   cluster_selection_epsilon,
                                   allow_single_cluster)
                end

            for c in is_cluster
                if c in selected_clusters
                is_cluster[c] = true
                else
                is_cluster[c] = false
                end
            end
        end

    elseif cluster_selection_method == "leaf"
        leaves = Set(get_cluster_tree_leaves(cluster_tree))

        if isempty(leaves)
            for n in is_cluster
                is_cluster[n] = false
            end
            is_cluster[minimum(n.parent for n in condensed_tree)] = true
        end

        if cluster_selection_epsilon != 0.0
            selected_clusters = epsilon_search(leaves, cluster_tree,
                           cluster_selection_epsilon,
                           allow_single_cluster)
        else
            selected_clusters = leaves

        for c in is_cluster
            is_cluster[c] = c in selected_clusters
            end
        end
    end

    clusters = Set(c for (c,v) in is_cluster if v)

    cluster_map = Dict(c => i-1 for (i,c) in enumerate(sort(collect(clusters))))
    reverse_cluster_map = Dict(v => k for (k,v) in cluster_map)

    labels = do_labelling(condensed_tree, clusters, cluster_map,
                          allow_single_cluster, cluster_selection_epsilon)

    probs = get_probabilities(condensed_tree, reverse_cluster_map, labels)

    return labels, probs
end