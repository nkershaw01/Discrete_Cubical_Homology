#====================================================================
This file contains version 2 of the code, with a new method for 
computing faces, making it slightly faster than the old version 

The main function is discrete_homology(G,n) which computes 
H_n(G;F_p) where p is the smallest prime larger than n+1
====================================================================#


# Necessary packages
using SparseArrays; 
using LinearAlgebra; 
using Base.Threads; 


#===================================================================#
# Structure for graphs
#===================================================================#

# Define Graph Structure
mutable struct graph
    # array of vertex labels
    vertices::Vector{Any}

    # array of edges in the graph. Needs to be simple, symmetric, and reflexive
    edges::Vector{Tuple{Any,Any}}
end

# convenience constructor accepting ranges and other iterable containers
function graph(vertices, edges)
    new_vertices = Any[v for v in vertices]
    new_edges = Tuple{Any,Any}[(e[1], e[2]) for e in edges]
    return graph(new_vertices, new_edges)
end

#===================================================================#
# Finite-field helpers
#===================================================================#

# Determine whether an integer is prime 
function is_prime_int(n::Int)
    n < 2 && return false
    n == 2 && return true
    iseven(n) && return false

    d = 3
    while d <= n ÷ d
        n % d == 0 && return false
        d += 2
    end
    return true
end

# Return the smallest prime greater than n
function next_prime_after(n::Int)
    candidate = n + 1
    while !is_prime_int(candidate)
        candidate += 1
    end
    return candidate
end

# Compute the multiplicative inverse of a modulo the prime p
function inverse_mod_p(a::Integer, p::Int)
    a0 = mod(Int(a), p)
    a0 == 0 && error("zero has no multiplicative inverse modulo p")

    old_r, r = p, a0
    old_t, t = 0, 1

    while r != 0
        q = old_r ÷ r
        old_r, r = r, old_r - q * r
        old_t, t = t, old_t - q * t
    end

    old_r == 1 || error("coefficient is not invertible modulo p")
    return mod(old_t, p)
end

# Function to compute the rank of a sparse matrix over F_p
function ref_rank!(A::SparseMatrixCSC{T,Ti}, p::Int, inplace::Bool=false) where {T<:Integer,Ti<:Integer}
    p >= 2 || error("p must be at least 2")
    is_prime_int(p) || error("p must be prime")

    m, n = size(A)
    (m == 0 || n == 0) && return 0

    # pivot_basis[r] is either nothing or a normalized sparse column whose
    # largest nonzero row is r and whose coefficient at r is 1
    pivot_basis = Vector{Union{Nothing,Dict{Int,Int}}}(undef, m)
    fill!(pivot_basis, nothing)

    rows = rowvals(A)
    values = nonzeros(A)
    rank = 0

    for col_index in 1:n
        column = Dict{Int,Int}()

        for ptr in nzrange(A, col_index)
            row = rows[ptr]
            value = mod(Int(values[ptr]), p)
            value == 0 && continue

            new_value = mod(get(column, row, 0) + value, p)
            if new_value == 0
                delete!(column, row)
            else
                column[row] = new_value
            end
        end

        while !isempty(column)
            pivot = maximum(keys(column))
            basis_column = pivot_basis[pivot]

            if basis_column === nothing
                inv_pivot = inverse_mod_p(column[pivot], p)
                for row in collect(keys(column))
                    value = mod(column[row] * inv_pivot, p)
                    if value == 0
                        delete!(column, row)
                    else
                        column[row] = value
                    end
                end

                pivot_basis[pivot] = column
                rank += 1
                break
            end

            factor = column[pivot]
            for (row, basis_value) in basis_column
                new_value = mod(get(column, row, 0) - factor * basis_value, p)
                if new_value == 0
                    delete!(column, row)
                else
                    column[row] = new_value
                end
            end
        end
    end

    return rank
end

# dense fallback
function ref_rank!(A::AbstractMatrix{T}, p::Int, inplace::Bool=false) where {T<:Integer}
    return ref_rank!(sparse(A), p, inplace)
end

#===================================================================#
# Functions for preprocessing graphs
#===================================================================#

# Function that removes a vertex from a graph and neighborhood dictionary
function remove_vertex(v, G, nhoodDict)
    filter!(x -> x != v, G.vertices)
    filter!(e -> e[1] != v && e[2] != v, G.edges)

    pop!(nhoodDict, v, nothing)
    for key in keys(nhoodDict)
        delete!(nhoodDict[key], v)
    end

    return G, nhoodDict
end

#Function to detect and remove one dominated vertex. If none is found return the graph along with a false flag
function remove_vert_deg_n(G, nhoodDict)
    for key in collect(keys(nhoodDict))
        for v in nhoodDict[key]
            v == key && continue

            connected_to_everything = true
            for w in nhoodDict[key]
                w == key && continue
                if !(w in nhoodDict[v])
                    connected_to_everything = false
                    break
                end
            end

            if connected_to_everything
                G, nhoodDict = remove_vertex(key, G, nhoodDict)
                return G, nhoodDict, true
            end
        end
    end

    return G, nhoodDict, false
end

# Function to preprocess a graph, making homology computations quicker
function preprocess_graph(G::graph)
    G2 = graph(deepcopy(G.vertices), deepcopy(G.edges))
    nhoodDict = get_nhood_dict(G2)
    removed = true

    while removed
        G2, nhoodDict, removed = remove_vert_deg_n(G2, nhoodDict)
    end

    return G2
end

#===================================================================#
# Functions relevant to inputting and constructing graphs
#===================================================================#

# Function to make an edge collection symmetric and reflexive
function makeSymmetricReflexive(A)
    B = Tuple{Any,Any}[]
    seen = Set{Tuple{Any,Any}}()

    function add_edge!(e)
        edge = (e[1], e[2])
        if !(edge in seen)
            push!(seen, edge)
            push!(B, edge)
        end
    end

    for a in A
        add_edge!((a[1], a[2]))
        add_edge!((a[2], a[1]))
        add_edge!((a[1], a[1]))
        add_edge!((a[2], a[2]))
    end

    return B
end

# Function to quotient a graph by relating two vertices
function quotient_vertices(G::graph, v1, v2)
    new_vertices = Any[v for v in G.vertices if v != v2]
    new_edges = Tuple{Any,Any}[]

    for e in G.edges
        a = e[1] == v2 ? v1 : e[1]
        b = e[2] == v2 ? v1 : e[2]
        push!(new_edges, (a, b))
    end

    return graph(new_vertices, unique(new_edges))
end

# Function to generate the suspension of a graph
function suspension(G::graph, n::Int)
    n >= 1 || error("the suspension length must be at least 1")
    levels = n - 1
    vertices = Any[]
    edges = Tuple{Any,Any}[]

    for i in 1:levels
        for v in G.vertices
            push!(vertices, (v, i))
        end
    end

    for x in vertices
        for y in vertices
            if x[1] == y[1] && abs(x[2] - y[2]) == 1
                push!(edges, (x, y))
            elseif x[2] == y[2] && ((x[1], y[1]) in G.edges)
                push!(edges, (x, y))
            end
        end
    end

    push!(vertices, 'n')
    push!(vertices, 's')

    if levels >= 1
        for x in vertices
            if x isa Tuple
                x[2] == 1 && push!(edges, (x, 'n'))
                x[2] == 1 && push!(edges, ('n', x))
                x[2] == levels && push!(edges, (x, 's'))
                x[2] == levels && push!(edges, ('s', x))
            end
        end
    end

    push!(edges, ('n', 'n'))
    push!(edges, ('s', 's'))

    return graph(vertices, unique(edges))
end

# Function to generate the Cartesian product of an array of arrays
function cartesian_product(arrays)
    isempty(arrays) && return [()]

    first_array = arrays[1]
    rest_product = cartesian_product(arrays[2:end])
    return [(x, rest...) for x in first_array for rest in rest_product]
end

# Function to compute the box product of a list of graphs
function multBoxProd(graphs)
    vertices = cartesian_product([G.vertices for G in graphs])
    edges = Tuple{Any,Any}[]

    for v in vertices
        for w in vertices
            number_changed = 0
            valid = true

            for i in eachindex(v)
                if v[i] == w[i]
                    continue
                elseif (v[i], w[i]) in graphs[i].edges
                    number_changed += 1
                    if number_changed > 1
                        valid = false
                        break
                    end
                else
                    valid = false
                    break
                end
            end

            valid && push!(edges, (v, w))
        end
    end

    return graph(vertices, edges)
end

# Function to generate the complete graph on n vertices
function completeGraph(n::Int)
    vertices = Any[i for i in 1:n]
    edges = Tuple{Any,Any}[(v, w) for v in vertices for w in vertices]
    return graph(vertices, edges)
end

# Function to relabel the vertices of a graph as integers
function relabel_vertices(G::graph)
    transformDict = Dict{Any,Int}(G.vertices[i] => i for i in eachindex(G.vertices))
    new_vertices = Any[i for i in eachindex(G.vertices)]
    new_edges = Vector{Tuple{Any,Any}}(undef, length(G.edges))

    @threads for i in eachindex(G.edges)
        e = G.edges[i]
        new_edges[i] = (transformDict[e[1]], transformDict[e[2]])
    end

    return graph(new_vertices, new_edges)
end

# internal package-free pseudo-random integer generator
function _next_random_vertex!(state::Base.RefValue{UInt64}, n::Int)
    x = state[]
    x ⊻= x << 13
    x ⊻= x >> 7
    x ⊻= x << 17
    state[] = x
    return Int(mod(x, UInt64(n))) + 1
end

# Generate a random reflexive symmetric graph 
function random_graph(n_vert::Int, av_edge::Int; seed::Integer=0x9e3779b97f4a7c15)
    n_vert >= 1 || error("n_vert must be positive")
    av_edge >= 0 || error("av_edge must be nonnegative")

    vertices = Any[i for i in 1:n_vert]
    edges = Tuple{Any,Any}[]
    state = Ref(UInt64(seed == 0 ? 1 : seed))

    for x in vertices
        for _ in 1:av_edge
            y = _next_random_vertex!(state, n_vert)
            push!(edges, (x, y))
        end
        push!(edges, (x, x))
    end

    return graph(vertices, makeSymmetricReflexive(edges))
end

#=======================================================================================#
# Functions for generating equivalence classes under the hyperoctahedral group
#=======================================================================================#

# Function to generate the n-cube. Vertices are ordered so coordinate 1 changes fastest
function nCube(n::Int)
    n >= 0 || error("cube dimension must be nonnegative")

    vertices = Any[]
    for index in 0:(1 << n)-1
        push!(vertices, [Int((index >> (i - 1)) & 1) for i in 1:n])
    end

    face_pairs = Any[]
    for i in 1:n
        negative_face = Any[v for v in vertices if v[i] == 0]
        positive_face = Any[v for v in vertices if v[i] == 1]
        push!(face_pairs, [negative_face, positive_face])
    end

    return [vertices, face_pairs]
end

# Function to generate reversals as lists of -1 and 1
function generate_reversals(n::Int)
    n == 0 && return [Int[]]
    shorter_lists = generate_reversals(n - 1)
    return vcat([[1; list] for list in shorter_lists], [[-1; list] for list in shorter_lists])
end

# Function to generate all permutations of a list
function permutations(lst)
    isempty(lst) && return [Int[]]

    result = Any[]
    for i in eachindex(lst)
        current = lst[i]
        remaining = [lst[j] for j in eachindex(lst) if j != i]
        for permutation in permutations(remaining)
            push!(result, [current; permutation])
        end
    end
    return result
end

# Function to determine the sign of a permutation
function sign_of_permutation(perm)
    inversions = 0
    for i in 1:length(perm)-1
        for j in i+1:length(perm)
            perm[i] > perm[j] && (inversions += 1)
        end
    end
    return iseven(inversions) ? 1 : -1
end

# Generate all elements of the n-th hyperoctahedral group, stored as [symmetry, reversal, sign]
function hyperOctahedrial(n::Int)
    group = Any[]
    symmetric_group = permutations(collect(1:n))
    reversals = generate_reversals(n)

    for symmetry in symmetric_group
        for reversal in reversals
            sign = (isempty(reversal) ? 1 : prod(reversal)) * sign_of_permutation(symmetry)
            push!(group, [symmetry, reversal, sign])
        end
    end

    return group
end

# Permute the coordinates of a cube vertex according to a group element
function permuteCoords(vertex, gpElement)
    symmetry = gpElement[1]
    reversals = gpElement[2]
    image = Vector{Int}(undef, length(vertex))

    for i in eachindex(vertex)
        value = vertex[symmetry[i]]
        image[i] = reversals[i] == -1 ? 1 - value : value
    end

    return image
end

# Calculate the image of the standard n-cube under a group element
function calculateImageCube(cube, gpElement)
    return [permuteCoords(vertex, gpElement) for vertex in cube[1]]
end

# Convert a transformed vertex list into an index transformation
function transformationCoords(cube, image)
    coordinate_index = Dict{Tuple,Int}(Tuple(vertex) => i for (i, vertex) in enumerate(cube[1]))
    return [coordinate_index[Tuple(vertex)] for vertex in image]
end

# Generate the list [vertex permutation, sign] for the hyperoctahedral group
function generateEQClist(n::Int)
    cube = nCube(n)
    group = hyperOctahedrial(n)
    EQClist = Any[]

    for element in group
        transformation = transformationCoords(cube, calculateImageCube(cube, element))
        push!(EQClist, [transformation, element[3]])
    end

    return EQClist
end

# Generate the signed equivalence class of a map

function generateEQC(map, EQClist)
    equivalence_class = Any[]
    seen = Set{Tuple{Tuple,Int}}()

    for element in EQClist
        transformed_map = [map[index] for index in element[1]]
        key = (Tuple(transformed_map), element[2])
        if !(key in seen)
            push!(seen, key)
            push!(equivalence_class, [transformed_map, element[2]])
        end
    end

    return equivalence_class
end

#===================================================================#
# Face and graph-map functions
#===================================================================#

# Compute one face of a cube directly from the standard cube ordering
function face(cube, i, sgn)
    block = 1 << (Int(i) - 1)
    period = block << 1
    offset = Int(sgn) * block

    cube_face = Any[]
    sizehint!(cube_face, length(cube) ÷ 2)

    for start in (1 + offset):period:length(cube)
        for j in 0:block-1
            push!(cube_face, cube[start + j])
        end
    end

    return cube_face
end

# Create a list of face directions
faceList(n::Int) = collect(1:n)

# Compute all paired negative and positive faces of a map
function faces(map, facesList)
    return [[face(map, i, 0), face(map, i, 1)] for i in facesList]
end

# Generate a dictionary of graph neighborhoods
function get_nhood_dict(G::graph)
    nhoodDict = Dict{Any,Set{Any}}(v => Set{Any}() for v in G.vertices)
    for e in G.edges
        haskey(nhoodDict, e[1]) || (nhoodDict[e[1]] = Set{Any}())
        push!(nhoodDict[e[1]], e[2])
    end
    return nhoodDict
end

# Determine if a pair of signed (n-1)-cubes forms an n-cube
function isPairNcube(f, g, nhoodDict)
    f_map = f[1]
    g_map = g[1]
    length(f_map) == length(g_map) || return false

    for i in eachindex(f_map)
        if !(g_map[i] in nhoodDict[f_map[i]])
            return false
        end
    end
    return true
end

# Generate n-cube equivalence classes from (n-1)-cube equivalence classes
function graphMaps(A, G, EQClist, nhoodDict)
    if isempty(A)
        return Any[[[[v], 1]] for v in G.vertices]
    end

    thread_results = [Any[] for _ in 1:nthreads()]

    @threads :static for i in eachindex(A)
        local_results = thread_results[threadid()]
        representative = A[i][1]

        for j in i:length(A)
            for signed_map in A[j]
                if isPairNcube(representative, signed_map, nhoodDict)
                    new_map = vcat(representative[1], signed_map[1])
                    push!(local_results, generateEQC(new_map, EQClist))
                end
            end
        end
    end

    return reduce(vcat, thread_results; init=Any[])
end

# Check if a map is semi-degenerate
function checkSemiDegen(map, EQClist)
    for element in EQClist
        element[2] == -1 || continue
        transformed_map = [map[index] for index in element[1]]
        transformed_map == map && return true
    end
    return false
end

# Remove semi-degenerate equivalence classes
function semiNonDegen(maps)
    keep = Vector{Bool}(undef, length(maps))

    @threads for i in eachindex(maps)
        orbit = maps[i]
        representative_map = orbit[1][1]
        representative_sign = orbit[1][2]
        semi_degenerate = false

        for signed_map in orbit
            if signed_map[2] == -representative_sign && signed_map[1] == representative_map
                semi_degenerate = true
                break
            end
        end

        keep[i] = !semi_degenerate
    end

    return Any[maps[i] for i in eachindex(maps) if keep[i]]
end

# Check if a map appears in an equivalence class
function is_related(f, g)
    target = Tuple(f[1])
    return any(Tuple(element[1]) == target for element in g)
end

# Remove duplicate equivalence classes
function remove_duplicates(maps)
    unique_maps = Any[]
    seen_maps = Set{Tuple}()

    for orbit in maps
        representative = Tuple(orbit[1][1])
        representative in seen_maps && continue

        push!(unique_maps, orbit)
        for signed_map in orbit
            push!(seen_maps, Tuple(signed_map[1]))
        end
    end

    return unique_maps
end

# Generate a coordinate dictionary for equivalence classes.
function coordDict(lowerCubes)
    cdict = Dict{Tuple,Tuple{Int,Int}}()

    for i in eachindex(lowerCubes)
        for signed_map in lowerCubes[i]
            key = Tuple(signed_map[1])
            value = (i, signed_map[2])

            if haskey(cdict, key) && cdict[key] != value
                error("inconsistent signs in a non-semi-degenerate equivalence class")
            end
            cdict[key] = value
        end
    end

    return cdict
end

#===================================================================#
# Functions to compute homology over F_p
#===================================================================#

# Compute one boundary column as [row indices, values]
function boundarySumModP(cube_map, cdict, face_list, p::Int)
    coefficients = Dict{Int,Int}()

    for (i, face_pair) in enumerate(faces(cube_map, face_list))
        negative_key = Tuple(face_pair[1])
        if haskey(cdict, negative_key)
            row, orbit_sign = cdict[negative_key]
            value = mod(get(coefficients, row, 0) + (isodd(i) ? -1 : 1) * orbit_sign, p)
            value == 0 ? delete!(coefficients, row) : (coefficients[row] = value)
        end

        positive_key = Tuple(face_pair[2])
        if haskey(cdict, positive_key)
            row, orbit_sign = cdict[positive_key]
            value = mod(get(coefficients, row, 0) + (isodd(i) ? 1 : -1) * orbit_sign, p)
            value == 0 ? delete!(coefficients, row) : (coefficients[row] = value)
        end
    end

    rows = sort!(collect(keys(coefficients)))
    values = Int[coefficients[row] for row in rows]
    return [rows, values]
end

# Calculate the image columns of the n-th boundary map
function calculateImageModP(nonDegens, lowerNonDegen, face_list, p::Int)
    cdict = coordDict(lowerNonDegen)
    image = Vector{Any}(undef, length(nonDegens))

    @threads for i in eachindex(nonDegens)
        image[i] = boundarySumModP(nonDegens[i][1][1], cdict, face_list, p)
    end

    return image
end

# Convert sparse coordinate columns to a sparse integer matrix
function columns_to_sparse(columns, number_rows::Int)
    row_indices = Int[]
    column_indices = Int[]
    values = Int[]

    for (column_index, column) in enumerate(columns)
        for k in eachindex(column[1])
            push!(row_indices, column[1][k])
            push!(column_indices, column_index)
            push!(values, column[2][k])
        end
    end

    return sparse(row_indices, column_indices, values, number_rows, length(columns))
end

# Generate the matrix representing the n-th boundary map
function boundaryMapMatrixSparseModP(nonDegens, lowerNonDegens, face_list, p::Int)
    image = calculateImageModP(nonDegens, lowerNonDegens, face_list, p)
    return columns_to_sparse(image, length(lowerNonDegens))
end

# Generate the (n+1)-st boundary matrix directly, without storing (n+1)-cubes.
function graphMapsMatrixModP(A, lowerNonDegen, facesList, nhoodDict, EQClist, p::Int)
    cdict = coordDict(lowerNonDegen)
    thread_results = [Any[] for _ in 1:nthreads()]

    @threads :static for i in eachindex(A)
        local_columns = thread_results[threadid()]
        representative = A[i][1]

        for j in i:length(A)
            for signed_map in A[j]
                if isPairNcube(representative, signed_map, nhoodDict)
                    new_map = vcat(representative[1], signed_map[1])

                    if !checkSemiDegen(new_map, EQClist)
                        push!(local_columns, boundarySumModP(new_map, cdict, facesList, p))
                    end
                end
            end
        end
    end

    columns = reduce(vcat, thread_results; init=Any[])
    return columns_to_sparse(columns, length(lowerNonDegen))
end

# Compute the dimension of the n-th discrete cubical homology group over F_p, where p is the smallest prime greater than n+1
function discrete_homology(G::graph, n::Int; preprocess::Bool=false)
    n >= 0 || error("dimension must be at least 0")

    working_graph = preprocess ? preprocess_graph(G) : G
    p = next_prime_after(n + 1)

    EQCdict = Dict{Int,Vector{Any}}()
    faceDict = Dict{Int,Vector{Int}}()

    for i in 1:n+1
        EQCdict[i] = generateEQClist(i)
    end
    for i in 0:n+1
        faceDict[i] = faceList(i)
    end

    nhoodDict = get_nhood_dict(working_graph)
    maps = Dict{Int,Vector{Any}}()
    maps[-1] = Any[]
    maps[0] = graphMaps(Any[], working_graph, Any[], nhoodDict)

    for i in 1:n
        maps[i] = remove_duplicates(graphMaps(maps[i - 1], working_graph, EQCdict[i], nhoodDict))
    end

    nonDegen = semiNonDegen(maps[n])
    lowerNonDegen = n == 0 ? Any[] : semiNonDegen(maps[n - 1])

    if n == 0
        dimKer = length(nonDegen)
    else
        delN = boundaryMapMatrixSparseModP(nonDegen, lowerNonDegen, faceDict[n], p)
        dimKer = size(delN, 2) - ref_rank!(delN, p, true)
    end

    delN1 = graphMapsMatrixModP(maps[n], nonDegen, faceDict[n + 1], nhoodDict, EQCdict[n + 1], p)
    dimIM = ref_rank!(delN1, p, true)

    return dimKer - dimIM
end

#===================================================================#
# Example graphs
#===================================================================#

# C5
vertex = Any[0, 1, 2, 3, 4]
edges = makeSymmetricReflexive([(0,1), (1,2), (2,3), (3,4), (4,0)])
C5 = graph(vertex, edges)

# Greene sphere
greeneSph = graph(Any[1,2,3,4,5,6,7,8,9,10],
    [(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10),
     (1,2),(2,1),(1,3),(3,1),(1,4),(4,1),(1,5),(5,1),
     (2,6),(6,2),(2,7),(7,2),(3,6),(6,3),(3,8),(8,3),
     (4,7),(7,4),(4,9),(9,4),(5,8),(8,5),(5,9),(9,5),
     (10,6),(6,10),(10,7),(7,10),(10,8),(8,10),(10,9),(9,10)])

# 3D torus example
vert = Any[0,1,2,3,4,5]
edge = makeSymmetricReflexive([(0,1),(1,2),(2,3),(3,4),(4,5)])
I5 = graph(vert, edge)

G_torus = multBoxProd([I5, I5, I5])
for i in 0:5, j in 0:5
    global G_torus = quotient_vertices(G_torus, (0,i,j), (5,i,j))
end
for i in 0:4, j in 0:5
    global G_torus = quotient_vertices(G_torus, (i,0,j), (i,5,j))
end
for i in 0:4, j in 0:4
    global G_torus = quotient_vertices(G_torus, (i,j,0), (i,j,5))
end
T3 = relabel_vertices(G_torus)

# K10
K10 = completeGraph(10)

# C5 with a triangle attached to each edge
v_star = Any[0,1,2,3,4,5,6,7,8,9]
e_star = makeSymmetricReflexive([
    (0,1),(1,2),(2,3),(3,4),(4,0),
    (5,0),(5,1),(6,1),(6,2),(7,2),(7,3),(8,3),(8,4),(9,4),(9,0)
])
C5_star = relabel_vertices(graph(v_star, e_star))

# RP2
vertices_RP2 = Any[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
edges_RP2 = Any[(1, 1), (1, 2), (1, 6), (2, 1), (2, 2), (2, 3), (2, 7), (2, 25), (3, 2), (3, 3), (3, 4), (3, 8), (3, 24),
    (4, 3), (4, 4), (4, 5), (4, 9), (4, 23), (5, 4), (5, 5), (5, 10), (5, 22), (6, 1), (6, 6), (6, 7), (6, 11), (6, 25),
    (7, 2), (7, 6), (7, 7), (7, 8), (7, 12), (8, 3), (8, 7), (8, 8), (8, 9), (8, 13), (9, 4), (9, 8), (9, 9), (9, 10),
    (9, 14), (10, 5), (10, 9), (10, 10), (10, 15), (10, 21), (11, 6), (11, 11), (11, 12), (11, 16), (11, 20), (12, 7),
    (12, 11), (12, 12), (12, 13), (12, 17), (13, 8), (13, 12), (13, 13), (13, 14), (13, 18), (14, 9), (14, 13), (14, 14),
    (14, 15), (14, 19), (15, 10), (15, 14), (15, 15), (15, 16), (15, 20), (16, 11), (16, 15), (16, 16), (16, 17), (16, 21),
    (17, 12), (17, 16), (17, 17), (17, 18), (17, 22), (18, 13), (18, 17), (18, 18), (18, 19), (18, 23), (19, 14), (19, 18),
    (19, 19), (19, 20), (19, 24), (20, 11), (20, 15), (20, 19), (20, 20), (20, 25), (21, 10), (21, 16), (21, 21), (21, 22),
    (22, 5), (22, 17), (22, 21), (22, 22), (22, 23), (23, 4), (23, 18), (23, 22), (23, 23), (23, 24), (24, 3), (24, 19),
    (24, 23), (24, 24), (24, 25), (25, 2), (25, 6), (25, 20), (25, 24), (25, 25)]
RP2 = graph(vertices_RP2, edges_RP2)

# Example usage:
#@time discrete_homology(T3, 2)

