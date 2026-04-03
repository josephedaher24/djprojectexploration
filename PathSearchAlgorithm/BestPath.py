# Given inputs of:
#     - a set of songs
#     - an arousal function that maps song to normalized value
#     - an arousal curve (defined over time) that maps [0,1] to target arousal shape,
#     - a mixeability function that maps pairs of songs to a normalized value
# The algorithm will find the best path through the songs that 
# have mixeability values exceeding a certain threshold, and
# arousal values that fit the shape given.

MIX_THRESHOLD = 0.6
NUM_SONGS = 4

# creates a graph of songs, where edges exists if and only 
# the mixeability value between the two songs exceed a certain target
def makeGraph(songs, mixeability_function):
    graph = {}
    for song in songs:
        graph[song] = []
        for i in range(len(songs)):
            if mixeability_function(song, songs[i]) > MIX_THRESHOLD:
                if songs[i] != song:
                    graph[song].append(songs[i])
                    
                    
    # make new graph with vertices (song, k), where k is the position in the path
    # edge (song1, k) to (song2, k+1) exists if and only if the edge (song1, song2) exists in the original graph
    new_graph = {}
    reverse = {}
    
    for song in songs:
        for k in range(NUM_SONGS):
            reverse[(song, k)] = []
        
    for song in graph:
        for k in range(NUM_SONGS):
            new_graph[(song, k)] = []
            if k < NUM_SONGS - 1:
                for neighbor in graph[song]:
                    new_graph[(song, k)].append((neighbor, k+1))
                    reverse[(neighbor, k+1)] = reverse.get((neighbor, k+1), []) + [(song, k)]
                    
    return new_graph, reverse
    

def cost(song, target_arousal, arousal_function):
    return (arousal_function(song) - target_arousal) ** 2

    


# finds best path through the graph of songs, where arousal
# values fit the target arousal curve
def BestPath(graph, pred, arousal_function, arousal_curve, songs, NUM_SONGS):
    # standardize arousal curve to NUM_SONGS timesteps, and match max,min to 
    # max, min of arousal of songs
    # We use dynamic programming to find the best path, starting from 
    # any song and ending in any song with NUM_SONGS songs. 
    # We let DP(k, i) denote the min cost of a path of length k
    # ending in song i, where cost is defined as the squared deviation 
    # of the arousal value of song i from teh target arousal value.
    
    # We construct a new graph with vertices (song, k), so we can have
    # an explicit cost calculation for each state. We have an edge from (song1, k) to (song2, k+1) if and only if
    # song2 is mixable with song1, and we have a cost of cost(song2, target(k), arousal_function) for the vertex (song2, k+1)
    
    NUM_SONGS = min(NUM_SONGS, len(songs))
    
    target = []
    for i in range(NUM_SONGS):
        target.append(arousal_curve(i / NUM_SONGS))
    
    
    DP = {}
    parent = {}
    
    for k in range(NUM_SONGS):
        for song in songs:
            node = (song, k)
            if k == 0:
                DP[node] = cost(song, target[k], arousal_function)
                parent[node] = None
            else:
                DP[node] = float('inf')
                parent[node] = None
                for pre in pred.get(node, []):
                    if DP[pre] + cost(song, target[k], arousal_function) < DP[node]:
                        DP[node] = DP[pre] + cost(song, target[k], arousal_function)
                        parent[node] = pre
    
    # find min cost path from any (song, 0) to any (song, NUM_SONGS - 1)
    min_cost = float('inf')
    min_song = None
    for song in songs:
        if (song, NUM_SONGS - 1) in DP:
            if DP[(song, NUM_SONGS - 1)] < min_cost:
                min_cost = DP[(song, NUM_SONGS - 1)]
                min_song = song

            
    path = []
    node = (min_song, NUM_SONGS - 1)
    
    while node is not None and node[0] is not None:
        path.append(node[0])
        node = parent[node]
        
    path.reverse()
            
    return min_cost, path



def mix(songs, mixeability_function, arousal_function, arousal_curve, NUM_SONGS = NUM_SONGS):
    graph, reverse = makeGraph(songs, mixeability_function)
    return BestPath(graph, reverse, arousal_function, arousal_curve, songs, NUM_SONGS)


