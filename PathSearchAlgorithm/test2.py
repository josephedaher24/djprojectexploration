from BestPath import mix

# Adjust these if you want shorter runs while testing
# If your BestPath.py defines NUM_SONGS globally, these tests assume it is 10.

def print_result(name, result, arousal_function=None):
    cost_val, path = result
    print(f"\n{name}")
    print("Cost:", cost_val)
    print("Path:", path)
    if arousal_function is not None and path:
        print("Arousal along path:", [round(arousal_function(s), 3) for s in path])


# ---------------------------
# Test 6: decreasing curve
# ---------------------------
songs = ["A", "B", "C", "D"]
arousal_map = {"A": 0.1, "B": 0.4, "C": 0.7, "D": 0.95}

def arousal_function(song):
    return arousal_map[song]

def mix_all(s1, s2):
    return 1.0

def decreasing_curve(t):
    return 1 - t

print_result(
    "Test 6: decreasing curve",
    mix(songs, mix_all, arousal_function, decreasing_curve),
    arousal_function
)


# ---------------------------
# Test 7: valley-shaped curve
# ---------------------------
def valley_curve(t):
    return 4 * (t - 0.5) ** 2   # high-low-high, in [0,1]

print_result(
    "Test 7: valley-shaped curve",
    mix(songs, mix_all, arousal_function, valley_curve),
    arousal_function
)


# ---------------------------
# Test 8: only one valid full chain
# ---------------------------
songs = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
arousal_map = {s: i / 9 for i, s in enumerate(songs)}

def arousal_function(song):
    return arousal_map[song]

allowed_chain = {(songs[i], songs[i+1]) for i in range(len(songs)-1)}

def mix_chain(s1, s2):
    return 1.0 if (s1, s2) in allowed_chain else 0.0

def increasing_curve(t):
    return t

print_result(
    "Test 8: one valid chain of length 10",
    mix(songs, mix_chain, arousal_function, increasing_curve),
    arousal_function
)


# ---------------------------
# Test 9: disconnected components
# ---------------------------
songs = ["L1", "L2", "H1", "H2"]
arousal_map = {"L1": 0.1, "L2": 0.2, "H1": 0.8, "H2": 0.9}

def arousal_function(song):
    return arousal_map[song]

def mix_two_components(s1, s2):
    low = {"L1", "L2"}
    high = {"H1", "H2"}
    if s1 in low and s2 in low and s1 != s2:
        return 1.0
    if s1 in high and s2 in high and s1 != s2:
        return 1.0
    return 0.0

def high_curve(t):
    return 0.85

print_result(
    "Test 9: disconnected components, should stay in high group",
    mix(songs, mix_two_components, arousal_function, high_curve),
    arousal_function
)


# ---------------------------
# Test 10: one song has no outgoing edges
# ---------------------------
songs = ["A", "B", "C"]
arousal_map = {"A": 0.2, "B": 0.5, "C": 0.9}

def arousal_function(song):
    return arousal_map[song]

def mix_dead_end(s1, s2):
    allowed = {
        ("A", "B"),
        ("B", "A"),
        # C is a dead end: no outgoing edges
    }
    return 1.0 if (s1, s2) in allowed else 0.0

def mid_curve(t):
    return 0.45

print_result(
    "Test 10: dead-end song should be avoided except maybe at end",
    mix(songs, mix_dead_end, arousal_function, mid_curve),
    arousal_function
)


# ---------------------------
# Test 11: impossible after a few steps
# ---------------------------
songs = ["A", "B", "C"]

def arousal_function(song):
    return {"A": 0.1, "B": 0.5, "C": 0.9}[song]

def mix_short_chain(s1, s2):
    allowed = {
        ("A", "B"),
        ("B", "C"),
    }
    return 1.0 if (s1, s2) in allowed else 0.0

print_result(
    "Test 11: no path of full required length, should be inf/[]",
    mix(songs, mix_short_chain, arousal_function, increasing_curve),
    arousal_function
)


# ---------------------------
# Test 12: tie case
# ---------------------------
songs = ["A", "B"]
arousal_map = {"A": 0.5, "B": 0.5}

def arousal_function(song):
    return arousal_map[song]

def mix_bidirectional(s1, s2):
    return 1.0 if s1 != s2 else 0.0

def flat_curve(t):
    return 0.5

print_result(
    "Test 12: tie case, many optimal paths",
    mix(songs, mix_bidirectional, arousal_function, flat_curve),
    arousal_function
)


# ---------------------------
# Test 13: asymmetric mixability
# ---------------------------
songs = ["A", "B", "C"]
arousal_map = {"A": 0.1, "B": 0.5, "C": 0.9}

def arousal_function(song):
    return arousal_map[song]

def mix_asymmetric(s1, s2):
    allowed = {
        ("A", "B"),
        ("B", "C"),
        ("C", "B"),   # asymmetric structure
    }
    return 1.0 if (s1, s2) in allowed else 0.0

print_result(
    "Test 13: asymmetric graph",
    mix(songs, mix_asymmetric, arousal_function, increasing_curve),
    arousal_function
)


# ---------------------------
# Test 14: custom target that favors alternating levels
# ---------------------------
songs = ["L", "M", "H"]
arousal_map = {"L": 0.1, "M": 0.5, "H": 0.9}

def arousal_function(song):
    return arousal_map[song]

def mix_all(s1, s2):
    return 1.0

def zigzag_curve(t):
    # crude alternating preference over 10 slots
    vals = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
    idx = min(int(t * 10), 9)
    return vals[idx]

print_result(
    "Test 14: zigzag target",
    mix(songs, mix_all, arousal_function, zigzag_curve),
    arousal_function
)