from BestPath import mix

songs = ["A", "B", "C", "D"]

# fixed arousal values
arousal_map = {
    "A": 0.1,
    "B": 0.4,
    "C": 0.7,
    "D": 0.9
}

def arousal_function(song):
    return arousal_map[song]

# allow all transitions
def mixeability_function(s1, s2):
    return 1.0

# increasing target curve
def arousal_curve(t):
    return t  # linear from 0 to 1

print("Test 1:")
cost_val, path = mix(songs, mixeability_function, arousal_function, arousal_curve)
print("Cost:", cost_val)
print("Path:", path)


def mixeability_function_blocked(s1, s2):
    return 0.0  # nothing allowed

print("\nTest 2:")
cost_val, path = mix(songs, mixeability_function_blocked, arousal_function, arousal_curve)
print("Cost:", cost_val)
print("Path:", path)


def mixeability_chain(s1, s2):
    allowed = {
        ("A", "B"),
        ("B", "C"),
        ("C", "D")
    }
    return 1.0 if (s1, s2) in allowed else 0.0

print("\nTest 3:")
cost_val, path = mix(songs, mixeability_chain, arousal_function, arousal_curve)
print("Cost:", cost_val)
print("Path:", path)

def flat_curve(t):
    return 0.5

print("\nTest 4:")
cost_val, path = mix(songs, mixeability_function, arousal_function, flat_curve)
print("Cost:", cost_val)
print("Path:", path)


import random

songs = [f"S{i}" for i in range(8)]

arousal_map = {s: random.random() for s in songs}

def arousal_function(song):
    return arousal_map[song]

def mixeability_function(s1, s2):
    return random.random()

def arousal_curve(t):
    return 0.5 + 0.4 * (t - 0.5)  # mild slope

print("\nTest 5:")
cost_val, path = mix(songs, mixeability_function, arousal_function, arousal_curve)
print("Cost:", cost_val)
print("Path:", path)