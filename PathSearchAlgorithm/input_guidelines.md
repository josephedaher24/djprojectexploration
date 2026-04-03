## 1. Songs

A list/collection of songs  
Type: iterable (e.g., list)  

Each song should be:
- unique  
- hashable (usable as a dictionary key)  
- non-empty  


## 2. arousal_function(song) -> float

Returns a numeric measure of a song’s energy.  

Requirements:
- Return a finite number for every song  
- Deterministic  
- Should reflect meaningful differences between songs  

Notes:
- No need to normalize values  
- No fixed range required (handled internally)  

Invalid outputs:
- NaN  
- inf or -inf  


## 3. arousal_curve(t: float) -> float

Defines the desired energy profile of the sequence.  

Input: t ∈ [0, 1]  
Output: numeric value  

Requirements:
- Return finite numbers  
- Should encode the intended shape (e.g. increasing, peak, etc.)  

Notes:
- Scale does not matter  
- Only relative shape is used (internally normalized)  


## 4. mixeability_function(song1, song2) -> float

Scores how well song1 transitions into song2 (not the other way round)  

Requirements:
- Return a finite number between 0 and 1 inclusive for every pair of songs  
- Higher values = better transitions  
- Consistent  

Important:
- The algorithm uses a threshold (MIX_THRESHOLD = 0.6) to only allow transitions of above 0.6  
- This threshold can be changed
