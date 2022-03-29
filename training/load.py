import numpy as np

# load fens
with open('fens.txt','r') as f:
    fens = f.readlines()
    fens = [fen[:-1] for fen in fens]

print(fens)

# load turns

turns = np.loadtxt('turns.txt', delimiter='\n', dtype=np.int)
print(turns)

# load values

values = np.loadtxt('values.txt', delimiter='\n', dtype=np.int)
print(values)

# load avs
with open('avs.txt','r') as a:
    avs_str = a.readlines()
    avs_str = [avs.split(sep=", ") for avs in avs_str]

    avs_fl = [np.array(avs, dtype=np.float64) for avs in avs_str]
avs_arr = np.array(avs_fl)

# print(type(avs_arr))
# print(type(avs_arr[1]))
# print(type(avs_arr[1][1]))
# print(max(avs_arr[2]))



