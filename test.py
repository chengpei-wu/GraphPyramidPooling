a = [
    [5, 4, 3],
    [5, 5, 4],
    [4, 4, 4],
    [4, 3, 4],
    [4, 4, 5],
    [3, 2, 1],
    [5, 4, 4]
]

print(sorted(a, key=lambda x: (x[0], x[1], x[2]), reverse=True))
