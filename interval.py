def find_intervals(lst):
    intervals = []
    start = None
    for i, val in enumerate(lst):
        if val == 1:
            if start is None:
                start = i
        elif start is not None:
            intervals.append((start, i-1))
            start = None
    if start is not None:
        intervals.append((start, len(lst)-1))
    return intervals

# Example usage
lst = [0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1]
intervals = find_intervals(lst)
for interval in intervals:
    print(f"Start: {interval[0]}, End: {interval[1]}")
