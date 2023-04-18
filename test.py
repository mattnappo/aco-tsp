import collections

with open("sample_test.txt") as f:
    samples = [int(x) for x in f.readlines()]
    n = len(samples)
    samples = collections.Counter(samples)
    vals = list(samples.keys())
    vals.sort()
    p = { k: samples[k]/n for k in vals }
    print(p)
