#!/usr/bin/env python3
import sys
import math

ABS_TOL = 1e-6
REL_TOL = 1e-5
MIN_DEN = 1e-12

def read_floats(path):
    with open(path, 'r') as f:
        data = f.read().strip().split()
        return [float(x) for x in data]

def close(a, b):
    diff = abs(a - b)
    if diff <= ABS_TOL:
        return True
    denom = max(MIN_DEN, abs(b))
    return diff / denom <= REL_TOL

def main():
    if len(sys.argv) != 3:
        print("usage: verify.py <my_output> <golden_output>")
        sys.exit(2)
    a = read_floats(sys.argv[1])
    b = read_floats(sys.argv[2])
    if len(a) != len(b):
        print("length mismatch", len(a), len(b))
        sys.exit(1)
    for i, (x, y) in enumerate(zip(a, b)):
        if not close(x, y):
            print("mismatch at", i, x, y)
            sys.exit(1)
    print("OK")

if __name__ == '__main__':
    main()


