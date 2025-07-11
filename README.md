# Collatz Conjecture Tester

Fast C++ implementation for testing the Collatz conjecture on single numbers or ranges.

## The Collatz Conjecture

For any positive integer n, repeatedly apply:
- If n is even: n → n/2
- If n is odd: n → 3n + 1

The conjecture states that this sequence always reaches 1.

## Build

```bash
# Single-threaded version
g++ -O3 -std=c++11 collatz.cpp -o collatz

# With OpenMP for parallel processing
g++ -O3 -std=c++11 -fopenmp collatz.cpp -o collatz
```

## Usage

```bash
# Test single number
./collatz 27

# Test range
./collatz 1 1000000

# Test range with sample sequences
./collatz 1 1000000 --sample
```

## Features

- Efficient caching to avoid recomputation
- OpenMP parallelization for range testing
- Tracks peak values and growth ratios
- Overflow protection for large sequences
- Progress reporting for large ranges

## License

MIT License
