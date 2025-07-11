#ifndef COLLATZ_H
#define COLLATZ_H

#include <vector>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <climits>
#include <string>
#include <sstream>
#include <mutex>
#include <array>
#include <functional>
#include <atomic>
#include <cerrno>

// Cache entry structure for steps and peak values
struct Entry {
    uint32_t steps;
    uint64_t peak;
    bool overflow;  // Mark if computation hit overflow
    Entry() : steps(0), peak(0), overflow(false) {}
    Entry(uint32_t s, uint64_t p, bool o = false) : steps(s), peak(p), overflow(o) {}
};

// Structure to store results for each number (backward compatibility)
struct CollatzResult {
    std::vector<long long> sequence;
    int steps;
    long long peak;
    double growthFactor;
    
    CollatzResult() : steps(0), peak(0), growthFactor(0.0) {}
};

// Structure for aggregated statistics
struct CollatzStats {
    std::atomic<long long> totalNumbers;
    long long totalSteps;
    int maxSteps;
    long long numberWithMaxSteps;
    long long maxPeak;
    long long numberWithMaxPeak;
    double maxGrowthRatio;
    long long numberWithMaxGrowth;
    
    CollatzStats() : totalNumbers(0), totalSteps(0), maxSteps(0), 
                     numberWithMaxSteps(0), maxPeak(0), numberWithMaxPeak(0),
                     maxGrowthRatio(0.0), numberWithMaxGrowth(0) {}
    
    // Copy constructor - needed because atomic cannot be copied
    CollatzStats(const CollatzStats& other) 
        : totalNumbers(other.totalNumbers.load()), totalSteps(other.totalSteps),
          maxSteps(other.maxSteps), numberWithMaxSteps(other.numberWithMaxSteps),
          maxPeak(other.maxPeak), numberWithMaxPeak(other.numberWithMaxPeak),
          maxGrowthRatio(other.maxGrowthRatio), numberWithMaxGrowth(other.numberWithMaxGrowth) {}
    
    // Move constructor
    CollatzStats(CollatzStats&& other) noexcept
        : totalNumbers(other.totalNumbers.load()), totalSteps(other.totalSteps),
          maxSteps(other.maxSteps), numberWithMaxSteps(other.numberWithMaxSteps),
          maxPeak(other.maxPeak), numberWithMaxPeak(other.numberWithMaxPeak),
          maxGrowthRatio(other.maxGrowthRatio), numberWithMaxGrowth(other.numberWithMaxGrowth) {}
    
    // Copy assignment operator
    CollatzStats& operator=(const CollatzStats& other) {
        if (this != &other) {
            totalNumbers.store(other.totalNumbers.load());
            totalSteps = other.totalSteps;
            maxSteps = other.maxSteps;
            numberWithMaxSteps = other.numberWithMaxSteps;
            maxPeak = other.maxPeak;
            numberWithMaxPeak = other.numberWithMaxPeak;
            maxGrowthRatio = other.maxGrowthRatio;
            numberWithMaxGrowth = other.numberWithMaxGrowth;
        }
        return *this;
    }
    
    // Move assignment operator
    CollatzStats& operator=(CollatzStats&& other) noexcept {
        if (this != &other) {
            totalNumbers.store(other.totalNumbers.load());
            totalSteps = other.totalSteps;
            maxSteps = other.maxSteps;
            numberWithMaxSteps = other.numberWithMaxSteps;
            maxPeak = other.maxPeak;
            numberWithMaxPeak = other.numberWithMaxPeak;
            maxGrowthRatio = other.maxGrowthRatio;
            numberWithMaxGrowth = other.numberWithMaxGrowth;
        }
        return *this;
    }
    
    double getAverageSteps() const {
        long long total = totalNumbers.load();
        return total > 0 ? static_cast<double>(totalSteps) / total : 0.0;
    }
};

class CollatzTester {
private:
    // Single mutex for thread-safe cache access
    static std::mutex cacheMutex;
    static std::unordered_map<uint64_t, Entry> stepCache;
    
    // Optimized step counting without storing sequence
    static int getStepCountOptimized(long long n);
    static Entry getStepCountOptimizedWithPeak(long long n);
    
    // Parse command line arguments
    static bool parseArguments(int argc, char* argv[], long long& N1, long long& N2, bool& sampleMode);
    
    // Print usage information
    static void printUsage(const char* programName);
    

    
public:
    // Cache initialization
    static void initializeCache(size_t expectedSize);
    
    // Core functions
    static CollatzResult calculateCollatz(long long n, bool storeSequence = true);
    static int getStepCount(long long n);
    static Entry getStepAndPeak(long long n);  // Get both steps and peak
    static size_t getCacheSize();
    static void clearCache(); // Manual cache clearing if needed
    static void printResult(long long n, const CollatzResult& result);
    
    // Main optimized functions
    static CollatzStats testRangeOptimized(long long N1, long long N2, bool sampleMode = false);
    static void printOptimizedSummary(const CollatzStats& stats, long long N1, long long N2, double elapsedTime);
    static void runOptimizedVersion(int argc, char* argv[]);
};

#endif // COLLATZ_H