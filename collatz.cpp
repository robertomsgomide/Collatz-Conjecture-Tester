#include "collatz.h"
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

// Initialize static members
std::mutex CollatzTester::cacheMutex;
std::unordered_map<uint64_t, Entry> CollatzTester::stepCache;

void CollatzTester::initializeCache(size_t expectedSize) {
    std::lock_guard<std::mutex> lock(cacheMutex);
    // Reserve 2x to account for intermediate values (~1.7x empirically)
    stepCache.reserve(2 * expectedSize);
    stepCache.max_load_factor(0.75f);
}

int CollatzTester::getStepCountOptimized(long long n) {
    if (n <= 1) return 0;
    
    uint64_t key = static_cast<uint64_t>(n);
    
    // Check cache first with thread-safe lookup
    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        auto it = stepCache.find(key);
        if (it != stepCache.end()) {
            if (it->second.overflow) {
                std::cerr << "Error: Input " << n << " causes overflow, computation unsupported" << std::endl;
                return -1;  // Return error code for overflow
            }
            return static_cast<int>(it->second.steps);
        }
    }
    
    long long original = n;
    long long peak = n;
    int steps = 0;
    
    while (n != 1) {
        // Check cache for current value
        uint64_t currentKey = static_cast<uint64_t>(n);
        
        {
            std::lock_guard<std::mutex> lock(cacheMutex);
            auto cached = stepCache.find(currentKey);
            if (cached != stepCache.end()) {
                steps += static_cast<int>(cached->second.steps);
                peak = std::max(peak, static_cast<long long>(cached->second.peak));
                break;
            }
        }
        
        if (n % 2 == 0) {
            n = n >> 1;  // Divide by 2
            steps++;
        } else {
            // Exact peak tracking: compute tmp = 3*n + 1, track peak, then optimize
            if (n > (LLONG_MAX - 1) / 3) {
                // Overflow detected - refuse to cache wrong result
                std::cerr << "Error: Input " << original << " causes overflow, computation unsupported" << std::endl;
                return -1;  // Return error code for overflow
            } else {
                long long tmp = 3 * n + 1;
                peak = std::max(peak, tmp);
                n = tmp >> 1;
                steps += 2;
            }
        }
        
        // Safety check - longest known 64-bit trajectory is >20,000 steps
        if (steps > 100000) {
            std::cerr << "Error: Input " << original << " exceeds step limit, computation unsupported" << std::endl;
            return -1;  // Return error code for step limit
        }
    }
    
    // Cache the result with thread-safe insertion (only if computation completed)
    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        stepCache.emplace(key, Entry(static_cast<uint32_t>(steps), static_cast<uint64_t>(peak)));
    }
    
    return steps;
}

bool CollatzTester::parseArguments(int argc, char* argv[], long long& N1, long long& N2, bool& sampleMode) {
    if (argc < 2) {
        return false;
    }
    
    N1 = 0;
    N2 = 0;
    sampleMode = false;
    std::vector<long long> numbers;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sample") == 0) {
            sampleMode = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            return false;
        } else {
            // Try to parse as number
            char* endptr;
            errno = 0;  // Clear errno before call
            long long val = strtoll(argv[i], &endptr, 10);
            if (*endptr == '\0' && val > 0 && errno != ERANGE) {
                numbers.push_back(val);
            }
        }
    }
    
    if (numbers.size() == 1) {
        // Single number: test just that number N
        N1 = numbers[0];
        N2 = numbers[0];
    } else if (numbers.size() == 2) {
        // Two numbers: test from N1 to N2
        N1 = std::min(numbers[0], numbers[1]);
        N2 = std::max(numbers[0], numbers[1]);
    } else {
        return false;
    }
    
    return N1 > 0 && N2 > 0 && N1 <= N2;
}

void CollatzTester::printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " N [--sample]           (test just number N)\n";
    std::cout << "       " << programName << " N1 N2 [--sample]       (test range N1 to N2)\n";
    std::cout << "  N        : Test just the specific number N\n";
    std::cout << "  N1 N2    : Test numbers from N1 to N2 (N1 <= N2)\n";
    std::cout << "  --sample : Store and display sequences for first/last 5 numbers (ranges only)\n";
    std::cout << "  --help   : Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << programName << " 27                    (test just number 27)\n";
    std::cout << "  " << programName << " 1 1000000 --sample    (test 1 to 1,000,000)\n";
    std::cout << "  " << programName << " 500000 1000000       (test 500,000 to 1,000,000)\n";
}



CollatzStats CollatzTester::testRangeOptimized(long long N1, long long N2, bool sampleMode) {
    CollatzStats stats;
    
    // Sample results for first and last few numbers if requested
    std::vector<CollatzResult> sampleResults;
    const int SAMPLE_SIZE = 5;
    
    if (sampleMode) {
        sampleResults.reserve(SAMPLE_SIZE * 2);
    }
    
    // Initialize cache with expected size for better performance
    initializeCache(N2 - N1 + 1);
    
    // For single number, just show the sequence
    if (N1 == N2) {
        CollatzResult result = calculateCollatz(N1, true);
        printResult(N1, result);
        return stats;  // Early return for single number
    }
    
    // Progress reporting for range computations
    long long rangeSize = N2 - N1 + 1;
    bool showProgress = (rangeSize >= 1000);
    long long progressStep = showProgress ? std::max(1LL, rangeSize / 100) : 0; // 1% intervals
    long long nextProgress = progressStep;
    
    if (showProgress) {
        std::cout << "Computing." << std::flush;
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Use OpenMP if available
#ifdef _OPENMP
    #pragma omp parallel
    {
        // Thread-local variables
        CollatzStats localStats;
        std::vector<CollatzResult> localSamples;
        
        #pragma omp for schedule(dynamic, 1000)
        for (long long i = N1; i <= N2; i++) {  // Test all numbers in range
            int steps;
            long long peak;
            
            // Get exact steps and peak from cache-enabled computation
            if (sampleMode && ((i <= N1 + SAMPLE_SIZE - 1) || (i > N2 - SAMPLE_SIZE))) {
                CollatzResult result = calculateCollatz(i, true);
                if (result.steps == 0 && result.peak == 0) {
                    // Skip overflow inputs in sample mode too
                    continue;
                }
                localSamples.push_back(result);
                steps = result.steps;
                peak = result.peak;
            } else {
                Entry entry = getStepAndPeak(i);
                if (entry.overflow) {
                    // Skip unsupported inputs
                    continue;
                }
                steps = static_cast<int>(entry.steps);
                peak = static_cast<long long>(entry.peak);
            }
            
            // Update local statistics
            localStats.totalNumbers.fetch_add(1);
            localStats.totalSteps += steps;
            
            if (steps > localStats.maxSteps) {
                localStats.maxSteps = steps;
                localStats.numberWithMaxSteps = i;
            }
            
            if (peak > localStats.maxPeak) {
                localStats.maxPeak = peak;
                localStats.numberWithMaxPeak = i;
            }
            
            double growthRatio = static_cast<double>(peak) / static_cast<double>(i);
            if (growthRatio > localStats.maxGrowthRatio) {
                localStats.maxGrowthRatio = growthRatio;
                localStats.numberWithMaxGrowth = i;
            }
        }
        
        // Merge local results into global stats
        #pragma omp critical
        {
            stats.totalNumbers.fetch_add(localStats.totalNumbers.load());
            stats.totalSteps += localStats.totalSteps;
            
            if (localStats.maxSteps > stats.maxSteps) {
                stats.maxSteps = localStats.maxSteps;
                stats.numberWithMaxSteps = localStats.numberWithMaxSteps;
            }
            
            if (localStats.maxPeak > stats.maxPeak) {
                stats.maxPeak = localStats.maxPeak;
                stats.numberWithMaxPeak = localStats.numberWithMaxPeak;
            }
            
            if (localStats.maxGrowthRatio > stats.maxGrowthRatio) {
                stats.maxGrowthRatio = localStats.maxGrowthRatio;
                stats.numberWithMaxGrowth = localStats.numberWithMaxGrowth;
            }
            
            if (sampleMode) {
                for (const auto& sample : localSamples) {
                    sampleResults.push_back(sample);
                }
            }
        }
        
        // Progress reporting (single thread to free up workers)
        #pragma omp single
        {
            if (showProgress) {
                long long completed = stats.totalNumbers.load();
                while (nextProgress <= completed && nextProgress <= rangeSize) {
                    std::cout << "." << std::flush;
                    nextProgress += progressStep;
                }
            }
        }
    }
#else
    // Single-threaded version
    for (long long i = N1; i <= N2; i++) {  // Test all numbers in range
        int steps;
        long long peak;
        
        // Get exact steps and peak from cache-enabled computation
        if (sampleMode && ((i <= N1 + SAMPLE_SIZE - 1) || (i > N2 - SAMPLE_SIZE))) {
            CollatzResult result = calculateCollatz(i, true);
            if (result.steps == 0 && result.peak == 0) {
                // Skip overflow inputs in sample mode too
                continue;
            }
            sampleResults.push_back(result);
            steps = result.steps;
            peak = result.peak;
        } else {
            Entry entry = getStepAndPeak(i);
            if (entry.overflow) {
                // Skip unsupported inputs
                continue;
            }
            steps = static_cast<int>(entry.steps);
            peak = static_cast<long long>(entry.peak);
        }
        
        stats.totalNumbers.fetch_add(1);
        stats.totalSteps += steps;
        
        if (steps > stats.maxSteps) {
            stats.maxSteps = steps;
            stats.numberWithMaxSteps = i;
        }
        
        if (peak > stats.maxPeak) {
            stats.maxPeak = peak;
            stats.numberWithMaxPeak = i;
        }
        
        double growthRatio = static_cast<double>(peak) / static_cast<double>(i);
        if (growthRatio > stats.maxGrowthRatio) {
            stats.maxGrowthRatio = growthRatio;
            stats.numberWithMaxGrowth = i;
        }
        
        // Progress reporting for single-threaded
        if (showProgress && stats.totalNumbers.load() >= nextProgress) {
            std::cout << "." << std::flush;
            nextProgress += progressStep;
        }
    }
#endif
    
    if (showProgress) {
        std::cout << " Done!" << std::endl;
    }
    
    // Display sample results if requested
    if (sampleMode && !sampleResults.empty()) {
        std::cout << "\nSample sequences:" << std::endl;
        
        // Sort samples by number
        std::sort(sampleResults.begin(), sampleResults.end(), 
                  [](const CollatzResult& a, const CollatzResult& b) {
                      return a.sequence[0] < b.sequence[0];
                  });
        
        for (const auto& result : sampleResults) {
            if (!result.sequence.empty()) {
                printResult(result.sequence[0], result);
            }
        }
    }
    
    return stats;
}

void CollatzTester::printOptimizedSummary(const CollatzStats& stats, long long N1, long long N2, double elapsedTime) {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "Tested " << N1 << ".." << N2 << " (" << stats.totalNumbers.load() << " numbers)" << std::endl;
    std::cout << "Average steps ........... " << std::fixed << std::setprecision(2) 
              << stats.getAverageSteps() << std::endl;
    std::cout << "Max steps ............... " << stats.maxSteps 
              << "  (at n = " << stats.numberWithMaxSteps << ")" << std::endl;
    std::cout << "Largest peak ............ " << stats.maxPeak 
              << "  (at n = " << stats.numberWithMaxPeak << ")" << std::endl;
    std::cout << "Largest growth ratio .... " << std::fixed << std::setprecision(3)
              << stats.maxGrowthRatio 
              << "  (at n = " << stats.numberWithMaxGrowth << ")" << std::endl;
    std::cout << "Elapsed wall time ....... " << std::fixed << std::setprecision(1)
              << elapsedTime << "s" << std::endl;
    
#ifdef _OPENMP
    std::cout << "Threads used ............ " << omp_get_max_threads() << std::endl;
#endif
    std::cout << "Cache entries ........... " << getCacheSize() << std::endl;
}

void CollatzTester::runOptimizedVersion(int argc, char* argv[]) {
    long long N1, N2;
    bool sampleMode;
    
    if (!parseArguments(argc, argv, N1, N2, sampleMode)) {
        printUsage(argv[0]);
        return;
    }
    
    std::cout << "=== Optimized Collatz Conjecture Tester ===" << std::endl;
    
    if (N1 == N2) {
        // Single number test
        std::cout << "Testing number " << N1;
    } else {
        // Range test
        std::cout << "Testing range " << N1 << " to " << N2;
        if (sampleMode) {
            std::cout << " (with sample sequences)";
        }
    }
    
#ifdef _OPENMP
    if (N1 != N2) {  // Only show threads for range testing
        std::cout << " using " << omp_get_max_threads() << " threads";
    }
#endif
    std::cout << std::endl;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    CollatzStats stats = testRangeOptimized(N1, N2, sampleMode);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    double elapsedSeconds = duration.count() / 1000.0;
    
    printOptimizedSummary(stats, N1, N2, elapsedSeconds);
}

// Main implementation
CollatzResult CollatzTester::calculateCollatz(long long n, bool storeSequence) {
    CollatzResult result;
    
    if (n <= 0) return result;
    
    long long current = n;
    long long peak = n;
    int steps = 0;
    
    if (storeSequence) {
        // Reserve space for sequence - 63-bit numbers can have >1000 elements
        result.sequence.reserve(1100);
        result.sequence.push_back(current);
    }
    
    // Check cache first for optimization
    if (!storeSequence) {
        Entry entry = getStepAndPeak(n);
        if (entry.overflow) {
            std::cerr << "Error: Input " << n << " causes overflow, computation unsupported" << std::endl;
            return result;  // Return empty result
        }
        if (entry.steps > 0) {
            result.steps = static_cast<int>(entry.steps);
            result.peak = static_cast<long long>(entry.peak);
            result.growthFactor = static_cast<double>(result.peak) / static_cast<double>(n);
            return result;
        }
    }
    
    while (current != 1) {
        if (current % 2 == 0) {
            current = current / 2;
        } else {
            if (current > (LLONG_MAX - 1) / 3) {
                std::cerr << "Error: Input " << n << " causes overflow, computation unsupported" << std::endl;
                return result;  // Return empty result, don't cache wrong data
            }
            current = 3 * current + 1;
        }
        
        steps++;
        peak = std::max(peak, current);
        
        if (storeSequence) {
            result.sequence.push_back(current);
        }
        
        if (!storeSequence) {
            uint64_t currentKey = static_cast<uint64_t>(current);
            std::lock_guard<std::mutex> lock(cacheMutex);
            auto it = stepCache.find(currentKey);
            if (it != stepCache.end() && !it->second.overflow) {
                steps += static_cast<int>(it->second.steps);
                peak = std::max(peak, static_cast<long long>(it->second.peak));
                break;
            }
        }
        
        if (steps > 100000) {
            std::cerr << "Error: Input " << n << " exceeds step limit, computation unsupported" << std::endl;
            return result;  // Return empty result, don't cache wrong data
        }
    }
    
    result.steps = steps;
    result.peak = peak;
    
    // Calculate growth factor inline
    result.growthFactor = static_cast<double>(peak) / static_cast<double>(n);
    
    // Cache the result with thread-safe insertion (only cache successful computations)
    if (storeSequence) {  // Only cache when we computed the full sequence
        uint64_t key = static_cast<uint64_t>(n);
        {
            std::lock_guard<std::mutex> lock(cacheMutex);
            stepCache.emplace(key, Entry(static_cast<uint32_t>(steps), static_cast<uint64_t>(peak)));
        }
    }
    return result;
}

int CollatzTester::getStepCount(long long n) {
    int result = getStepCountOptimized(n);
    if (result == -1) {
        std::cerr << "Error: Input " << n << " causes overflow or exceeds limits" << std::endl;
    }
    return result;
}

Entry CollatzTester::getStepCountOptimizedWithPeak(long long n) {
    if (n <= 1) return Entry(0, n);
    
    uint64_t key = static_cast<uint64_t>(n);
    
    // Check cache first with thread-safe lookup
    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        auto it = stepCache.find(key);
        if (it != stepCache.end()) {
            return it->second;  // Return as-is, including overflow flag
        }
    }
    
    long long original = n;
    long long peak = n;
    int steps = 0;
    
    while (n != 1) {
        // Check cache for current value
        uint64_t currentKey = static_cast<uint64_t>(n);
        
        {
            std::lock_guard<std::mutex> lock(cacheMutex);
            auto cached = stepCache.find(currentKey);
            if (cached != stepCache.end()) {
                if (cached->second.overflow) {
                    std::cerr << "Error: Input " << original << " causes overflow, computation unsupported" << std::endl;
                    return Entry(0, 0, true);  // Return overflow marker
                }
                steps += static_cast<int>(cached->second.steps);
                peak = std::max(peak, static_cast<long long>(cached->second.peak));
                break;
            }
        }
        
        if (n % 2 == 0) {
            n = n >> 1;  // Divide by 2
            steps++;
        } else {
            // Exact peak tracking: compute tmp = 3*n + 1, track peak, then optimize
            if (n > (LLONG_MAX - 1) / 3) {
                // Overflow detected - refuse to cache wrong result
                std::cerr << "Error: Input " << original << " causes overflow, computation unsupported" << std::endl;
                return Entry(0, 0, true);  // Return overflow marker
            } else {
                long long tmp = 3 * n + 1;
                peak = std::max(peak, tmp);
                n = tmp >> 1;
                steps += 2;
            }
        }
        
        // Safety check - longest known 64-bit trajectory is >20,000 steps
        if (steps > 100000) {
            std::cerr << "Error: Input " << original << " exceeds step limit, computation unsupported" << std::endl;
            return Entry(0, 0, true);  // Return overflow marker
        }
    }
    
    // Cache the result with thread-safe insertion (only if computation completed)
    Entry result(static_cast<uint32_t>(steps), static_cast<uint64_t>(peak));
    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        stepCache.emplace(key, result);
    }
    
    return result;
}

Entry CollatzTester::getStepAndPeak(long long n) {
    return getStepCountOptimizedWithPeak(n);
}



void CollatzTester::clearCache() {
    std::lock_guard<std::mutex> lock(cacheMutex);
    stepCache.clear();
}

size_t CollatzTester::getCacheSize() {
    return stepCache.size();
}

void CollatzTester::printResult(long long n, const CollatzResult& result) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Number: " << n << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    std::cout << "Sequence: ";
    if (result.sequence.size() <= 20) {
        for (size_t i = 0; i < result.sequence.size(); ++i) {
            std::cout << result.sequence[i];
            if (i < result.sequence.size() - 1) std::cout << " -> ";
        }
    } else {
        for (int i = 0; i < 10; ++i) {
            std::cout << result.sequence[i] << " -> ";
        }
        std::cout << "... -> ";
        for (size_t i = result.sequence.size() - 10; i < result.sequence.size(); ++i) {
            std::cout << result.sequence[i];
            if (i < result.sequence.size() - 1) std::cout << " -> ";
        }
    }
    std::cout << std::endl;
    
    std::cout << "Steps to termination: " << result.steps << std::endl;
    std::cout << "Peak value: " << result.peak << std::endl;
    std::cout << "Growth factor: " << std::fixed << std::setprecision(3) 
              << result.growthFactor << std::endl;
}



int main(int argc, char* argv[]) {
    // Use optimized version if command line arguments are provided
    if (argc > 1) {
        CollatzTester::runOptimizedVersion(argc, argv);
        return 0;
    }
    
    // Fallback to original interactive version
    std::cout << "=== Collatz Conjecture Tester ===" << std::endl;
    std::cout << "Enter a number N (test just N) or two numbers N1 N2 (test range N1 to N2): ";
    
    std::string input;
    std::getline(std::cin, input);
    
    // Parse the input
    std::istringstream iss(input);
    std::vector<long long> numbers;
    long long num;
    
    std::string token;
    while (iss >> token) {
        char* endptr;
        errno = 0;  // Clear errno before call
        long long num = strtoll(token.c_str(), &endptr, 10);
        if (*endptr == '\0' && num > 0 && errno != ERANGE) {
            numbers.push_back(num);
        }
    }
    
    if (numbers.empty()) {
        std::cout << "Invalid input. Please enter positive integer(s)." << std::endl;
        return 1;
    }
    
    long long N1, N2;
    bool isRange = false;
    
    if (numbers.size() == 1) {
        // Single number
        N1 = N2 = numbers[0];
    } else if (numbers.size() == 2) {
        // Range
        N1 = std::min(numbers[0], numbers[1]);
        N2 = std::max(numbers[0], numbers[1]);
        isRange = true;
    } else {
        std::cout << "Please enter either one number or two numbers." << std::endl;
        return 1;
    }
    
    // Warning for large computations
    long long rangeSize = N2 - N1 + 1;
    if (rangeSize > 1000000) {
        std::cout << "Warning: Testing " << rangeSize << " numbers may take a long time." << std::endl;
        std::cout << "Continue? (y/n): ";
        char choice;
        std::cin >> choice;
        if (choice != 'y' && choice != 'Y') {
            return 0;
        }
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    CollatzStats stats = CollatzTester::testRangeOptimized(N1, N2, false);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double elapsedSeconds = duration.count() / 1000.0;
    
    // Print summary for ranges, single numbers already print their own output
    if (isRange) {
        CollatzTester::printOptimizedSummary(stats, N1, N2, elapsedSeconds);
    } else {
        std::cout << "\nExecution time: " << std::fixed << std::setprecision(1) 
                  << elapsedSeconds << "s" << std::endl;
        std::cout << "Cache entries: " << CollatzTester::getCacheSize() << std::endl;
    }
    
    return 0;
}