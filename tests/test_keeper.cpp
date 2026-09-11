#include <cassert>
#include <cmath>
#include <iostream>

#include "llm/checkpoint.h"
// NOTE: never put side-effecting calls inside assert() — Release builds define
// NDEBUG so assert() compiles to ((void)0) and the call never happens.
int main() {
    // D38: simulated ppl sequence keeps correct top-3 + best
    llm::BestKeeper k("/tmp/keeper", 3);
    bool r1 = k.consider(1, 10.0f);
    bool r2 = k.consider(2, 8.0f);
    bool r3 = k.consider(3, 9.0f);
    bool r4 = k.consider(4, 20.0f);  // worst, outside top-3
    bool r5 = k.consider(5, 7.0f);   // new best
    float best = k.best_ppl();
    assert(r1 == true);
    assert(r2 == true);
    assert(r3 == true);
    assert(r4 == false);
    assert(r5 == true);
    assert(std::fabs(best - 7.0f) < 1e-6);
    std::cout << "keeper test passed best=" << best << "\n";
    return 0;
}
