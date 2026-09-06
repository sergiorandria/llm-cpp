#include <cassert>
#include <iostream>

#include "llm/early_exit.h"

int main() {
    // High confidence logits should early exit
    llm::Tensor logits({2, 4}, 0.0f);
    logits(1, 0) = 10.0f;  // row 1 high confidence
    logits(1, 1) = 0.0f;
    logits(1, 2) = 0.0f;
    logits(1, 3) = 0.0f;
    bool should = llm::should_early_exit(logits, 0.9f);
    std::cout << "should early exit high conf " << should << "\n";
    assert(should == true);
    // Uniform logits should not exit
    llm::Tensor logits2({1, 4}, 0.0f);
    // all zeros -> softmax uniform 0.25 <0.9
    bool should2 = llm::should_early_exit(logits2, 0.9f);
    std::cout << "should early exit uniform " << should2 << "\n";
    assert(should2 == false);
    // Threshold edge: 0.95 high
    llm::Tensor logits3({1, 2}, 0.0f);
    logits3(0, 0) = 2.0f;
    logits3(0, 1) = 0.0f;  // softmax ~0.88
    bool s3 = llm::should_early_exit(logits3, 0.95f);
    assert(s3 == false);
    std::cout << "early_exit test passed\n";
    return 0;
}
