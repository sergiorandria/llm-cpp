// I87: backend reports CPU on this build; matmul stays correct either way.
#include <cassert>
#include <iostream>

#include "llm/backend.h"
#include "llm/tensor.h"
int main() {
    assert(!llm::cuda_available());
    assert(std::string(llm::backend_name()) == "cpu");
    assert(llm::active_backend() == llm::Backend::CPU);
    llm::Tensor A({2, 2}, 0.0f), B({2, 2}, 0.0f);
    A.data = {1, 2, 3, 4};
    B.data = {5, 6, 7, 8};
    auto C = A.matmul(B);
    assert(C.data[0] == 19 && C.data[3] == 50);
    std::cout << "backend test passed (" << llm::backend_name() << ")\n";
    return 0;
}
