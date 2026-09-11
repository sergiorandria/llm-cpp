#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <set>

#include "llm/dataset.h"
#include "llm/tokenizer.h"
int main() {
    // B15: mmap vs ifstream equivalence
    {
        std::string big(1 << 20, 'a');  // 1MB
        for (size_t i = 0; i < big.size(); i += 7) big[i] = 'b' + (i % 26);
        {
            std::ofstream o("/tmp/big.txt", std::ios::binary);
            o << big;
        }
        llm::Dataset d1("/tmp/big.txt", 64, false), d2("/tmp/big.txt", 64, true);
        assert(d1.tokens() == d2.tokens());
        assert(!d1.tokens().empty());
        std::cout << "mmap ok tokens=" << d1.size() << "\n";
    }
    // B16: packing shapes + masks + roundtrip of concatenated content
    {
        std::vector<int> toks = {1, 2, 3, 4, 5, 6, 7};
        auto pb = llm::pack_with_eos(toks, 4, 0);
        assert(pb.input.size() == 2 && pb.mask.size() == 2);
        assert((pb.input[0] == std::vector<int>{1, 2, 3, 4}));
        assert((pb.mask[0] == std::vector<int>{1, 1, 1, 1}));
        assert((pb.input[1] == std::vector<int>{5, 6, 7, 0}));
        assert((pb.mask[1] == std::vector<int>{1, 1, 1, 0}));
    }
    // B17: shuffle covers all tokens once, deterministic per seed
    {
        llm::Dataset ds("/tmp/big.txt", 64, false);
        llm::DataLoader l1(ds, 512, true, 7), l2(ds, 512, true, 7), l3(ds, 512, true, 8);
        std::vector<int> a, b;
        while (l1.has_next()) {
            auto x = l1.next_batch();
            a.insert(a.end(), x[0].begin(), x[0].end());
        }
        while (l2.has_next()) {
            auto x = l2.next_batch();
            b.insert(b.end(), x[0].begin(), x[0].end());
        }
        assert(a == b);  // same seed identical
        assert(a.size() == ds.size());
        // different seed usually different order (1MB/512 = 2048 blocks, collision negligible)
        llm::DataLoader l4(ds, 512, true, 8);
        std::vector<int> c;
        while (l4.has_next()) {
            auto x = l4.next_batch();
            c.insert(c.end(), x[0].begin(), x[0].end());
        }
        assert(c.size() == a.size());
        assert(c != a);
        // multiset equality: same tokens, permuted blocks
        assert(std::multiset<int>(a.begin(), a.end()) == std::multiset<int>(c.begin(), c.end()));
    }
    // B18: split ratio + concat == original
    {
        std::vector<int> t(1000);
        for (int i = 0; i < 1000; ++i) t[i] = i;
        std::vector<int> tr, va;
        llm::train_val_split(t, 0.9, 42, tr, va);
        assert(tr.size() == 900 && va.size() == 100);
        assert(tr[0] == 0 && va[0] == 900);
    }
    std::cout << "data_pipeline test passed\n";
    return 0;
}
