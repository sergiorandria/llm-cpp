#include "llm/tokenizer.h"
#include <random>
int main(){
    llm::Tokenizer tok(1000);
    std::mt19937 rng(0);
    for(int i=0;i<1000;++i){
        std::string s; int len=rng()%20;
        for(int j=0;j<len;++j) s.push_back(char(rng()%256));
        auto ids=tok.encode(s);
        auto dec=tok.decode(ids);
        // Not asserting equality for random bytes, just that it doesn't crash
        (void)dec;
    }
    return 0;
}
