#include "llm/cli.h"
namespace llm {
std::string Args::get(const std::string& k, const std::string& def) const { auto it=kv.find(k); return it==kv.end()?def:it->second; }
bool Args::has(const std::string& k) const { return kv.find(k)!=kv.end(); }
Args parse_args(int argc, char* argv[]){
    Args a;
    for(int i=1;i<argc;++i){
        std::string s=argv[i];
        if(s.rfind("--",0)==0){
            size_t eq=s.find("=");
            if(eq!=std::string::npos){ a.kv[s.substr(2,eq-2)]=s.substr(eq+1); }
            else if(i+1<argc && std::string(argv[i+1]).rfind("--",0)!=0){ a.kv[s.substr(2)]=argv[++i]; }
            else a.kv[s.substr(2)]="true";
        }
    }
    return a;
}
}
