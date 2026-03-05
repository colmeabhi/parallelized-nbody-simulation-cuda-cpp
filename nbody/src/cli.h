// cli.h  —  Minimal "--key value" / "--flag" command-line parser.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

class Cli {
public:
    Cli(int argc, char** argv) {
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            if (a.rfind("--", 0) == 0) {
                std::string key = a.substr(2), val = "1";
                auto eq = key.find('=');
                if (eq != std::string::npos) {
                    val = key.substr(eq + 1); key = key.substr(0, eq);
                } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                    val = argv[++i];
                }
                kv_[key] = val;
            } else {
                positional_.push_back(a);
            }
        }
    }
    bool has(const std::string& k) const { return kv_.count(k) != 0; }
    std::string str(const std::string& k, const std::string& def) const {
        auto it = kv_.find(k); return it == kv_.end() ? def : it->second;
    }
    long num(const std::string& k, long def) const {
        auto it = kv_.find(k); return it == kv_.end() ? def : std::atol(it->second.c_str());
    }
    double real(const std::string& k, double def) const {
        auto it = kv_.find(k); return it == kv_.end() ? def : std::atof(it->second.c_str());
    }
    const std::vector<std::string>& positional() const { return positional_; }

private:
    std::map<std::string, std::string> kv_;
    std::vector<std::string> positional_;
};

// Common CSV schema written by both benchmark binaries.
inline void print_bench_header(FILE* f) {
    std::fprintf(f, "impl,integrator,n,block,threads,steps,ms_per_step,"
                    "interactions_per_s,gflops,energy_drift,force_rel_err\n");
}
