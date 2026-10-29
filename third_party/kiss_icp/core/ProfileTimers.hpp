// Temporizadores ligeros (std::chrono) para perfilar KISS-ICP por dentro.
// Se aplica como parche (patches/kiss-icp-timers.patch); no forma parte de KISS-ICP.
//
// Al terminar el programa imprime en stderr una tabla con las llamadas, el tiempo
// total y el tiempo medio de cada region, mas unos contadores. Si la variable de
// entorno KISS_PROFILE_OUT apunta a un archivo, tambien escribe un CSV.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace kiss_icp::prof {

struct Entry {
    std::string name;
    std::uint64_t calls = 0;
    double total = 0.0;  // ms para temporizadores, suma de valores para contadores
    double max = 0.0;
};

class Registry {
public:
    ~Registry() { Report(); }

    void AddTime(const char *name, double ms) { Add(timers_, name, ms); }
    void AddCount(const char *name, double value) { Add(counters_, name, value); }

private:
    static double Mean(const Entry &e) { return e.total / static_cast<double>(e.calls); }

    void Add(std::vector<Entry> &v, const char *name, double value) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = std::find_if(v.begin(), v.end(), [&](const Entry &e) { return e.name == name; });
        if (it == v.end()) {
            v.push_back(Entry{name});
            it = v.end() - 1;
        }
        it->calls += 1;
        it->total += value;
        it->max = std::max(it->max, value);
    }

    void Report() const {
        if (timers_.empty() && counters_.empty()) return;
        double ref = 0.0;
        for (const auto &e : timers_)
            if (e.name == "register_frame") ref = e.total;

        std::fprintf(stderr, "\n=== KISS-ICP: tiempos internos (std::chrono, pared, ms) ===\n");
        std::fprintf(stderr, "%-22s %8s %12s %12s %10s %8s\n", "region", "llamadas", "total",
                     "media", "max", "% frame");
        for (const auto &e : timers_) {
            std::fprintf(stderr, "%-22s %8llu %12.2f %12.3f %10.2f %7.1f%%\n", e.name.c_str(),
                         static_cast<unsigned long long>(e.calls), e.total, Mean(e),
                         e.max, ref > 0.0 ? 100.0 * e.total / ref : 0.0);
        }
        std::fprintf(stderr, "\n%-22s %8s %14s %12s %10s\n", "contador", "eventos", "suma", "media",
                     "max");
        for (const auto &e : counters_) {
            std::fprintf(stderr, "%-22s %8llu %14.0f %12.2f %10.0f\n", e.name.c_str(),
                         static_cast<unsigned long long>(e.calls), e.total, Mean(e),
                         e.max);
        }
        std::fprintf(stderr, "\n");

        if (const char *path = std::getenv("KISS_PROFILE_OUT")) {
            if (FILE *f = std::fopen(path, "w")) {
                std::fprintf(f, "tipo,nombre,eventos,total,media,max\n");
                for (const auto &e : timers_)
                    std::fprintf(f, "tiempo_ms,%s,%llu,%.4f,%.5f,%.4f\n", e.name.c_str(),
                                 static_cast<unsigned long long>(e.calls), e.total,
                                 Mean(e), e.max);
                for (const auto &e : counters_)
                    std::fprintf(f, "contador,%s,%llu,%.0f,%.3f,%.0f\n", e.name.c_str(),
                                 static_cast<unsigned long long>(e.calls), e.total,
                                 Mean(e), e.max);
                std::fclose(f);
            }
        }
    }

    std::mutex mutex_;
    std::vector<Entry> timers_;
    std::vector<Entry> counters_;
};

inline Registry &GetRegistry() {
    static Registry registry;  // se destruye al terminar el programa e imprime el informe
    return registry;
}

class ScopedTimer {
public:
    explicit ScopedTimer(const char *name)
        : name_(name), t0_(std::chrono::steady_clock::now()) {}
    ~ScopedTimer() {
        const auto dt = std::chrono::steady_clock::now() - t0_;
        GetRegistry().AddTime(name_, std::chrono::duration<double, std::milli>(dt).count());
    }
    ScopedTimer(const ScopedTimer &) = delete;
    ScopedTimer &operator=(const ScopedTimer &) = delete;

private:
    const char *name_;
    std::chrono::steady_clock::time_point t0_;
};

inline void Count(const char *name, double value) { GetRegistry().AddCount(name, value); }

}  // namespace kiss_icp::prof

#define KISS_PROF_CAT2(a, b) a##b
#define KISS_PROF_CAT(a, b) KISS_PROF_CAT2(a, b)
#define KISS_TIMER(name) ::kiss_icp::prof::ScopedTimer KISS_PROF_CAT(kiss_timer_, __LINE__)(name)
