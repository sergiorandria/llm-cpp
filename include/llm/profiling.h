#pragma once
// Tracy profiling hooks — no-op if TRACY_ENABLE not defined, otherwise zone scopes
#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#define LLM_ZONE ZoneScoped
#define LLM_ZONE_N(n) ZoneScopedN(n)
#else
#define LLM_ZONE do{}while(0)
#define LLM_ZONE_N(n) do{}while(0)
#endif
// SoA layout note: Tensor currently AoS (vector<float> row-major), SoA would be
// struct { vector<float> data; } with column-major for matmul — future: transpose B for better cache
