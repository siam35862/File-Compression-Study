// ============================================================================
// PAQ9-CPU-MT
// CPU multi-threaded port of the uploaded PAQ9-CUDA warp-cooperative codec.
//
// FORMAT COMPATIBILITY:
//   The on-disk archive format is intentionally kept byte-for-byte compatible
//   with the uploaded CUDA program:
//       magic = "PAQ9-CUDA"\n
//       version = 1\n//       filename = NUL-terminated string
//       mode = 'c' for compressed data
//       total_size, chunk_MB, memory_level, chunk_level, num_of_chunks
//       each chunk: uint32 input_size, uint32 output_size, payload
//
//   The arithmetic coder and adaptive model use the same equations and lookup
//   tables as the uploaded CUDA source. Therefore archives made by this CPU
//   program can be decompressed by the CUDA program, and archives made by the
//   CUDA program can be decompressed by this CPU program.
//
// MULTITHREADING:
//   One CPU worker owns one complete model state and processes one independent
//   chunk at a time. The seven CUDA-parallel hash lookups and eleven StateMap
//   lookups are executed serially on that worker because they are semantically
//   independent; chunk-level parallelism is retained across std::threads.
//
// NOTE:
//   Different CPUs are allowed to produce different compressed byte streams
//   only if integer semantics differ. This implementation uses explicit
//   32-bit wrapping where the original CUDA code relies on 32-bit arithmetic.
//   It is designed to preserve codec correctness and cross-decompression, not
//   to require identical compressed bytes for every compiler/CPU.
// ============================================================================

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using U8  = std::uint8_t;
using U16 = std::uint16_t;
using U32 = std::uint32_t;
using I32 = std::int32_t;

constexpr int COMPRESS = 0;
constexpr int DECOMPRESS = 1;
constexpr std::size_t MB = 1024ull * 1024ull;
constexpr int BASE_MEMORY_LEVEL = 19;
constexpr int DEFAULT_MEMORY_LEVEL = 1;
constexpr int DEFAULT_CHUNK_LEVEL = 1;
constexpr std::size_t ENCODER_BUFSIZE = 0x20000;
constexpr const char *ARCHIVE_MAGIC = "PAQ9-CUDA";
constexpr std::size_t ARCHIVE_MAGIC_LEN = 9;

int memory_level = DEFAULT_MEMORY_LEVEL;
int chunk_MB = 1;
int chunk_level = DEFAULT_CHUNK_LEVEL;
std::size_t total_uncompressed_size = 0;
std::size_t total_compressed_size = 0;

// ---------------------------------------------------------------------------
// Explicit two's-complement 32-bit arithmetic helpers.
// ---------------------------------------------------------------------------
static inline I32 wrap32(std::int64_t x) {
    return static_cast<I32>(static_cast<U32>(x));
}

static inline I32 add32(I32 a, std::int64_t b) {
    return wrap32(static_cast<std::int64_t>(a) + b);
}

static inline I32 sub32(I32 a, I32 b) {
    return wrap32(static_cast<std::int64_t>(a) - static_cast<std::int64_t>(b));
}

static inline I32 sar32(I32 x, unsigned n) {
    // C++ signed-right-shift is implementation-defined for negative values.
    // Convert through int64_t so the intended arithmetic shift is explicit.
    return static_cast<I32>(static_cast<std::int64_t>(x) >> n);
}

static inline U32 u32_add(U32 a, U32 b) {
    return static_cast<U32>(static_cast<std::uint64_t>(a) + b);
}

static inline bool add_will_overflow(std::size_t a, std::size_t b, std::size_t limit) {
    return b > limit - a;
}

// ---------------------------------------------------------------------------
// Squash / Stretch / Ilog
// ---------------------------------------------------------------------------
class Squash {
    std::array<I32, 4096> tab{};
public:
    Squash();
    int operator()(int d) const;
};

Squash::Squash() {
    static const int t[33] = {
        1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101,
        1546, 2047, 2549, 2994, 3348, 3607, 3785, 3901, 3975, 4022,
        4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094};
    for (int i = -2048; i < 2048; ++i) {
        int w = i & 127;
        int d = (i >> 7) + 16;
        tab[i + 2048] = static_cast<I32>((t[d] * (128 - w) + t[d + 1] * w + 64) >> 7);
    }
}

int Squash::operator()(int d) const {
    d += 2048;
    if (d < 0) return 0;
    if (d > 4095) return 4095;
    return tab[d];
}

class Stretch {
    std::array<I32, 4096> tab{};
public:
    explicit Stretch(const Squash &squash);
    int operator()(int p) const;
};

Stretch::Stretch(const Squash &squash) {
    int pi = 0;
    for (int x = -2047; x <= 2047; ++x) {
        int i = squash(x);
        for (int j = pi; j <= i && j < 4096; ++j) tab[j] = x;
        pi = i + 1;
    }
    tab[4095] = 2047;
}

int Stretch::operator()(int p) const {
    assert(p >= 0 && p < 4096);
    return tab[p];
}

class Ilog {
    std::array<U8, 65536> table{};
public:
    Ilog();
    int operator()(U16 x) const { return table[x]; }
    int operator()(U32 x) const;
};

Ilog::Ilog() {
    U32 x = 14155776;
    for (int i = 2; i < 65536; ++i) {
        x = static_cast<U32>(x + 774541002u / static_cast<U32>(i * 2 - 1));
        table[i] = static_cast<U8>(x >> 24);
    }
}

int Ilog::operator()(U32 x) const {
    if (x >= 0x1000000u) return 256 + table[x >> 16];
    if (x >= 0x10000u)   return 128 + table[x >> 8];
    return table[x];
}

struct CodecTables {
    Squash squash;
    Stretch stretch;
    Ilog ilog;
    CodecTables() : squash(), stretch(squash), ilog() {}
};

static CodecTables &tables() {
    static CodecTables t;
    return t;
}

// ---------------------------------------------------------------------------
// State table / StateMap
// ---------------------------------------------------------------------------
static const U8 State_table[256][2] = 
{{1, 2}, {3, 5}, {4, 6}, {7, 10}, {8, 12}, {9, 13}, {11, 14}, {15, 19}, {16, 23}, {17, 24}, {18, 25}, {20, 27}, {21, 28}, {22, 29}, {26, 30}, {31, 33}, {32, 35}, {32, 35}, {32, 35}, {32, 35}, {34, 37}, {34, 37}, {34, 37}, {34, 37}, {34, 37}, {34, 37}, {36, 39}, {36, 39}, {36, 39}, {36, 39}, {38, 40}, {41, 43}, {42, 45}, {42, 45}, {44, 47}, {44, 47}, {46, 49}, {46, 49}, {48, 51}, {48, 51}, {50, 52}, {53, 43}, {54, 57}, {54, 57}, {56, 59}, {56, 59}, {58, 61}, {58, 61}, {60, 63}, {60, 63}, {62, 65}, {62, 65}, {50, 66}, {67, 55}, {68, 57}, {68, 57}, {70, 73}, {70, 73}, {72, 75}, {72, 75}, {74, 77}, {74, 77}, {76, 79}, {76, 79}, {62, 81}, {62, 81}, {64, 82}, {83, 69}, {84, 71}, {84, 71}, {86, 73}, {86, 73}, {44, 59}, {44, 59}, {58, 61}, {58, 61}, {60, 49}, {60, 49}, {76, 89}, {76, 89}, {78, 91}, {78, 91}, {80, 92}, {93, 69}, {94, 87}, {94, 87}, {96, 45}, {96, 45}, {48, 99}, {48, 99}, {88, 101}, {88, 101}, {80, 102}, {103, 69}, {104, 87}, {104, 87}, {106, 57}, {106, 57}, {62, 109}, {62, 109}, {88, 111}, {88, 111}, {80, 112}, {113, 85}, {114, 87}, {114, 87}, {116, 57}, {116, 57}, {62, 119}, {62, 119}, {88, 121}, {88, 121}, {90, 122}, {123, 85}, {124, 97}, {124, 97}, {126, 57}, {126, 57}, {62, 129}, {62, 129}, {98, 131}, {98, 131}, {90, 132}, {133, 85}, {134, 97}, {134, 97}, {136, 57}, {136, 57}, {62, 139}, {62, 139}, {98, 141}, {98, 141}, {90, 142}, {143, 95}, {144, 97}, {144, 97}, {68, 57}, {68, 57}, {62, 81}, {62, 81}, {98, 147}, {98, 147}, {100, 148}, {149, 95}, {150, 107}, {150, 107}, {108, 151}, {108, 151}, {100, 152}, {153, 95}, {154, 107}, {108, 155}, {100, 156}, {157, 95}, {158, 107}, {108, 159}, {100, 160}, {161, 105}, {162, 107}, {108, 163}, {110, 164}, {165, 105}, {166, 117}, {118, 167}, {110, 168}, {169, 105}, {170, 117}, {118, 171}, {110, 172}, {173, 105}, {174, 117}, {118, 175}, {110, 176}, {177, 105}, {178, 117}, {118, 179}, {110, 180}, {181, 115}, {182, 117}, {118, 183}, {120, 184}, {185, 115}, {186, 127}, {128, 187}, {120, 188}, {189, 115}, {190, 127}, {128, 191}, {120, 192}, {193, 115}, {194, 127}, {128, 195}, {120, 196}, {197, 115}, {198, 127}, {128, 199}, {120, 200}, {201, 115}, {202, 127}, {128, 203}, {120, 204}, {205, 115}, {206, 127}, {128, 207}, {120, 208}, {209, 125}, {210, 127}, {128, 211}, {130, 212}, {213, 125}, {214, 137}, {138, 215}, {130, 216}, {217, 125}, {218, 137}, {138, 219}, {130, 220}, {221, 125}, {222, 137}, {138, 223}, {130, 224}, {225, 125}, {226, 137}, {138, 227}, {130, 228}, {229, 125}, {230, 137}, {138, 231}, {130, 232}, {233, 125}, {234, 137}, {138, 235}, {130, 236}, {237, 125}, {238, 137}, {138, 239}, {130, 240}, {241, 125}, {242, 137}, {138, 243}, {130, 244}, {245, 135}, {246, 137}, {138, 247}, {140, 248}, {249, 135}, {250, 69}, {80, 251}, {140, 252}, {249, 135}, {250, 69}, {80, 251}, {140, 252}, {0, 0}, {0, 0}, {0, 0}};

#define nex(state, sel) State_table[(state)][(sel)]

class StateMap {
protected:
    const int N;
    int cntxt;
    U32 *prediction_table;
    static std::array<int, 1024> make_dt() {
        std::array<int, 1024> dt{};
        for (int i = 0; i < 1024; ++i) dt[i] = 16384 / (i + i + 3);
        return dt;
    }
    static const std::array<int, 1024> &dt() {
        static const std::array<int, 1024> v = make_dt();
        return v;
    }
public:
    StateMap(U32 *prediction_table_ptr, int n = 256);
    void update(int y, int limit = 255);
    int predict_next_bit(int cntx);
};

StateMap::StateMap(U32 *prediction_table_ptr, int n)
    : N(n), cntxt(0), prediction_table(prediction_table_ptr) {
    for (int i = 0; i < N; ++i) prediction_table[i] = 2147483648u;
    (void)dt();
}

void StateMap::update(int y, int limit) {
    assert(cntxt >= 0 && cntxt < N);
    U32 v = prediction_table[cntxt];
    int n = static_cast<int>(v & 1023u);
    I32 p = static_cast<I32>(v >> 10);

    if (n < limit) {
        prediction_table[cntxt] = v + 1u;
    } else {
        prediction_table[cntxt] = (v & 0xfffffc00u) | static_cast<U32>(limit);
    }

    I32 dy = wrap32((static_cast<std::int64_t>(y) << 22) - static_cast<std::int64_t>(p));
    I32 d = sar32(dy, 3);
    I32 product = wrap32(static_cast<std::int64_t>(d) * dt()[n]);
    U32 masked = static_cast<U32>(product) & 0xfffffc00u;
    prediction_table[cntxt] += masked;
}

int StateMap::predict_next_bit(int cntx) {
    assert(cntx >= 0 && cntx < N);
    cntxt = cntx;
    return static_cast<I32>(prediction_table[cntxt] >> 20);
}

// ---------------------------------------------------------------------------
// Mix / APM
// ---------------------------------------------------------------------------
class Mix {
protected:
    const int N;
    I32 *wt;
    I32 x1, x2;
    int context;
    I32 last_prediction;
public:
    Mix(I32 *weight_ptr, int n = 512);
    int prediction(int p1, int p2, int cntxt);
    void update(int y);
};

Mix::Mix(I32 *weight_ptr, int n)
    : N(n), wt(weight_ptr), x1(0), x2(0), context(0), last_prediction(0) {
    for (int i = 0; i < N * 2; ++i) wt[i] = 1 << 23;
}

int Mix::prediction(int p1, int p2, int cntxt) {
    assert(cntxt >= 0 && cntxt < N);
    context = cntxt * 2;
    x1 = static_cast<I32>(p1);
    x2 = static_cast<I32>(p2);
    std::int64_t sum = static_cast<std::int64_t>(x1) * sar32(wt[context], 16)
                       + static_cast<std::int64_t>(x2) * sar32(wt[context + 1], 16)
                       + 128;
    last_prediction = wrap32(sum >> 8);
    return last_prediction;
}

void Mix::update(int y) {
    assert(y == 0 || y == 1);
    int error = static_cast<int>(static_cast<I32>((y << 12) - tables().squash(last_prediction)));
    if ((static_cast<U32>(wt[context]) & 3u) < 3u) {
        wt[context] = add32(wt[context], 1);
        int scale = 4 - (static_cast<U32>(wt[context]) & 3u);
        error = static_cast<int>(wrap32(static_cast<std::int64_t>(error) * scale));
    }
    error = static_cast<int>(wrap32((static_cast<std::int64_t>(error) + 8) >> 4));
    I32 d0 = wrap32(static_cast<std::int64_t>(x1) * error);
    d0 = static_cast<I32>(static_cast<U32>(d0) & 0xfffffffcU);
    wt[context] = add32(wt[context], d0);
    I32 d1 = wrap32(static_cast<std::int64_t>(x2) * error);
    wt[context + 1] = add32(wt[context + 1], d1);
}

class APM : public Mix {
public:
    APM(I32 *weight_ptr, int n) : Mix(weight_ptr, n) {
        for (int i = 0; i < n; ++i) wt[2 * i] = 0;
    }
};

// ---------------------------------------------------------------------------
// HashTable
// ---------------------------------------------------------------------------
template <int B>
class HashTable {
    U8 *table;
    const U32 N;
public:
    HashTable(U32 n, U8 *table_ptr) : table(table_ptr), N(n) {
        static_assert(B >= 2 && (B & (B - 1)) == 0, "B must be a power of 2");
        assert(N >= static_cast<U32>(B * 4) && (N & (N - 1)) == 0);
        std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(table_ptr);
        std::size_t offset = 64 - (addr & 63u);
        if (offset >= 64) offset = 64;
        table += offset;
    }
    U8 *operator[](U32 i) {
        i = static_cast<U32>(i * 123456791u);
        i = (i << 16) | (i >> 16);
        i = static_cast<U32>(i * 234567891u);
        int chk = static_cast<int>(i >> 24);
        i = static_cast<U32>(i * B) & (N - B);
        if (table[i] == static_cast<U8>(chk)) return table + i;
        if (table[i ^ B] == static_cast<U8>(chk)) return table + (i ^ B);
        if (table[i ^ (B * 2)] == static_cast<U8>(chk)) return table + (i ^ (B * 2));
        if (table[i + 1] > table[(i + 1) ^ B] || table[i + 1] > table[(i + 1) ^ (B * 2)]) i ^= B;
        if (table[i + 1] > table[(i + 1) ^ B ^ (B * 2)]) i ^= (B ^ B * 2);
        std::memset(table + i, 0, B);
        table[i] = static_cast<U8>(chk);
        return table + i;
    }
};

// ---------------------------------------------------------------------------
// LZP
// ---------------------------------------------------------------------------
static inline bool MEMORY_OK(std::size_t x) { return x > 0; }

class LZP {
    const std::size_t N, H;
    enum { MINLEN = 12 };
    U8 *buffer;
    U32 *table;
    int match;
    std::size_t len, pos;
    U32 hash, hash1, hash2;
    StateMap *statemap;
    APM *apm1, *apm2, *apm3;
    int literals, matches;
    static inline bool isalpha_c(char ch) {
        unsigned char c = static_cast<unsigned char>(ch);
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    }
    static inline char tolower_c(char ch) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch + ('a' - 'A'));
        return ch;
    }
public:
    U32 word0, word1;
    LZP(std::size_t mem, StateMap *statemap1, U8 *buf, U32 *tab, APM *a1, APM *a2, APM *a3)
        : N(mem / 8), H(mem / 32), buffer(buf), table(tab), match(-1), len(0), pos(0),
          hash(0), hash1(0), hash2(0), statemap(statemap1), apm1(a1), apm2(a2), apm3(a3),
          literals(0), matches(0), word0(0), word1(0) {
        assert(MEMORY_OK(N) && MEMORY_OK(H));
    }
    int predict_char() const { return len >= MINLEN ? buffer[static_cast<std::size_t>(match) & (N - 1)] : -1; }
    int context(int i) const { assert(i > 0); return buffer[(pos - static_cast<std::size_t>(i)) & (N - 1)]; }
    int context4() const { return static_cast<int>(hash2); }
    int context8() const { return static_cast<int>(hash1); }
    int probability();
    void update(int ch);
};

int LZP::probability() {
    if (len < MINLEN) return 0;
    int cxt = static_cast<int>(len);
    if (len > 28) cxt = 28 + (len >= 32) + (len >= 64) + (len >= 128);
    int pc = predict_char();
    int pr = statemap->predict_next_bit(cxt);
    pr = tables().stretch(pr);
    pr = ((apm1->prediction(2048, pr * 2, static_cast<int>((hash2 * 256u + static_cast<U32>(pc)) & 0xffffu)) * 3) + pr) >> 2;
    pr = ((apm2->prediction(2048, pr * 2, static_cast<int>((hash1 * (11u << 6) + static_cast<U32>(pc)) & 0x3ffffu)) * 3) + pr) >> 2;
    pr = ((apm3->prediction(2048, pr * 2, static_cast<int>((hash1 * (7u << 4) + static_cast<U32>(pc)) & 0xfffffu)) * 3) + pr) >> 2;
    return tables().squash(pr);
}

void LZP::update(int ch) {
    int y = (predict_char() == ch);
    hash1 = static_cast<U32>(hash1 * (3u << 4) + static_cast<U32>(ch) + 1u);
    hash2 = static_cast<U32>((hash2 << 8) | static_cast<U32>(ch));
    hash = static_cast<U32>(hash * (5u << 2) + static_cast<U32>(ch) + 1u) & static_cast<U32>(H - 1);
    if (len >= MINLEN) {
        statemap->update(y);
        apm1->update(y);
        apm2->update(y);
        apm3->update(y);
    }
    if (isalpha_c(static_cast<char>(ch))) {
        word0 = static_cast<U32>(word0 * (29u << 2) + static_cast<unsigned char>(tolower_c(static_cast<char>(ch))));
    } else if (word0) {
        word1 = word0;
        word0 = 0;
    }
    buffer[pos & (N - 1)] = static_cast<U8>(ch);
    ++pos;
    if (y) {
        ++len;
        ++match;
        ++matches;
    } else {
        ++literals;
        y = 0;
        len = 1;
        match = static_cast<int>(table[hash]);
        if (((static_cast<std::size_t>(match) ^ pos) & (N - 1)) == 0) --match;
        while (len <= 128 && buffer[(static_cast<std::size_t>(match) - len) & (N - 1)] == buffer[(pos - len) & (N - 1)]) ++len;
        --len;
    }
    table[hash] = static_cast<U32>(pos);
}

// ---------------------------------------------------------------------------
// Predictor
// ---------------------------------------------------------------------------
class Predictor {
    static constexpr int N = 11;
    int c0;
    int nibble;
    int bcount;
    HashTable<16> *hashtable;
    StateMap *statemap[N];
    U8 *cp[N];
    U8 *sp[N];
    Mix *mix[N - 1];
    APM *apm1, *apm2, *apm3;
    U8 *context1;
    std::array<int, N> stretched_cache{};
public:
    Predictor(U8 *context1_ptr, StateMap *statemap1[N], Mix *mix1[N - 1], APM *a1, APM *a2, APM *a3, HashTable<16> *hashtable_ptr);
    void update(int y);
    int predict_next_bit(LZP &lzp);
};

Predictor::Predictor(U8 *context1_ptr, StateMap *statemap1[N], Mix *mix1[N - 1], APM *a1, APM *a2, APM *a3, HashTable<16> *hashtable_ptr)
    : c0(0), nibble(1), bcount(0), hashtable(hashtable_ptr), apm1(a1), apm2(a2), apm3(a3), context1(context1_ptr) {
    for (int i = 0; i < N; ++i) {
        sp[i] = cp[i] = context1;
        statemap[i] = statemap1[i];
        if (i < N - 1) mix[i] = mix1[i];
    }
}

void Predictor::update(int y) {
    assert(y == 0 || y == 1);
    if (c0 == 0) {
        c0 = 1 - y;
        return;
    }

    // Same logical operations as CUDA lanes 0..10, but serialized on CPU.
    *sp[0] = nex(*sp[0], y);
    statemap[0]->update(y);
    for (int lane = 1; lane < N; ++lane) {
        *sp[lane] = nex(*sp[lane], y);
        statemap[lane]->update(y);
        mix[lane - 1]->update(y);
    }

    c0 += c0 + y;
    ++bcount;
    if (bcount == 8) bcount = c0 = 0;
    if ((nibble += nibble + y) >= 16) nibble = 1;
    apm1->update(y);
    apm2->update(y);
    apm3->update(y);
}

int Predictor::predict_next_bit(LZP &lzp) {
    if (c0 == 0) return lzp.probability();

    int pc = lzp.predict_char();
    int r = ((pc + 256) >> (8 - bcount)) == c0;
    U32 c4 = static_cast<U32>(lzp.context4());
    U32 c8 = static_cast<U32>(lzp.context8() << 4) - 1u;
    int bc = bcount;

    if ((bc & 3) == 0) {
        int pcr = pc & -r;
        U32 c4p = c4 << 8;
        if (bc == 0) {
            cp[0] = context1 + ((c4 >> 16) & 0xff00u);
            cp[1] = context1 + ((c4 >> 8) & 0xff00u) + 0x10000u;
            cp[2] = context1 + (c4 & 0xff00u) + 0x20000u;
            cp[3] = context1 + ((c4 << 8) & 0xff00u) + 0x30000u;
        }
        cp[4] = hashtable->operator[]((c4p & 0xffff00u) - static_cast<U32>(c0));
        cp[5] = hashtable->operator[]((c4p & 0xffffff00u) * 3u + static_cast<U32>(c0));
        cp[6] = hashtable->operator[](c4 * 7u + static_cast<U32>(c0));
        cp[7] = hashtable->operator[]((c8 * 5u & 0xfffffcu) + static_cast<U32>(c0));
        cp[8] = hashtable->operator[]((c8 * 11u & 0xffffff0u) + static_cast<U32>(c0) + static_cast<U32>(pcr * 13));
        cp[9] = hashtable->operator[]((lzp.word0 * 5u + static_cast<U32>(c0) + static_cast<U32>(pcr * 17)));
        cp[10] = hashtable->operator[]((lzp.word1 * 7u + lzp.word0 * 11u + static_cast<U32>(c0) + static_cast<U32>(pcr * 37)));
    }

    r <<= 8;
    for (int lane = 0; lane < N; ++lane) {
        sp[lane] = &cp[lane][lane < 4 ? c0 : nibble];
        int st = *sp[lane];
        stretched_cache[lane] = tables().stretch(statemap[lane]->predict_next_bit(st));
    }

    int pr = stretched_cache[0];
    for (int i = 1; i < N; ++i) {
        pr = ((mix[i - 1]->prediction(pr, stretched_cache[i], *sp[i] + r) * 3) + pr) >> 2;
    }
    pr = ((apm1->prediction(512, pr * 2, (c0 + pc * 256) & 0xffff) * 3) + pr) >> 2;
    pr = ((apm2->prediction(512, pr * 2, ((c4 << 8) & 0xff00u) | static_cast<U32>(c0)) * 3) + pr) >> 2;
    pr = ((apm3->prediction(512, pr * 2, (c4 * 3u + static_cast<U32>(c0)) & 0xffffu) * 3) + pr) >> 2;
    return tables().squash(pr);
}

// ---------------------------------------------------------------------------
// Per-worker model storage. Each worker owns one model, so no model state is
// shared between threads.
// ---------------------------------------------------------------------------
class CodecModel {
public:
    std::vector<U32> lzp_statemap;
    std::array<std::vector<I32>, 3> lzp_apm;
    std::vector<U8> lzp_buffer;
    std::vector<U32> lzp_table;
    std::array<std::vector<U32>, 11> predictor_statemap;
    std::array<std::vector<I32>, 10> predictor_mix;
    std::array<std::vector<I32>, 3> predictor_apm;
    std::vector<U8> predictor_hashtable;
    std::vector<U8> predictor_context1;

    std::unique_ptr<StateMap> lzp_sm;
    std::array<std::unique_ptr<APM>, 3> lzp_apm_obj;
    std::unique_ptr<LZP> lzp;

    std::array<std::unique_ptr<StateMap>, 11> psm_obj;
    std::array<std::unique_ptr<Mix>, 10> pmix_obj;
    std::array<std::unique_ptr<APM>, 3> papm_obj;
    std::unique_ptr<HashTable<16>> phash_obj;
    std::unique_ptr<Predictor> predictor;

    std::vector<U8> encoder_buffer;
    std::size_t MEM;

    explicit CodecModel(int mem_level)
        : MEM(std::size_t(1) << (BASE_MEMORY_LEVEL + mem_level)) {
        lzp_statemap.resize(0x200);
        lzp_apm[0].resize(0x20000);
        lzp_apm[1].resize(0x80000);
        lzp_apm[2].resize(0x200000);
        lzp_buffer.resize(MEM / 8);
        lzp_table.resize(MEM / 32);

        for (auto &v : predictor_statemap) v.resize(0x100);
        for (auto &v : predictor_mix) v.resize(0x400);
        for (auto &v : predictor_apm) v.resize(0x20000);
        predictor_hashtable.resize(MEM / 2 + 128);
        predictor_context1.resize(0x40000);
        encoder_buffer.resize(ENCODER_BUFSIZE);

        std::fill(predictor_hashtable.begin(), predictor_hashtable.end(), 0);
        std::fill(predictor_context1.begin(), predictor_context1.end(), 0);
        std::fill(lzp_buffer.begin(), lzp_buffer.end(), 0);
        std::fill(lzp_table.begin(), lzp_table.end(), 0);

        lzp_sm = std::make_unique<StateMap>(lzp_statemap.data(), 0x200);
        for (int j = 0; j < 3; ++j)
            lzp_apm_obj[j] = std::make_unique<APM>(lzp_apm[j].data(), j == 0 ? 0x10000 : (j == 1 ? 0x40000 : 0x100000));
        lzp = std::make_unique<LZP>(MEM, lzp_sm.get(), lzp_buffer.data(), lzp_table.data(),
                                    lzp_apm_obj[0].get(), lzp_apm_obj[1].get(), lzp_apm_obj[2].get());

        std::array<StateMap*, 11> psm{};
        std::array<Mix*, 10> pmix{};
        for (int j = 0; j < 11; ++j) {
            psm_obj[j] = std::make_unique<StateMap>(predictor_statemap[j].data(), 0x100);
            psm[j] = psm_obj[j].get();
        }
        for (int j = 0; j < 10; ++j) {
            pmix_obj[j] = std::make_unique<Mix>(predictor_mix[j].data(), 0x200);
            pmix[j] = pmix_obj[j].get();
        }
        for (int j = 0; j < 3; ++j)
            papm_obj[j] = std::make_unique<APM>(predictor_apm[j].data(), 0x10000);
        phash_obj = std::make_unique<HashTable<16>>(static_cast<U32>(MEM / 2), predictor_hashtable.data());
        predictor = std::make_unique<Predictor>(predictor_context1.data(), psm.data(), pmix.data(),
                                                 papm_obj[0].get(), papm_obj[1].get(), papm_obj[2].get(), phash_obj.get());
    }
};

// ---------------------------------------------------------------------------
// Arithmetic encoder/decoder. The algorithm is identical to the uploaded
// CUDA Encoder, but it directly owns one CPU model instance.
// ---------------------------------------------------------------------------
class Encoder {
    const int mode;
    std::vector<U8> &inout;
    std::vector<U8> &buffer;
    std::size_t total_size;
    U32 x1, x2, x;
    std::size_t usize, csize;
    CodecModel &model;
public:
    std::size_t iterator_size;
    Encoder(int m, std::vector<U8> &temp, std::vector<U8> &buffer_ptr, std::size_t tsz,
            std::size_t itr, CodecModel &mstate)
        : mode(m), inout(temp), buffer(buffer_ptr), total_size(tsz), x1(0), x2(0xffffffffu), x(0),
          usize(0), csize(0), model(mstate), iterator_size(itr) {
        if (mode == DECOMPRESS) {
            for (int i = 0; i < 4; ++i) {
                if (iterator_size >= inout.size()) throw std::runtime_error("truncated arithmetic block");
                x = (x << 8) + inout[iterator_size++];
            }
            csize = 4;
        }
    }

    bool put4(U32 c) {
        if (iterator_size > total_size || total_size - iterator_size < 4 || iterator_size + 4 > inout.size()) return false;
        inout[iterator_size++] = static_cast<U8>(c >> 24);
        inout[iterator_size++] = static_cast<U8>(c >> 16);
        inout[iterator_size++] = static_cast<U8>(c >> 8);
        inout[iterator_size++] = static_cast<U8>(c);
        return true;
    }

    int code(int y = 0) {
        int p = model.predictor->predict_next_bit(*model.lzp);
        assert(p >= 0 && p < 4096);
        p += p < 2048;
        U32 xmid = x1 + ((x2 - x1) >> 12) * static_cast<U32>(p)
                 + (((x2 - x1) & 0xfffu) * static_cast<U32>(p) >> 12);
        assert(xmid >= x1 && xmid < x2);
        if (mode == DECOMPRESS) y = x <= xmid;
        if (y) x2 = xmid; else x1 = xmid + 1;
        model.predictor->update(y);
        while (((x1 ^ x2) & 0xff000000u) == 0) {
            if (mode == COMPRESS) {
                if (csize >= buffer.size()) throw std::runtime_error("encoder buffer overflow");
                buffer[csize++] = static_cast<U8>(x2 >> 24);
            }
            x1 <<= 8;
            x2 = (x2 << 8) + 255u;
            if (mode == DECOMPRESS) {
                if (iterator_size >= inout.size()) throw std::runtime_error("truncated arithmetic data");
                x = (x << 8) + inout[iterator_size++];
            }
        }
        return y;
    }

    bool flush() {
        if (mode != COMPRESS) return true;
        if (csize + 4 > buffer.size()) return false;
        buffer[csize++] = static_cast<U8>(x1 >> 24);
        buffer[csize++] = 255;
        buffer[csize++] = 255;
        buffer[csize++] = 255;
        if (!put4(static_cast<U32>(usize))) return false;
        if (!put4(static_cast<U32>(csize))) return false;
        if (iterator_size > inout.size() || csize > inout.size() - iterator_size) return false;
        std::copy(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(csize), inout.begin() + static_cast<std::ptrdiff_t>(iterator_size));
        iterator_size += csize;
        x1 = x = 0;
        usize = csize = 0;
        x2 = 0xffffffffu;
        return true;
    }

    bool count() {
        assert(mode == COMPRESS);
        ++usize;
        if (csize > ENCODER_BUFSIZE - 256) return flush();
        return true;
    }
};

static U32 get4_bytes(const std::vector<U8> &in, std::size_t &pos) {
    if (pos + 4 > in.size()) throw std::runtime_error("truncated 32-bit integer");
    U32 r = in[pos++];
    r = r * 256u + in[pos++];
    r = r * 256u + in[pos++];
    r = r * 256u + in[pos++];
    return r;
}

static U32 get4_stream(std::istream &in) {
    int a = in.get(), b = in.get(), c = in.get(), d = in.get();
    if (a == EOF || b == EOF || c == EOF || d == EOF) throw std::runtime_error("truncated archive");
    return (static_cast<U32>(a) << 24) | (static_cast<U32>(b) << 16) | (static_cast<U32>(c) << 8) | static_cast<U32>(d);
}

static void put4_stream(U32 c, std::ostream &out) {
    out.put(static_cast<char>(c >> 24));
    out.put(static_cast<char>(c >> 16));
    out.put(static_cast<char>(c >> 8));
    out.put(static_cast<char>(c));
}

static std::uint64_t get8_stream(std::istream &in) {
    std::uint64_t r = 0;
    for (int i = 0; i < 8; ++i) {
        int c = in.get();
        if (c == EOF) throw std::runtime_error("truncated archive");
        r = (r << 8) | static_cast<unsigned char>(c);
    }
    return r;
}

static void put8_stream(std::uint64_t c, std::ostream &out) {
    for (int shift = 56; shift >= 0; shift -= 8) out.put(static_cast<char>(c >> shift));
}

// ---------------------------------------------------------------------------
// CPU memory sizing
// ---------------------------------------------------------------------------
static std::size_t calculateModelBytes(int ml) {
    const std::size_t MEM = std::size_t(1) << (BASE_MEMORY_LEVEL + ml);
    std::size_t total = 0;
    total += 0x200 * sizeof(U32);
    total += 0x20000 * sizeof(I32);
    total += 0x80000 * sizeof(I32);
    total += 0x200000 * sizeof(I32);
    total += (MEM / 8) * sizeof(U8);
    total += (MEM / 32) * sizeof(U32);
    total += 11 * (0x100 * sizeof(U32));
    total += 10 * (0x400 * sizeof(I32));
    total += 3 * (0x20000 * sizeof(I32));
    total += (MEM / 2 + 128) * sizeof(U8);
    total += 0x40000 * sizeof(U8);
    total += ENCODER_BUFSIZE;
    // object/allocator overhead margin
    total += 2 * MB;
    return total;
}

static std::size_t get_available_memory() {
#ifdef _WIN32
    MEMORYSTATUSEX s{};
    s.dwLength = sizeof(s);
    if (GlobalMemoryStatusEx(&s)) return static_cast<std::size_t>(s.ullAvailPhys);
    return 0;
#else
    std::ifstream f("/proc/meminfo");
    std::string key;
    std::uint64_t value = 0;
    std::string unit;
    while (f >> key >> value >> unit) {
        if (key == "MemAvailable:") return static_cast<std::size_t>(value) * 1024ull;
    }
    return 0;
#endif
}

static unsigned choose_worker_count(int ml, int cl) {
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    std::size_t chunk_bytes = (std::size_t(1) << (cl - 1)) * MB;
    std::size_t per_worker = calculateModelBytes(ml) + 2 * chunk_bytes + 2 * MB;
    std::size_t avail = get_available_memory();
    if (avail != 0) {
        std::size_t safe = avail * 70 / 100;
        unsigned by_mem = static_cast<unsigned>(std::max<std::size_t>(1, safe / std::max<std::size_t>(1, per_worker)));
        hw = std::min(hw, by_mem);
    }
    return std::max(1u, hw);
}

static int parse_level_arg(const char *s, int fallback) {
    if (!s || s[0] != '-') return fallback;
    if (std::strlen(s) < 2) return fallback;
    for (std::size_t i = 1; i < std::strlen(s); ++i) if (!std::isdigit(static_cast<unsigned char>(s[i]))) return fallback;
    int v = std::atoi(s + 1);
    return (v >= 1 && v <= 11) ? v : fallback;
}

// ---------------------------------------------------------------------------
// Chunk results
// ---------------------------------------------------------------------------
struct ChunkResult {
    std::size_t index = 0;
    std::vector<U8> output;
};

static ChunkResult compress_chunk(std::size_t index, const std::vector<U8> &input, int ml) {
    CodecModel model(ml);
    std::vector<U8> output(input.size() + 2);
    output[0] = '0';
    Encoder enc(COMPRESS, output, model.encoder_buffer, input.size(), 1, model);
    std::size_t pos = 0;
    bool ok = true;
    while (pos < input.size()) {
        int ch = input[pos++];
        int cp = model.lzp->predict_char();
        if (ch == cp) {
            enc.code(1);
        } else {
            for (int i = 8; i >= 0; --i) enc.code((ch >> i) & 1);
        }
        if (!enc.count()) { ok = false; break; }
        model.lzp->update(ch);
    }
    if (ok) ok = enc.flush();
    if (!ok || enc.iterator_size > output.size()) {
        output.clear();
        output.reserve(input.size() + 1);
        output.push_back('1');
        output.insert(output.end(), input.begin(), input.end());
    } else {
        output.resize(enc.iterator_size);
    }
    return {index, std::move(output)};
}

static ChunkResult decompress_chunk(std::size_t index, const std::vector<U8> &input, std::size_t expected_output_size, int ml) {
    if (input.empty()) throw std::runtime_error("empty chunk");
    if (input[0] == '1') {
        if (input.size() - 1 != expected_output_size) throw std::runtime_error("raw chunk size mismatch");
        std::vector<U8> out(input.begin() + 1, input.end());
        return {index, std::move(out)};
    }
    if (input[0] != '0') throw std::runtime_error("invalid chunk mode");

    CodecModel model(ml);
    std::vector<U8> out(expected_output_size);
    std::size_t itr = 1;
    std::size_t outpos = 0;

    // A compressed chunk is a sequence of arithmetic blocks. Each block has:
    //   uint32 usize, uint32 csize, csize bytes of arithmetic-coded data.
    // The CUDA decoder reads exactly this structure and then constructs Encoder
    // at the first arithmetic byte (after the 8-byte block header).
    while (outpos < expected_output_size) {
        if (itr + 8 > input.size()) throw std::runtime_error("truncated arithmetic block header");
        std::size_t usize = get4_bytes(input, itr);
        std::size_t csize = get4_bytes(input, itr);
        if (usize == 0) throw std::runtime_error("zero-sized arithmetic block");
        if (csize < 4 || csize > input.size() - itr) throw std::runtime_error("invalid arithmetic block size");
        if (usize > expected_output_size - outpos) throw std::runtime_error("arithmetic block output overflow");

        const std::size_t block_data_end = itr + csize;
        Encoder enc(DECOMPRESS, const_cast<std::vector<U8>&>(input), model.encoder_buffer, input.size(), itr, model);
        for (std::size_t produced = 0; produced < usize; ++produced) {
            int cp = model.lzp->predict_char();
            int first = enc.code();
            if (first == 0) {
                cp = 1;
                while (cp < 256) cp += cp + enc.code();
                cp &= 255;
            }
            if (outpos >= out.size()) throw std::runtime_error("decompress output overflow");
            out[outpos++] = static_cast<U8>(cp);
            model.lzp->update(cp);
        }
        if (enc.iterator_size > block_data_end)
            throw std::runtime_error("arithmetic decoder consumed beyond block");
        // The original CUDA decoder advances to Encoder::iterator_size.
        // A well-formed block normally consumes all csize bytes.
        itr = enc.iterator_size;
    }

    if (outpos != expected_output_size) throw std::runtime_error("decompress size mismatch");
    return {index, std::move(out)};
}

// ---------------------------------------------------------------------------
// Parallel batch runner. The batch size is capped by memory-driven worker
// count, and output is written in input order to keep the archive format's
// chunk sequence deterministic.
// ---------------------------------------------------------------------------
template <typename Fn>
static std::vector<ChunkResult> parallel_batch(std::size_t first_index,
                                                const std::vector<std::vector<U8>> &inputs,
                                                unsigned workers,
                                                Fn fn) {
    const std::size_t n = inputs.size();
    std::vector<ChunkResult> results(n);
    std::vector<std::thread> threads;
    threads.reserve(n);
    std::atomic<std::size_t> next{0};
    std::mutex err_mtx;
    std::exception_ptr first_error;

    auto worker = [&]() {
        try {
            while (true) {
                std::size_t local = next.fetch_add(1);
                if (local >= n) break;
                results[local] = fn(first_index + local, inputs[local]);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(err_mtx);
            if (!first_error) first_error = std::current_exception();
        }
    };

    unsigned nthreads = static_cast<unsigned>(std::min<std::size_t>(workers, n));
    for (unsigned i = 0; i < nthreads; ++i) threads.emplace_back(worker);
    for (auto &t : threads) t.join();
    if (first_error) std::rethrow_exception(first_error);
    return results;
}

// ---------------------------------------------------------------------------
// Archive I/O
// ---------------------------------------------------------------------------
static void write_archive_header(std::ostream &dest, const std::string &source_file,
                                 std::uint64_t total_B, int chunk_mb, int ml, int cl, int chunks) {
    dest.write(ARCHIVE_MAGIC, static_cast<std::streamsize>(ARCHIVE_MAGIC_LEN));
    dest.put(1);
    dest.write(source_file.data(), static_cast<std::streamsize>(source_file.size()));
    dest.put('\0');
    dest.put('c');
    put8_stream(total_B, dest);
    put4_stream(static_cast<U32>(chunk_mb), dest);
    put4_stream(static_cast<U32>(ml), dest);
    put4_stream(static_cast<U32>(cl), dest);
    put4_stream(static_cast<U32>(chunks), dest);
}

static void compress_file(const std::string &destination_file, const std::string &source_file) {
    std::ifstream source(source_file, std::ios::binary);
    if (!source) throw std::runtime_error("Cannot open " + source_file);
    source.seekg(0, std::ios::end);
    std::uint64_t total_B = static_cast<std::uint64_t>(source.tellg());
    source.seekg(0, std::ios::beg);

    chunk_MB = (1 << (chunk_level - 1));
    std::size_t chunk_B = static_cast<std::size_t>(chunk_MB) * MB;
    std::size_t num_chunks = total_B == 0 ? 0 : static_cast<std::size_t>((total_B + chunk_B - 1) / chunk_B);
    unsigned workers = choose_worker_count(memory_level, chunk_level);

    std::ofstream dest(destination_file, std::ios::binary);
    if (!dest) throw std::runtime_error("Cannot create/open " + destination_file);
    write_archive_header(dest, source_file, total_B, chunk_MB, memory_level, chunk_level, static_cast<int>(num_chunks));

    std::cout << "CPU version of PAQ9 started successfully.\n\n";
    std::cout << "Memory Chunk Level: " << chunk_MB << " MB\n";
    std::cout << "Memory Level: " << memory_level << "\n";
    std::cout << "Level: " << chunk_level << "\n";
    std::cout << "Number of Chunks: " << num_chunks << "\n";
    std::cout << "CPU workers: " << workers << "\n";

    total_uncompressed_size = 0;
    total_compressed_size = 0;
    auto start = std::chrono::steady_clock::now();

    for (std::size_t base = 0; base < num_chunks; base += workers) {
        std::size_t count = std::min<std::size_t>(workers, num_chunks - base);
        std::vector<std::vector<U8>> inputs(count);
        for (std::size_t i = 0; i < count; ++i) {
            std::size_t current = std::min<std::size_t>(chunk_B, static_cast<std::size_t>(total_B - (base + i) * chunk_B));
            inputs[i].resize(current);
            if (current) source.read(reinterpret_cast<char*>(inputs[i].data()), static_cast<std::streamsize>(current));
            if (!source && current) throw std::runtime_error("read failed while compressing");
        }

        auto results = parallel_batch(base, inputs, workers,
            [&](std::size_t idx, const std::vector<U8> &in) { return compress_chunk(idx, in, memory_level); });

        std::sort(results.begin(), results.end(), [](const ChunkResult &a, const ChunkResult &b) { return a.index < b.index; });
        for (std::size_t i = 0; i < count; ++i) {
            put4_stream(static_cast<U32>(inputs[i].size()), dest);
            put4_stream(static_cast<U32>(results[i].output.size()), dest);
            dest.write(reinterpret_cast<const char*>(results[i].output.data()), static_cast<std::streamsize>(results[i].output.size()));
            total_uncompressed_size += inputs[i].size();
            total_compressed_size += results[i].output.size();
        }
    }

    auto end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    std::cout << "Uncompressed  ->  Compressed\n";
    std::cout << "Total: " << total_uncompressed_size << " Byte -> " << total_compressed_size << " Byte\n";
    if (total_compressed_size) std::cout << "Compression Ratio: " << (1.0 * total_uncompressed_size / total_compressed_size) << "\n";
    std::cout << "Time Taken: " << seconds << " seconds\n";
    if (seconds > 0) std::cout << "Compression Speed: " << total_uncompressed_size / seconds / 1024.0 << " KB/s\n";
}

struct ArchiveInfo {
    std::uint64_t total_uncompressed = 0;
    int chunk_mb = 1;
    int mem_level = 1;
    int chunk_level = 1;
    int num_chunks = 0;
    std::string original_name;
    char mode = 0;
};

static ArchiveInfo read_archive_header(std::istream &source) {
    char magic[ARCHIVE_MAGIC_LEN];
    source.read(magic, ARCHIVE_MAGIC_LEN);
    if (source.gcount() != static_cast<std::streamsize>(ARCHIVE_MAGIC_LEN) || std::memcmp(magic, ARCHIVE_MAGIC, ARCHIVE_MAGIC_LEN) != 0)
        throw std::runtime_error("This is not a compatible PAQ9-CUDA archive");
    int version = source.get();
    if (version != 1) throw std::runtime_error("Unsupported PAQ9 archive version");
    std::string filename;
    char c;
    while (source.get(c) && c != '\0') filename.push_back(c);
    if (!source) throw std::runtime_error("truncated archive filename");
    ArchiveInfo a;
    a.original_name = filename;
    a.mode = static_cast<char>(source.get());
    if (a.mode != 'c' && a.mode != 's') throw std::runtime_error("Unsupported archive mode");
    a.total_uncompressed = get8_stream(source);
    a.chunk_mb = static_cast<int>(get4_stream(source));
    a.mem_level = static_cast<int>(get4_stream(source));
    a.chunk_level = static_cast<int>(get4_stream(source));
    a.num_chunks = static_cast<int>(get4_stream(source));
    if (a.chunk_mb <= 0 || a.mem_level < 1 || a.mem_level > 11 || a.chunk_level < 1 || a.chunk_level > 11 || a.num_chunks < 0)
        throw std::runtime_error("invalid archive parameters");
    return a;
}

static void decompress_file(const std::string &destination_file_in, const std::string &source_file) {
    std::ifstream source(source_file, std::ios::binary);
    if (!source) throw std::runtime_error("Cannot open " + source_file);

    ArchiveInfo a = read_archive_header(source);
    std::string destination_file = destination_file_in.empty() ? a.original_name : destination_file_in;
    if (a.mode != 'c') throw std::runtime_error("This archive does not contain compressed chunks");

    unsigned workers = choose_worker_count(a.mem_level, a.chunk_level);
    std::size_t chunk_B = static_cast<std::size_t>(a.chunk_mb) * MB;

    std::ofstream dest(destination_file, std::ios::binary);
    if (!dest) throw std::runtime_error("Cannot create/open " + destination_file);

    std::cout << "CPU version of PAQ9 started successfully.\n\n";
    std::cout << "Memory Chunk Level: " << a.chunk_mb << " MB\n";
    std::cout << "Memory Level: " << a.mem_level << "\n";
    std::cout << "Level: " << a.chunk_level << "\n";
    std::cout << "Number of Chunks: " << a.num_chunks << "\n";
    std::cout << "CPU workers: " << workers << "\n";

    total_uncompressed_size = 0;
    total_compressed_size = 0;
    auto start = std::chrono::steady_clock::now();

    for (int base = 0; base < a.num_chunks; base += static_cast<int>(workers)) {
        int count = std::min<unsigned>(workers, static_cast<unsigned>(a.num_chunks - base));
        std::vector<std::vector<U8>> inputs(count);
        std::vector<std::size_t> expected(count);
        std::vector<U32> stored_size(count);

        for (int i = 0; i < count; ++i) {
            U32 usize = get4_stream(source);
            U32 csize = get4_stream(source);
            if (csize > 0x7fffffffu) throw std::runtime_error("chunk too large");
            expected[i] = usize;
            stored_size[i] = csize;
            inputs[i].resize(csize);
            if (csize) source.read(reinterpret_cast<char*>(inputs[i].data()), csize);
            if (!source) throw std::runtime_error("truncated chunk payload");
            if (expected[i] > chunk_B && inputs[i].empty()) throw std::runtime_error("invalid chunk header");
        }

        auto results = parallel_batch(base, inputs, workers,
            [&](std::size_t idx, const std::vector<U8> &in) {
                return decompress_chunk(idx, in, expected[idx - base], a.mem_level);
            });
        std::sort(results.begin(), results.end(), [](const ChunkResult &x, const ChunkResult &y) { return x.index < y.index; });

        for (const auto &r : results) {
            dest.write(reinterpret_cast<const char*>(r.output.data()), static_cast<std::streamsize>(r.output.size()));
            total_compressed_size += stored_size[r.index - base];
            total_uncompressed_size += r.output.size();
        }
    }

    if (total_uncompressed_size != a.total_uncompressed)
        throw std::runtime_error("archive total size mismatch after decompression");

    auto end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    std::cout << "Compressed  ->  Decompressed\n";
    std::cout << "Total: " << total_compressed_size << " Byte -> " << total_uncompressed_size << " Byte\n";
    std::cout << "Time Taken: " << seconds << " seconds\n";
    if (seconds > 0) std::cout << "Decompression Speed: " << total_uncompressed_size / seconds / 1024.0 << " KB/s\n";
}

static const char *program_name(const char *p) {
    const char *s = std::strrchr(p, '/');
#ifdef _WIN32
    const char *b = std::strrchr(p, '\\');
    if (!s || (b && b > s)) s = b;
#endif
    return s ? s + 1 : p;
}

static void print_usage(const char *prog) {
    const char *name = program_name(prog);
    std::cout << "Usage:\n";
    std::cout << "  Compress:   " << name << " -c [-<memory_level>] <destination_file> [-<chunk_level>] <source_file>\n";
    std::cout << "  Decompress: " << name << " -d <source_file> <destination_file>\n\n";
    std::cout << "  Levels must be between 1 and 11. Defaults: memory=1, chunk=1.\n";
    std::cout << "  The archive format remains PAQ9-CUDA for CUDA <-> CPU compatibility.\n";
}

int main(int argc, char **argv) {
    try {
        if (argc < 3) { print_usage(argv[0]); return 1; }
        auto start = std::chrono::steady_clock::now();
        if (std::strcmp(argv[1], "-c") == 0) {
            int ind = 2;
            memory_level = DEFAULT_MEMORY_LEVEL;
            chunk_level = DEFAULT_CHUNK_LEVEL;

            if (ind < argc && argv[ind][0] == '-') {
                memory_level = parse_level_arg(argv[ind], DEFAULT_MEMORY_LEVEL);
                ++ind;
            }
            if (ind >= argc) { print_usage(argv[0]); return 1; }
            std::string destination = argv[ind++];
            if (ind < argc && argv[ind][0] == '-') {
                chunk_level = parse_level_arg(argv[ind], DEFAULT_CHUNK_LEVEL);
                ++ind;
            }
            if (ind >= argc) { print_usage(argv[0]); return 1; }
            std::string source = argv[ind++];
            compress_file(destination, source);
        } else if (std::strcmp(argv[1], "-d") == 0) {
            std::string source = argv[2];
            std::string destination = (argc >= 4) ? argv[3] : std::string();
            decompress_file(destination, source);
        } else {
            print_usage(argv[0]); return 1;
        }
        auto end = std::chrono::steady_clock::now();
        std::cout << "Overall Time Taken: " << std::chrono::duration<double>(end - start).count() << " seconds\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 2;
    }
}
