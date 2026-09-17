// =====================================================================
// PAQ9-CPU — multithreaded CPU port of PAQ9-CUDA (warp-cooperative).
//
// This is a faithful 1:1 translation of paq9-cuda-warp.cu to plain C++.
// The warp-cooperative GPU parallelism is collapsed back into serial
// per-chunk code (the lanes 0..10 that spread independent lookups across
// a warp are simply executed in sequence on a single CPU thread), and the
// CHUNK-level parallelism (one chunk per GPU "thread") is implemented
// with std::thread.
//
// The archive format is byte-for-byte identical to the CUDA version, so
// files compressed by either binary can be decompressed by the other.
// The compression is deterministic: given the same memory_level /
// chunk_level, CPU and GPU produce identical output.
// =====================================================================

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <thread>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <cassert>
#include <cctype>

typedef unsigned char U8;
typedef unsigned short U16;
typedef unsigned int U32;

#define MAX_THREADS 2048
#define COMPRESS 0
#define DECOMPRESS 1
constexpr size_t MB = 1024 * 1024;
#define base_memory_level 19

int memory_level = 1;   // MEM = 1 << (base_memory_level + memory_level)
int chunk_MB = 1;       // chunk size in MB
int chunk_level = 1;
size_t total_uncompressed_size = 0;
size_t total_compressed_size = 0;

size_t MEM = 1 << (base_memory_level + memory_level); // current MEM (set from memory_level)

///////////////////////////// Squash //////////////////////////////

// return p = 1/(1 + exp(-d)), d scaled by 8 bits, p scaled by 12 bits
class Squash
{
    short tab[4096];

public:
    Squash();
    int operator()(int d) const;
};

Squash::Squash()
{
    static const int t[33] = {
        1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101,
        1546, 2047, 2549, 2994, 3348, 3607, 3785, 3901, 3975, 4022,
        4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094};
    for (int i = -2048; i < 2048; ++i)
    {
        int w = i & 127;
        int d = (i >> 7) + 16;
        tab[i + 2048] = (t[d] * (128 - w) + t[(d + 1)] * w + 64) >> 7;
    }
}

int Squash::operator()(int d) const
{
    d += 2048;
    if (d < 0)
        return 0;
    else if (d > 4095)
        return 4095;
    else
        return tab[d];
}

Squash *squash = nullptr;

//////////////////////////// Stretch ///////////////////////////////

class Stretch
{
    short t[4096];

public:
    Stretch(const Squash &sq);
    int operator()(int p) const;
};

Stretch::Stretch(const Squash &sq)
{
    int pi = 0;
    for (int x = -2047; x <= 2047; ++x)
    { // invert squash()
        int i = sq(x);
        for (int j = pi; j <= i; ++j)
            t[j] = x;
        pi = i + 1;
    }
    t[4095] = 2047;
}

int Stretch::operator()(int p) const
{
    assert(p >= 0 && p < 4096);
    return t[p];
}

Stretch *stretch = nullptr;

///////////////////////// state table ////////////////////////

static const U8 State_table[256][2] = {
    {1, 2}, {3, 5}, {4, 6}, {7, 10}, {8, 12}, {9, 13}, {11, 14}, {15, 19}, {16, 23}, {17, 24}, {18, 25}, {20, 27}, {21, 28}, {22, 29}, {26, 30}, {31, 33}, {32, 35}, {32, 35}, {32, 35}, {32, 35}, {34, 37}, {34, 37}, {34, 37}, {34, 37}, {34, 37}, {34, 37}, {36, 39}, {36, 39}, {36, 39}, {36, 39}, {38, 40}, {41, 43}, {42, 45}, {42, 45}, {44, 47}, {44, 47}, {46, 49}, {46, 49}, {48, 51}, {48, 51}, {50, 52}, {53, 43}, {54, 57}, {54, 57}, {56, 59}, {56, 59}, {58, 61}, {58, 61}, {60, 63}, {60, 63}, {62, 65}, {62, 65}, {50, 66}, {67, 55}, {68, 57}, {68, 57}, {70, 73}, {70, 73}, {72, 75}, {72, 75}, {74, 77}, {74, 77}, {76, 79}, {76, 79}, {62, 81}, {62, 81}, {64, 82}, {83, 69}, {84, 71}, {84, 71}, {86, 73}, {86, 73}, {44, 59}, {44, 59}, {58, 61}, {58, 61}, {60, 49}, {60, 49}, {76, 89}, {76, 89}, {78, 91}, {78, 91}, {80, 92}, {93, 69}, {94, 87}, {94, 87}, {96, 45}, {96, 45}, {48, 99}, {48, 99}, {88, 101}, {88, 101}, {80, 102}, {103, 69}, {104, 87}, {104, 87}, {106, 57}, {106, 57}, {62, 109}, {62, 109}, {88, 111}, {88, 111}, {80, 112}, {113, 85}, {114, 87}, {114, 87}, {116, 57}, {116, 57}, {62, 119}, {62, 119}, {88, 121}, {88, 121}, {90, 122}, {123, 85}, {124, 97}, {124, 97}, {126, 57}, {126, 57}, {62, 129}, {62, 129}, {98, 131}, {98, 131}, {90, 132}, {133, 85}, {134, 97}, {134, 97}, {136, 57}, {136, 57}, {62, 139}, {62, 139}, {98, 141}, {98, 141}, {90, 142}, {143, 95}, {144, 97}, {144, 97}, {68, 57}, {68, 57}, {62, 81}, {62, 81}, {98, 147}, {98, 147}, {100, 148}, {149, 95}, {150, 107}, {150, 107}, {108, 151}, {108, 151}, {100, 152}, {153, 95}, {154, 107}, {108, 155}, {100, 156}, {157, 95}, {158, 107}, {108, 159}, {100, 160}, {161, 105}, {162, 107}, {108, 163}, {110, 164}, {165, 105}, {166, 117}, {118, 167}, {110, 168}, {169, 105}, {170, 117}, {118, 171}, {110, 172}, {173, 105}, {174, 117}, {118, 175}, {110, 176}, {177, 105}, {178, 117}, {118, 179}, {110, 180}, {181, 115}, {182, 117}, {118, 183}, {120, 184}, {185, 115}, {186, 127}, {128, 187}, {120, 188}, {189, 115}, {190, 127}, {128, 191}, {120, 192}, {193, 115}, {194, 127}, {128, 195}, {120, 196}, {197, 115}, {198, 127}, {128, 199}, {120, 200}, {201, 115}, {202, 127}, {128, 203}, {120, 204}, {205, 115}, {206, 127}, {128, 207}, {120, 208}, {209, 125}, {210, 127}, {128, 211}, {130, 212}, {213, 125}, {214, 137}, {138, 215}, {130, 216}, {217, 125}, {218, 137}, {138, 219}, {130, 220}, {221, 125}, {222, 137}, {138, 223}, {130, 224}, {225, 125}, {226, 137}, {138, 227}, {130, 228}, {229, 125}, {230, 137}, {138, 231}, {130, 232}, {233, 125}, {234, 137}, {138, 235}, {130, 236}, {237, 125}, {238, 137}, {138, 239}, {130, 240}, {241, 125}, {242, 137}, {138, 243}, {130, 244}, {245, 135}, {246, 137}, {138, 247}, {140, 248}, {249, 135}, {250, 69}, {80, 251}, {140, 252}, {249, 135}, {250, 69}, {80, 251}, {140, 252}, {0, 0}, {0, 0}, {0, 0}};

#define nex(state, sel) State_table[state][sel]

//////////////////////////// StateMap //////////////////////////

// state_map_dt is a shared read-only table (initialized once at startup).
static int state_map_dt[1024];
static const bool state_map_dt_initialized = []() {
    for (int i = 0; i < 1024; i++)
        state_map_dt[i] = 16384 / (i + i + 3);
    return true;
}();

class StateMap
{
protected:
    const int N;
    int cntxt;
    U32 *prediction_table;

public:
    StateMap(U32 *prediction_table_ptr, int n = 256);
    ~StateMap();
    void update(int y, int limit = 255);
    int predict_next_bit(int cntx);
};

StateMap::StateMap(U32 *prediction_table_ptr, int n) : prediction_table(prediction_table_ptr), N(n), cntxt(0)
{
    for (int i = 0; i < N; i++)
        prediction_table[i] = 2147483648U; // 1<<31
}

StateMap::~StateMap()
{
    prediction_table = 0;
}

void StateMap::update(int y, int limit)
{
    assert(cntxt >= 0 && cntxt < N);
    int n = prediction_table[cntxt] & 1023, p = prediction_table[cntxt] >> 10;

    if (n < limit)
        prediction_table[cntxt]++;
    else
        prediction_table[cntxt] = prediction_table[cntxt] & 0xfffffc00 | limit;

    prediction_table[cntxt] += (((y << 22) - p) >> 3) * state_map_dt[n] & 0xfffffc00;
}

int StateMap::predict_next_bit(int cntx)
{
    assert(cntx >= 0 && cntx < N);
    return prediction_table[cntxt = cntx] >> 20;
}

//////////////////////////// Mix, APM /////////////////////////

class Mix
{
protected:
    const int N;
    int *wt;
    int x1, x2;
    int context;
    int last_prediction;

public:
    Mix(int *weight_ptr, int n = 512);
    ~Mix();
    int prediction(int p1, int p2, int cntxt);
    void update(int y);
};

Mix::Mix(int *weight_ptr, int n) : wt(weight_ptr), N(n), x1(0), x2(0), context(0), last_prediction(0)
{
    for (int i = 0; i < N * 2; i++)
        wt[i] = 1 << 23;
}

Mix::~Mix()
{
    wt = 0;
}

int Mix::prediction(int p1, int p2, int cntxt)
{
    assert(cntxt >= 0 && cntxt < N);
    context = cntxt * 2;
    return last_prediction = ((x1 = p1) * (wt[context] >> 16) + (x2 = p2) * (wt[context + 1] >> 16) + 128) >> 8;
}

void Mix::update(int y)
{
    assert(y == 0 || y == 1);
    int error = ((y << 12) - (*squash)(last_prediction));
    if ((wt[context] & 3) < 3)
    {
        error *= 4 - (++wt[context] & 3);
    }
    error = (error + 8) >> 4;
    wt[context] += x1 * error & -4;
    wt[context + 1] += x2 * error;
}

class APM : public Mix
{
public:
    APM(int *weight_ptr, int n);
};

APM::APM(int *weight_ptr, int n) : Mix(weight_ptr, n)
{
    for (int i = 0; i < n; i++)
    {
        wt[2 * i] = 0;
    }
}

//////////////////////////// HashTable /////////////////////////

template <int B>
class HashTable
{
    U8 *table;
    U8 *raw_table;
    const U32 N;

public:
    HashTable(int n, U8 *table_ptr);
    ~HashTable();
    U8 *operator[](U32 i);
};

template <int B>
HashTable<B>::HashTable(int n, U8 *table_ptr) : table(table_ptr), raw_table(0), N(n)
{
    assert(B >= 2 && (B & (B - 1)) == 0);
    assert(N >= B * 4 && (N & (N - 1)) == 0);
    raw_table = table;
    table += 64 - int(reinterpret_cast<uintptr_t>(table) & 63);
}

template <int B>
U8 *HashTable<B>::operator[](U32 i)
{
    i *= 123456791;
    i = i << 16 | i >> 16;
    i *= 234567891;
    int chk = i >> 24;
    i = i * B & N - B;
    if (table[i] == chk)
        return table + i;
    if (table[i ^ B] == chk)
        return table + (i ^ B);
    if (table[i ^ B * 2] == chk)
        return table + (i ^ B * 2);
    if (table[i + 1] > table[i + 1 ^ B] || table[i + 1] > table[i + 1 ^ B * 2])
        i ^= B;

    if (table[i + 1] > table[i + 1 ^ B ^ B * 2])
        i ^= B ^ B * 2;
    memset(table + i, 0, B);
    table[i] = chk;
    return table + i;
}

template <int B>
HashTable<B>::~HashTable()
{
    raw_table = table = 0;
}

////////////////////////// LZP /////////////////////////

inline bool isalpha_device(char ch)
{
    return (ch >= 'A' && ch <= 'Z') ||
           (ch >= 'a' && ch <= 'z');
}

inline char tolower_device(char ch)
{
    if (ch >= 'A' && ch <= 'Z')
        ch += 'a' - 'A';
    return ch;
}

class LZP
{
private:
    const size_t N, H;
    enum
    {
        MINLEN = 12
    };
    U8 *buffer;
    U32 *table;
    int match;
    size_t len;
    size_t pos;
    U32 hash;
    U32 hash1;
    U32 hash2;
    StateMap *statemap;
    APM *apm1, *apm2, *apm3;
    int literals, matches;

public:
    U32 word0, word1;
    LZP(size_t mem, StateMap *statemap1, U8 *buffer, U32 *table, APM *apm1, APM *apm2, APM *apm3);
    ~LZP();
    int predict_char();
    int context(int i);
    int context4()
    {
        return hash2;
    }
    int context8()
    {
        return hash1;
    }
    int probability();
    void update(int ch);
};

LZP::LZP(size_t mem, StateMap *statemap, U8 *buf, U32 *tab, APM *apm1, APM *apm2, APM *apm3) : N(mem / 8), H(mem / 32),
                                                                                              match(-1), len(0), pos(0), hash(0), hash1(0), hash2(0),
                                                                                              statemap(statemap), apm1(apm1), apm2(apm2), apm3(apm3),
                                                                                              literals(0), matches(0), word0(0), word1(0)
{
    assert(mem > 0);
    assert(H > 0);
    buffer = buf;
    table = tab;
}

LZP::~LZP()
{
    delete statemap;
    delete apm1;
    delete apm2;
    delete apm3;
    table = 0;
    buffer = 0;
}

int LZP::predict_char()
{
    return len >= MINLEN ? buffer[match & N - 1] : -1;
}

int LZP::context(int i)
{
    assert(i > 0);
    return buffer[pos - i & N - 1];
}

int LZP::probability()
{
    if (len < MINLEN)
        return 0;
    int cxt = static_cast<int>(len);
    if (len > 28)
        cxt = 28 + (len >= 32) + (len >= 64) + (len >= 128);
    int pc = predict_char();
    int pr = statemap->predict_next_bit(cxt);
    pr = (*stretch)(pr);
    pr = apm1->prediction(2048, pr * 2, hash2 * 256 + pc & 0xffff) * 3 + pr >> 2;
    pr = apm2->prediction(2048, pr * 2, hash1 * (11 << 6) + pc & 0x3ffff) * 3 + pr >> 2;
    pr = apm3->prediction(2048, pr * 2, hash1 * (7 << 4) + pc & 0xfffff) * 3 + pr >> 2;
    pr = (*squash)(pr);
    return pr;
}

void LZP::update(int ch)
{
    int y = predict_char() == ch;
    hash1 = hash1 * (3 << 4) + ch + 1;
    hash2 = hash2 << 8 | ch;
    hash = hash * (5 << 2) + ch + 1 & H - 1;
    if (len >= MINLEN)
    {
        statemap->update(y);
        apm1->update(y);
        apm2->update(y);
        apm3->update(y);
    }
    if (isalpha_device(ch))
        word0 = word0 * (29 << 2) + tolower_device(ch);
    else if (word0)
        word1 = word0, word0 = 0;
    buffer[pos & N - 1] = ch;
    ++pos;
    if (y)
    {
        ++len;
        ++match;
        ++matches;
    }
    else
    {
        ++literals;
        y = 0;
        len = 1;
        match = table[hash];
        if (!((match ^ pos) & N - 1))
            --match;
        while (len <= 128 && buffer[match - len & N - 1] == buffer[pos - len & N - 1])
            ++len;
        --len;
    }
    table[hash] = pos;
}

//////////////////////////// Predictor /////////////////////////
//
// The warp-cooperative GPU version spread the 7 HashTable lookups and
// 11 StateMap lookups across lanes 4-10 and 0-10 of a warp, with the
// Mix/APM chain serial on lane 0. On CPU all of that simply executes in
// sequence; the results are identical.

class Predictor
{
    enum
    {
        N = 11
    };
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
    LZP *lzp;

public:
    Predictor(U8 *context1_ptr, StateMap *statemap1[N], Mix *mix1[N - 1], APM *apm1, APM *apm2, APM *apm3, HashTable<16> *hashtable_ptr, LZP *lzp_ptr);
    ~Predictor();
    int predict_next_bit();
    void update(int y);
};

Predictor::Predictor(U8 *context1_ptr, StateMap *statemap1[N], Mix *mix1[N - 1], APM *apm1, APM *apm2, APM *apm3, HashTable<16> *hashtable_ptr, LZP *lzp_ptr) : c0(0), context1(context1_ptr), nibble(1), bcount(0),
                                                                                                                                                             apm1(apm1), apm2(apm2), apm3(apm3), hashtable(hashtable_ptr), lzp(lzp_ptr)
{
    for (int i = 0; i < N; ++i)
    {
        sp[i] = cp[i] = context1;
        statemap[i] = statemap1[i];
        if (i < N - 1)
            mix[i] = mix1[i];
    }
}

Predictor::~Predictor()
{
    for (int i = 0; i < N; ++i)
    {
        delete statemap[i];
        if (i < N - 1)
            delete mix[i];
    }
    delete apm1;
    delete apm2;
    delete apm3;
    delete hashtable;
    context1 = 0;
}

void Predictor::update(int y)
{
    assert(y == 0 || y == 1);

    if (c0 == 0)
    {
        c0 = 1 - y;
        return;
    }

    *sp[0] = nex(*sp[0], y);
    statemap[0]->update(y);
    for (int lane = 1; lane < N; ++lane)
    {
        *sp[lane] = nex(*sp[lane], y);
        statemap[lane]->update(y);
        mix[lane - 1]->update(y);
    }

    c0 += c0 + y;
    bcount++;
    if (bcount == 8)
        bcount = c0 = 0;
    if ((nibble += nibble + y) >= 16)
        nibble = 1;
    apm1->update(y);
    apm2->update(y);
    apm3->update(y);
}

int Predictor::predict_next_bit()
{
    if (c0 == 0)
    {
        return lzp->probability();
    }

    int pc = lzp->predict_char();
    int r = pc + 256 >> 8 - bcount == c0;
    U32 c4 = lzp->context4();
    U32 c8 = (lzp->context8() << 4) - 1;

    if ((bcount & 3) == 0)
    { // nibble boundary? update context pointers
        int pcr = pc & -r;
        U32 c4p = c4 << 8;

        if (bcount == 0)
        { // byte boundary? update order-1 context pointers
            cp[0] = context1 + (c4 >> 16 & 0xff00);
            cp[1] = context1 + (c4 >> 8 & 0xff00) + 0x10000;
            cp[2] = context1 + (c4 & 0xff00) + 0x20000;
            cp[3] = context1 + (c4 << 8 & 0xff00) + 0x30000;
        }

        // 7 heavy HashTable lookups (independent on GPU; serial here)
        cp[4] = (*hashtable)[(c4p & 0xffff00) - c0];
        cp[5] = (*hashtable)[(c4p & 0xffffff00) * 3 + c0];
        cp[6] = (*hashtable)[c4 * 7 + c0];
        cp[7] = (*hashtable)[(c8 * 5 & 0xfffffc) + c0];
        cp[8] = (*hashtable)[(c8 * 11 & 0xffffff0) + c0 + pcr * 13];
        cp[9] = (*hashtable)[(lzp->word0 * 5 + c0 + pcr * 17)];
        cp[10] = (*hashtable)[(lzp->word1 * 7 + lzp->word0 * 11 + c0 + pcr * 37)];
    }

    // 11 StateMap predict_next_bit() calls (independent on GPU; serial here)
    r <<= 8;
    int stretched_cache[N];
    sp[0] = &cp[0][c0];
    stretched_cache[0] = (*stretch)(statemap[0]->predict_next_bit(*sp[0]));
    for (int i = 1; i < N; ++i)
    {
        sp[i] = &cp[i][i < 4 ? c0 : nibble];
        int st = *sp[i];
        stretched_cache[i] = (*stretch)(statemap[i]->predict_next_bit(st));
    }

    // serial Mix + APM chain
    int pr = stretched_cache[0];
    for (int i = 1; i < N; ++i)
    {
        int st_i = *sp[i];
        int stretched_i = stretched_cache[i];
        pr = mix[i - 1]->prediction(pr, stretched_i, st_i + r) * 3 + pr >> 2;
    }
    pr = apm1->prediction(512, pr * 2, c0 + pc * 256 & 0xffff) * 3 + pr >> 2;
    pr = apm2->prediction(512, pr * 2, c4 << 8 & 0xff00 | c0) * 3 + pr >> 2;
    pr = apm3->prediction(512, pr * 2, c4 * 3 + c0 & 0xffff) * 3 + pr >> 2;
    pr = (*squash)(pr);
    return pr;
}

//////////////////////////// Encoder ////////////////////////////

class Encoder
{
private:
    const int mode;
    char *inout;
    size_t total_size;

    U32 x1, x2;
    U32 x;
    enum
    {
        BUFSIZE = 0x20000
    };
    U8 *buffer;
    size_t usize, csize;
    double usum, csum;

public:
    size_t iterator_size;
    Predictor *predictor;
    Encoder(int m, char *temp, unsigned char *buffer_ptr, size_t tsz, size_t itr, Predictor *pred);
    ~Encoder();
    bool flush();
    bool put4(U32 c);

    int code(int y = 0)
    {
        int p = predictor->predict_next_bit();
        assert(p >= 0 && p < 4096);
        p += p < 2048;

        U32 xmid = x1 + (x2 - x1 >> 12) * p + ((x2 - x1 & 0xfff) * p >> 12);
        assert(xmid >= x1 && xmid < x2);
        if (mode == DECOMPRESS)
            y = x <= xmid;
        y ? (x2 = xmid) : (x1 = xmid + 1);

        predictor->update(y);

        while (((x1 ^ x2) & 0xff000000) == 0)
        { // pass equal leading bytes of range
            if (mode == COMPRESS)
                buffer[csize++] = x2 >> 24;
            x1 <<= 8;
            x2 = (x2 << 8) + 255;
            if (mode == DECOMPRESS)
                x = (x << 8) + (inout[iterator_size++] & 255);
        }
        return y;
    }

    bool count()
    {
        assert(mode == COMPRESS);
        ++usize;
        if (csize > BUFSIZE - 256)
            return flush();
        return true;
    }
};

Encoder::Encoder(int m, char *temp, unsigned char *buffer_ptr, size_t tsz, size_t itr, Predictor *pred) : mode(m), inout(temp), buffer(buffer_ptr), total_size(tsz), iterator_size(itr), x1(0), x2(0xffffffff), x(0),
                                                                                                          usize(0), csize(0), usum(0), csum(0), predictor(pred)
{
    if (mode == DECOMPRESS)
    { // x = first 4 bytes of archive
        for (int i = 0; i < 4; ++i)
            x = (x << 8) + (inout[iterator_size++] & 255);
        csize = 4;
    }
}

Encoder::~Encoder()
{
    buffer = 0;
}

bool Encoder::put4(U32 c)
{
    if (iterator_size > total_size)
        return false;
    inout[iterator_size++] = char(c >> 24);
    if (iterator_size > total_size)
        return false;
    inout[iterator_size++] = char(c >> 16);
    if (iterator_size > total_size)
        return false;
    inout[iterator_size++] = char(c >> 8);
    if (iterator_size > total_size)
        return false;
    inout[iterator_size++] = char(c);
    return true;
}

bool Encoder::flush()
{
    if (mode == COMPRESS)
    {
        buffer[csize++] = x1 >> 24;
        buffer[csize++] = 255;
        buffer[csize++] = 255;
        buffer[csize++] = 255;
        if (!put4(usize))
            return false;
        if (!put4(csize))
            return false;
        for (int i = 0; i < csize; i++)
        {
            if (iterator_size > total_size)
                return false;
            inout[iterator_size++] = buffer[i];
        }
        usum += usize;
        csum += csize + 10;
        x1 = x = usize = csize = 0;
        x2 = 0xffffffff;
        return true;
    }
    return true;
}

size_t get4(size_t &itr, const char *in)
{
    size_t r = (unsigned char)in[itr++];
    r = r * 256 + (unsigned char)in[itr++];
    r = r * 256 + (unsigned char)in[itr++];
    r = r * 256 + (unsigned char)in[itr++];

    return r;
}

//////////////////////////// Host / IO ////////////////////////////

void put4_stream(U32 c, std::ostream &out)
{
    out.put((c >> 24) & 0xFF);
    out.put((c >> 16) & 0xFF);
    out.put((c >> 8) & 0xFF);
    out.put(c & 0xFF);
}

unsigned int get4_stream(std::istream &in)
{
    unsigned int r = in.get();
    r = r * 256 + in.get();
    r = r * 256 + in.get();
    r = r * 256 + in.get();
    return r;
}

void put8_stream(size_t c, std::ostream &out)
{
    out.put((c >> 56) & 0xFF);
    out.put((c >> 48) & 0xFF);
    out.put((c >> 40) & 0xFF);
    out.put((c >> 32) & 0xFF);
    out.put((c >> 24) & 0xFF);
    out.put((c >> 16) & 0xFF);
    out.put((c >> 8) & 0xFF);
    out.put(c & 0xFF);
}

size_t get8_stream(std::istream &in)
{
    size_t r = in.get();
    r = r * 256 + in.get();
    r = r * 256 + in.get();
    r = r * 256 + in.get();
    r = r * 256 + in.get();
    r = r * 256 + in.get();
    r = r * 256 + in.get();
    r = r * 256 + in.get();
    return r;
}

struct ThreadBuffers
{
    U32 *lzp_statemap;
    int *lzp_apm[3];
    U8 *lzp_buffer;
    U32 *lzp_table;

    U32 *predictor_statemap[11];
    int *predictor_mix[10];
    int *predictor_apm[3];
    U8 *predictor_hashtable;
    U8 *predictor_context1;

    unsigned char *encoder_buffer;
};

size_t calculateThreadBufferBytes(int memory_level)
{
    U32 MEM_host = 1U << (base_memory_level + memory_level);

    size_t total = 0;

    // LZP
    total += 0x200 * sizeof(U32);
    total += 0x20000 * sizeof(int);
    total += 0x80000 * sizeof(int);
    total += 0x200000 * sizeof(int);
    total += (MEM_host / 8) * sizeof(U8);
    total += (MEM_host / 32) * sizeof(U32);

    // Predictor
    total += 11 * (0x100 * sizeof(U32));
    total += 10 * (0x400 * sizeof(int));
    total += 3 * (0x20000 * sizeof(int));
    total += (MEM_host / 2 + 128) * sizeof(U8);
    total += 0x40000 * sizeof(U8);

    // Encoder scratch
    total += 0x20000 * sizeof(unsigned char);

    return total;
}

void allocateThreadBuffer(ThreadBuffers &b, int memory_level)
{
    U32 MEM_host = 1U << (base_memory_level + memory_level);

    b.lzp_statemap = new U32[0x200];
    b.lzp_apm[0] = new int[0x20000];
    b.lzp_apm[1] = new int[0x80000];
    b.lzp_apm[2] = new int[0x200000];

    b.lzp_buffer = new U8[MEM_host / 8];
    b.lzp_table = new U32[MEM_host / 32];

    for (int j = 0; j < 11; j++)
        b.predictor_statemap[j] = new U32[0x100];
    for (int j = 0; j < 10; j++)
        b.predictor_mix[j] = new int[0x400];

    b.predictor_apm[0] = new int[0x20000];
    b.predictor_apm[1] = new int[0x20000];
    b.predictor_apm[2] = new int[0x20000];

    b.predictor_hashtable = new U8[MEM_host / 2 + 128];
    b.predictor_context1 = new U8[0x40000];
    b.encoder_buffer = new unsigned char[0x20000];
}

void freeThreadBuffer(ThreadBuffers &b)
{
    delete[] b.lzp_statemap;
    delete[] b.lzp_apm[0];
    delete[] b.lzp_apm[1];
    delete[] b.lzp_apm[2];

    delete[] b.lzp_buffer;
    delete[] b.lzp_table;

    for (int j = 0; j < 11; j++)
        delete[] b.predictor_statemap[j];
    for (int j = 0; j < 10; j++)
        delete[] b.predictor_mix[j];

    delete[] b.predictor_apm[0];
    delete[] b.predictor_apm[1];
    delete[] b.predictor_apm[2];

    delete[] b.predictor_hashtable;
    delete[] b.predictor_context1;
    delete[] b.encoder_buffer;
}

// Zero the four raw buffers that the original init() kernel memset, then
// build this chunk's LZP + Predictor object graph (mirrors the second
// init() kernel of the CUDA version).
static void initThreadBuffer(ThreadBuffers &b, size_t mem)
{
    memset(b.predictor_hashtable, 0, mem / 2 + 128);
    memset(b.predictor_context1, 0, 0x40000);
    memset(b.lzp_buffer, 0, mem / 8);
    memset(b.lzp_table, 0, mem / 32 * sizeof(U32));
}

// Compress one chunk. Equivalent to the COMPRESS branch of paq9_cuda for
// a single chunk id.
static void compressChunk(const char *input, size_t input_size, char *output,
                          size_t &output_size, ThreadBuffers &tb, size_t mem)
{
    initThreadBuffer(tb, mem);

    StateMap *lzp_statemap = new StateMap(tb.lzp_statemap, 0x200);
    APM *lzp_apm1 = new APM(tb.lzp_apm[0], 0x10000);
    APM *lzp_apm2 = new APM(tb.lzp_apm[1], 0x40000);
    APM *lzp_apm3 = new APM(tb.lzp_apm[2], 0x100000);
    LZP *lzpp = new LZP(mem, lzp_statemap, tb.lzp_buffer, tb.lzp_table, lzp_apm1, lzp_apm2, lzp_apm3);

    StateMap *predictor_statemap[11];
    for (int j = 0; j < 11; j++)
        predictor_statemap[j] = new StateMap(tb.predictor_statemap[j], 0x100);
    Mix *predictor_mix[10];
    for (int j = 0; j < 10; j++)
        predictor_mix[j] = new Mix(tb.predictor_mix[j], 0x200);
    APM *predictor_apm1 = new APM(tb.predictor_apm[0], 0x10000);
    APM *predictor_apm2 = new APM(tb.predictor_apm[1], 0x10000);
    APM *predictor_apm3 = new APM(tb.predictor_apm[2], 0x10000);
    HashTable<16> *predictor_hashtable = new HashTable<16>(mem / 2, tb.predictor_hashtable);
    Predictor *predictor = new Predictor(tb.predictor_context1, predictor_statemap, predictor_mix, predictor_apm1, predictor_apm2, predictor_apm3, predictor_hashtable, lzpp);

    size_t itr = 0;
    Encoder encoder(COMPRESS, output, tb.encoder_buffer, input_size, itr, predictor);
    int store_mode = 0;

    output[encoder.iterator_size++] = '0';

    itr = 0;
    while (true)
    {
        if (itr >= input_size)
            break;
        int ch = (unsigned char)input[itr];
        itr++;

        int cp = lzpp->predict_char();
        if (ch == cp)
        {
            encoder.code(1);
        }
        else
        {
            for (int i = 8; i >= 0; --i)
                encoder.code(ch >> i & 1);
        }

        if (!encoder.count())
        {
            store_mode = 1;
            break;
        }
        lzpp->update(ch);
    }

    if (!encoder.flush())
        store_mode = 1;

    if (store_mode)
    {
        encoder.iterator_size = 0;
        output[encoder.iterator_size++] = '1';
        itr = 0;
        while (itr < input_size)
            output[encoder.iterator_size++] = input[itr++];
    }

    output_size = encoder.iterator_size;

    delete predictor; // cascades delete of statemaps/mix/apms/hashtable
    delete lzpp;      // cascades delete of lzp statemap/apms
}

// Decompress one chunk. Equivalent to the DECOMPRESS branch of paq9_cuda.
static void decompressChunk(const char *input, size_t input_size, char *output,
                            size_t &output_size, ThreadBuffers &tb, size_t mem)
{
    initThreadBuffer(tb, mem);

    StateMap *lzp_statemap = new StateMap(tb.lzp_statemap, 0x200);
    APM *lzp_apm1 = new APM(tb.lzp_apm[0], 0x10000);
    APM *lzp_apm2 = new APM(tb.lzp_apm[1], 0x40000);
    APM *lzp_apm3 = new APM(tb.lzp_apm[2], 0x100000);
    LZP *lzpp = new LZP(mem, lzp_statemap, tb.lzp_buffer, tb.lzp_table, lzp_apm1, lzp_apm2, lzp_apm3);

    StateMap *predictor_statemap[11];
    for (int j = 0; j < 11; j++)
        predictor_statemap[j] = new StateMap(tb.predictor_statemap[j], 0x100);
    Mix *predictor_mix[10];
    for (int j = 0; j < 10; j++)
        predictor_mix[j] = new Mix(tb.predictor_mix[j], 0x200);
    APM *predictor_apm1 = new APM(tb.predictor_apm[0], 0x10000);
    APM *predictor_apm2 = new APM(tb.predictor_apm[1], 0x10000);
    APM *predictor_apm3 = new APM(tb.predictor_apm[2], 0x10000);
    HashTable<16> *predictor_hashtable = new HashTable<16>(mem / 2, tb.predictor_hashtable);
    Predictor *predictor = new Predictor(tb.predictor_context1, predictor_statemap, predictor_mix, predictor_apm1, predictor_apm2, predictor_apm3, predictor_hashtable, lzpp);

    if (input[0] == '1')
    {
        int itr = 0, itr2 = 1;
        while (itr2 < input_size)
        {
            output[itr++] = input[itr2++];
        }
        output_size = itr;
    }
    else
    {
        size_t itr2_shared = 0; // running output offset
        size_t itr = 1;
        while (true)
        {
            if (itr >= input_size)
                break;

            size_t usize = get4(itr, input);
            get4(itr, input); // csize, discarded (as in original)

            Encoder encoder(DECOMPRESS, const_cast<char *>(input), tb.encoder_buffer, input_size, itr, predictor);

            size_t remaining = usize;
            while (remaining > 0)
            {
                --remaining;
                int cp = lzpp->predict_char();
                int first = encoder.code();
                if (first == 0)
                {
                    cp = 1;
                    while (cp < 256)
                        cp += cp + encoder.code();
                    cp &= 255;
                }
                output[itr2_shared++] = cp;
                lzpp->update(cp);
            }

            itr = encoder.iterator_size;
            output_size = itr2_shared;
        }
    }

    delete predictor;
    delete lzpp;
}

//////////////////////////// compress ////////////////////////////

void compress(char *destination_file, char *source_file)
{
    std::ifstream source(source_file, std::ios::binary);
    if (!source)
    {
        std::cerr << "Cannot open " << source_file << std::endl;
        exit(1);
    }
    source.seekg(0, std::ios::end);
    size_t total_B = source.tellg();
    source.clear();
    source.seekg(0, std::ios::beg);

    chunk_MB = (1 << (chunk_level - 1));
    size_t chunk_B = chunk_MB * MB;
    int num_of_chunks = (int)((total_B + chunk_B - 1) / chunk_B);

    int hw = (int)std::thread::hardware_concurrency();
    if (hw <= 0)
        hw = 1;
    int num_threads = std::min(hw, num_of_chunks);
    if (num_threads > MAX_THREADS)
        num_threads = MAX_THREADS;
    if (num_threads < 1)
        num_threads = 1;

    std::ofstream dest(destination_file, std::ios::binary);
    if (!dest)
    {
        std::cout << std::string(destination_file) << " does not created/opened.\n";
        exit(1);
    }

    dest.write("PAQ9-CUDA", 9);
    dest.put(1);
    dest.write(source_file, strlen(source_file));
    dest.put(0);
    dest.put('c');
    put8_stream(total_B, dest);
    put4_stream(chunk_MB, dest);
    put4_stream(memory_level, dest);
    put4_stream(chunk_level, dest);
    put4_stream(num_of_chunks, dest);

    std::cout << "Memory Chunk Level: " << chunk_MB << "MB" << std::endl;
    std::cout << "Memory Level: " << memory_level << std::endl;
    std::cout << "Level: " << chunk_level << std::endl;
    std::cout << "Number of Chunks: " << num_of_chunks << std::endl;
    std::cout << "Total threads: " << num_of_chunks << std::endl;
    std::cout << "CPU Threads: " << num_threads << std::endl;

    MEM = (size_t)1 << (base_memory_level + memory_level);

    // Per-thread model buffers
    std::vector<ThreadBuffers> tbs(num_threads);
    for (auto &tb : tbs)
        allocateThreadBuffer(tb, memory_level);

    std::vector<char *> input_buf(num_threads, nullptr);
    std::vector<char *> output_buf(num_threads, nullptr);
    for (int i = 0; i < num_threads; i++)
        output_buf[i] = new char[chunk_B + 2];

    std::vector<size_t> in_size(num_threads);
    std::vector<size_t> out_size(num_threads);

    total_compressed_size = 0;
    total_uncompressed_size = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int base = 0; base < num_of_chunks; base += num_threads)
    {
        int n = std::min(num_threads, num_of_chunks - base);

        for (int i = 0; i < n; i++)
        {
            size_t current_B = std::min(chunk_B, total_B - (base + i) * chunk_B);
            if (input_buf[i])
                delete[] input_buf[i];
            input_buf[i] = new char[current_B];
            source.read(input_buf[i], current_B);
            in_size[i] = current_B;
        }

        std::vector<std::thread> threads;
        threads.reserve(n);
        for (int i = 0; i < n; i++)
        {
            threads.emplace_back([i, &in_size, &out_size, &input_buf, &output_buf, &tbs, mem = MEM]() {
                compressChunk(input_buf[i], in_size[i], output_buf[i], out_size[i], tbs[i], mem);
            });
        }
        for (auto &t : threads)
            t.join();

        size_t total_input = 0, total_output = 0;
        for (int i = 0; i < n; i++)
        {
            put4_stream((U32)(in_size[i]), dest);
            put4_stream((U32)(out_size[i]), dest);
            dest.write(output_buf[i], out_size[i]);
            total_input += in_size[i];
            total_output += out_size[i];
        }
        total_compressed_size += total_output;
        total_uncompressed_size += total_input;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Total execution time: " << duration.count() << " ms" << std::endl;

    for (int i = 0; i < num_threads; i++)
    {
        if (input_buf[i])
            delete[] input_buf[i];
        delete[] output_buf[i];
    }

    for (auto &tb : tbs)
        freeThreadBuffer(tb);

    source.close();
    dest.close();

    std::cout << "Uncompressed  ->  Compressed\n";
    std::cout << "Total: " << total_uncompressed_size << " Byte -> " << total_compressed_size << " Byte" << std::endl;
    std::cout << "Compression Ratio: " << 1.0 * total_uncompressed_size / total_compressed_size << std::endl;
}

//////////////////////////// decompress ////////////////////////////

bool check_archive(std::istream &in)
{
    std::string magic = "PAQ9-CUDA";

    for (char c : magic)
    {
        if (in.get() != c)
            return false;
    }

    return in.get() == 1;
}

std::string get_file_name(std::istream &in)
{
    std::string file_name;
    char c;

    while (in.get(c) && c != '\0')
    {
        file_name += c;
    }

    return file_name;
}

char *get_input(std::istream &source, size_t size)
{
    char *input = new char[size];

    source.read(input, size);

    return input;
}

void decompress(const char *destination_file, const char *source_file)
{
    total_compressed_size = 0;
    total_uncompressed_size = 0;

    std::ifstream source(source_file, std::ios::binary);
    if (!source)
    {
        std::cerr << "Cannot open " << source_file << std::endl;
        exit(1);
    }
    source.seekg(0, std::ios::end);
    size_t total_B = source.tellg();
    source.clear();
    source.seekg(0, std::ios::beg);

    if (!check_archive(source))
    {
        std::cout << "This is not a PAQ9-CUDA compressed file.\n";
        exit(1);
    }

    std::string filename = get_file_name(source);

    if (destination_file == 0)
    {
        destination_file = filename.c_str();
    }

    char mode = source.get();
    if (mode == 's')
    {
        std::cout << "Run again and provide proper arguments.\n";
        exit(1);
    }
    else if (mode == 'c')
    {
        size_t usize = get8_stream(source);
        chunk_MB = get4_stream(source);
        memory_level = get4_stream(source);
        chunk_level = get4_stream(source);
        int num_of_chunks = get4_stream(source);

        std::cout << "Memory Chunk Level: " << chunk_MB << "MB" << std::endl;
        std::cout << "Memory Level: " << memory_level << std::endl;
        std::cout << "Level: " << chunk_level << std::endl;
        std::cout << "Number of Chunks: " << num_of_chunks << std::endl;

        MEM = (size_t)1 << (base_memory_level + memory_level);
        size_t chunk_B = chunk_MB * MB;

        int hw = (int)std::thread::hardware_concurrency();
        if (hw <= 0)
            hw = 1;
        int num_threads = std::min(hw, num_of_chunks);
        if (num_threads > MAX_THREADS)
            num_threads = MAX_THREADS;
        if (num_threads < 1)
            num_threads = 1;

        std::vector<ThreadBuffers> tbs(num_threads);
        for (auto &tb : tbs)
            allocateThreadBuffer(tb, memory_level);

        std::vector<char *> input_buf(num_threads, nullptr);
        std::vector<char *> output_buf(num_threads, nullptr);
        for (int i = 0; i < num_threads; i++)
        {
            input_buf[i] = new char[chunk_B + 2];
            output_buf[i] = new char[chunk_B + 2];
        }

        std::vector<size_t> in_size(num_threads);
        std::vector<size_t> out_size(num_threads);

        std::ofstream dest(destination_file, std::ios::binary);
        if (!dest)
        {
            std::cout << std::string(destination_file) << " does not created/opened.\n";
            exit(1);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int base = 0; base < num_of_chunks; base += num_threads)
        {
            int n = std::min(num_threads, num_of_chunks - base);

            for (int i = 0; i < n; i++)
            {
                get4_stream(source); // uncompressed chunk size (usize)
                in_size[i] = get4_stream(source);
                source.read(input_buf[i], in_size[i]);
            }

            std::vector<std::thread> threads;
            threads.reserve(n);
            for (int i = 0; i < n; i++)
            {
                threads.emplace_back([i, &in_size, &out_size, &input_buf, &output_buf, &tbs, mem = MEM]() {
                    decompressChunk(input_buf[i], in_size[i], output_buf[i], out_size[i], tbs[i], mem);
                });
            }
            for (auto &t : threads)
                t.join();

            size_t total_input = 0, total_output = 0;
            for (int i = 0; i < n; i++)
            {
                dest.write(output_buf[i], out_size[i]);
                total_input += in_size[i];
                total_output += out_size[i];
            }
            total_compressed_size += total_input;
            total_uncompressed_size += total_output;
        }

        for (int i = 0; i < num_threads; i++)
        {
            delete[] input_buf[i];
            delete[] output_buf[i];
        }

        for (auto &tb : tbs)
            freeThreadBuffer(tb);

        source.close();
        dest.close();

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = end_time - start_time;
        std::cout << "Total Execution Time: " << elapsed_time.count() << " seconds" << std::endl;
        std::cout << "Compressed  ->  Decompressed \n";
        std::cout << "Total: " << total_compressed_size << " Byte -> "
                  << total_uncompressed_size << " Byte" << std::endl;
    }
    else
    {
        std::cout << "Run again and provide proper arguments.\n";
        exit(1);
    }
}

const char *get_file_name(const char *path)
{
    const char *slash_pos = strrchr(path, '/');
    if (slash_pos)
        return slash_pos + 1;

#ifdef _WIN32
    const char *backslash_pos = strrchr(path, '\\');
    if (backslash_pos)
        return backslash_pos + 1;
#endif

    return path;
}

void print_usage(const char *prog_name)
{
    const char *file_name = get_file_name(prog_name);

    std::cout << "Usage:\n";
    std::cout << "  Compress:   " << file_name << " -c [-<memory_level>] <destination_file> [-<chunk_level>] <source_file>\n";
    std::cout << "  Decompress: " << file_name << " -d <source_file> <destination_file>\n\n";

    std::cout << "  <memory_level> and <chunk_level> must be between 1 and 11.\n";
    std::cout << "  If not given, or out of bounds, both default to 1.\n\n";

    std::cout << "  memory_level: controls how much memory is used.\n";
    std::cout << "  chunk_level: controls compression ratio vs. speed.\n\n";

    std::cout << "Examples:\n";
    std::cout << "  " << file_name << " -c -8 output.paq -8 input.txt\n";
    std::cout << "  " << file_name << " -d output.paq input.txt\n\n";
    std::cout << "Note: [] is optional.\n";
}

int main(int argc, char **args)
{
    auto start = std::chrono::steady_clock::now();
    std::cout << "CPU version of PAQ9 (multithreaded) started successfully.\n\n";

    if (argc < 3)
    {
        print_usage(args[0]);
        exit(1);
    }

    // one-time global init (read-only after this point; safe to share
    // across the worker threads).
    squash = new Squash();
    stretch = new Stretch(*squash);

    int mode;
    char *destination_file_name = 0;
    char *source_file_name = 0;
    if (args[1][0] == '-')
    {
        if (args[1][1] == 'c')
            mode = COMPRESS;
        else if (args[1][1] == 'd')
            mode = DECOMPRESS;
        else
        {
            print_usage(args[0]);
            exit(1);
        }
    }
    else
    {
        print_usage(args[0]);
        exit(1);
    }

    std::cout << "Working mode: "
              << (mode == COMPRESS ? "Compressing" : "Decompressing")
              << std::endl;

    int ind = 2;
    if (mode == COMPRESS)
    {
        if (ind < argc && args[ind][0] == '-')
        {
            std::string temp;
            size_t len = strlen(args[ind]);
            for (int i = 1; i < (int)len; i++)
            {
                if (isdigit((unsigned char)args[ind][i]))
                    temp += args[ind][i];
                else
                {
                    print_usage(args[0]);
                    exit(1);
                }
            }
            memory_level = stoi(temp);
            if (memory_level < 1 || memory_level > 11)
            {
                memory_level = 1;
                std::cout << "Your provided memory level is not supported. It is set to default value 1.\n";
            }
            ind++;
        }
        else if (ind >= argc)
        {
            print_usage(args[0]);
            exit(1);
        }

        if (ind < argc)
        {
            destination_file_name = args[ind];
            ind++;
        }
        else
        {
            print_usage(args[0]);
            exit(1);
        }

        if (ind < argc && args[ind][0] == '-')
        {
            std::string temp;
            size_t len = strlen(args[ind]);
            for (int i = 1; i < (int)len; i++)
            {
                if (isdigit((unsigned char)args[ind][i]))
                    temp += args[ind][i];
                else
                {
                    print_usage(args[0]);
                    exit(1);
                }
            }
            chunk_level = stoi(temp);
            if (chunk_level < 1 || chunk_level > 11)
            {
                chunk_level = 1;
                std::cout << "Your provided level is not supported. It is set to default value 1.\n";
            }
            ind++;
        }
        else if (ind >= argc)
        {
            print_usage(args[0]);
            exit(1);
        }

        if (ind < argc)
        {
            source_file_name = args[ind];
            ind++;
        }
        else
        {
            print_usage(args[0]);
            exit(1);
        }

        compress(destination_file_name, source_file_name);
    }
    else
    {
        if (ind < argc)
        {
            source_file_name = args[ind];
            ind++;
        }
        else
        {
            print_usage(args[0]);
            exit(1);
        }
        if (ind < argc)
        {
            destination_file_name = args[ind];
            ind++;
        }

        decompress(destination_file_name, source_file_name);
    }

    delete squash;
    delete stretch;

    auto end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();

    std::cout << "Time Taken: " << seconds << " seconds\n";
    std::cout << "Compression/Decompression Speed: " << total_uncompressed_size / seconds / 1024 << " KB/seconds \n";

    return 0;
}