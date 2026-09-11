#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <chrono>
#include <cstring>
#include <cassert>
#include <cctype>
#include <algorithm>

int memory_level = 7;
int memory_chunk_size = 8;

#define COMPRESS 0
#define DECOMPRESS 1

// ============================================================
// FIX: one pointer slot per CUDA thread
// Current kernel uses at most 256 threads.
// ============================================================
constexpr int MAX_DEVICE_THREADS = 1024;

class Alloc;
class LZP;
class Predictor;

__device__ __forceinline__ int device_thread_id()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

// ============================================================
// Alloc
// ============================================================

class Alloc
{
    long long total_allocated_size;

public:
    __device__ Alloc()
    {
        total_allocated_size = 0;
    }

    template <class T>
    __device__ void alloc(T *&p, int allocate_size)
    {
        p = new T[allocate_size]();

        if (!p)
        {
            printf("Error: Out of memory (failed to allocate %d bytes)\n",
                   allocate_size * (int)sizeof(T));

            total_allocated_size = -1;
            return;
        }

        total_allocated_size +=
            (long long)allocate_size * sizeof(T);
    }
};

// ============================================================
// FIX: per-thread allocator instead of one shared allocator
// ============================================================

__device__ Alloc *allocator_pool[MAX_DEVICE_THREADS];

#define allocator (allocator_pool[device_thread_id()])

// 8, 16, 32 bit unsigned types
typedef unsigned char U8;
typedef unsigned short U16;
typedef unsigned int U32;

// ============================================================
// Squash
// ============================================================

class Squash
{
    short tab[4096];

public:
    __device__ Squash();
    __device__ int operator()(int d);
};

__device__ Squash::Squash()
{
    static const int t[33] = {
        1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747,
        1101, 1546, 2047, 2549, 2994, 3348, 3607, 3785, 3901,
        3975, 4022, 4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094};

    for (int i = -2048; i < 2048; ++i)
    {
        int w = i & 127;
        int d = (i >> 7) + 16;

        tab[i + 2048] =
            (t[d] * (128 - w) +
             t[d + 1] * w + 64) >>
            7;
    }
}

__device__ int Squash::operator()(int d)
{
    d += 2048;

    if (d < 0)
        return 0;
    else if (d > 4095)
        return 4095;
    else
        return tab[d];
}

__device__ Squash *squash;

// ============================================================
// Stretch
// ============================================================

class Stretch
{
    short t[4096];

public:
    __device__ Stretch();
    __device__ int operator()(int p) const;
};

__device__ Stretch::Stretch()
{
    int pi = 0;

    for (int x = -2047; x <= 2047; ++x)
    {
        int i = squash->operator()(x);

        for (int j = pi; j <= i; ++j)
            t[j] = x;

        pi = i + 1;
    }

    t[4095] = 2047;
}

__device__ int Stretch::operator()(int p) const
{
    assert(p >= 0 && p < 4096);
    return t[p];
}

__device__ Stretch *stretch;

// ============================================================
// ilog
// ============================================================

class Ilog
{
    U8 *t;

public:
    __device__ Ilog();
    __device__ int operator()(U16 x) const;
    __device__ int operator()(U32 x) const;
};

__device__ Ilog::Ilog()
{
    allocator->alloc(t, 65536);

    U32 x = 14155776;

    for (int i = 2; i < 65536; ++i)
    {
        x += 774541002 / (i * 2 - 1);
        t[i] = x >> 24;
    }
}

__device__ int Ilog::operator()(U16 x) const
{
    return t[x];
}
__device__ Ilog *ilog;
__device__ int Ilog::operator()(U32 x) const
{
    if (x >= 0x1000000)
        return 256 + ilog->operator()(x >> 16);
    else if (x >= 0x10000)
        return 128 + ilog->operator()(x >> 8);
    else
        return ilog->operator()(x);
}

// ============================================================
// State table
// ============================================================

__device__ static const U8 State_table[256][2] = {
    {1, 2}, {3, 5}, {4, 6}, {7, 10}, {8, 12}, {9, 13}, {11, 14}, {15, 19}, {16, 23}, {17, 24}, {18, 25}, {20, 27}, {21, 28}, {22, 29}, {26, 30}, {31, 33}, {32, 35}, {32, 35}, {32, 35}, {32, 35}, {34, 37}, {34, 37}, {34, 37}, {34, 37}, {34, 37}, {34, 37}, {36, 39}, {36, 39}, {36, 39}, {38, 40}, {41, 43}, {42, 45}, {42, 45}, {44, 47}, {44, 47}, {46, 49}, {46, 49}, {48, 51}, {48, 51}, {50, 52}, {53, 43}, {54, 57}, {54, 57}, {56, 59}, {56, 59}, {58, 61}, {58, 61}, {60, 63}, {60, 63}, {62, 65}, {62, 65}, {50, 66}, {67, 55}, {68, 57}, {68, 57}, {70, 73}, {70, 73}, {72, 75}, {72, 75}, {74, 77}, {74, 77}, {76, 79}, {76, 79}, {62, 81}, {62, 81}, {64, 82}, {83, 69}, {84, 71}, {84, 71}, {86, 73}, {86, 73}, {44, 59}, {44, 59}, {58, 61}, {58, 61}, {60, 49}, {60, 49}, {76, 89}, {76, 89}, {78, 91}, {78, 91}, {80, 92}, {93, 69}, {94, 87}, {94, 87}, {96, 45}, {96, 45}, {48, 99}, {48, 99}, {88, 101}, {88, 101}, {80, 102}, {103, 69}, {104, 87}, {104, 87}, {106, 57}, {106, 57}, {62, 109}, {62, 109}, {88, 111}, {88, 111}, {80, 112}, {113, 85}, {114, 87}, {114, 87}, {116, 57}, {116, 57}, {62, 119}, {62, 119}, {88, 121}, {88, 121}, {90, 122}, {123, 85}, {124, 97}, {124, 97}, {126, 57}, {126, 57}, {62, 129}, {62, 129}, {98, 131}, {98, 131}, {90, 132}, {133, 85}, {134, 97}, {134, 97}, {136, 57}, {136, 57}, {62, 139}, {62, 139}, {98, 141}, {98, 141}, {90, 142}, {143, 95}, {144, 97}, {144, 97}, {68, 57}, {68, 57}, {62, 81}, {62, 81}, {98, 147}, {98, 147}, {100, 148}, {149, 95}, {150, 107}, {150, 107}, {108, 151}, {108, 151}, {100, 152}, {153, 95}, {154, 107}, {108, 155}, {100, 156}, {157, 95}, {158, 107}, {108, 159}, {100, 160}, {161, 105}, {162, 107}, {108, 163}, {110, 164}, {165, 105}, {166, 117}, {118, 167}, {110, 168}, {169, 105}, {170, 117}, {118, 171}, {110, 172}, {173, 105}, {174, 117}, {118, 175}, {110, 176}, {177, 105}, {178, 117}, {118, 179}, {110, 180}, {181, 115}, {182, 117}, {118, 183}, {120, 184}, {185, 115}, {186, 127}, {128, 187}, {120, 188}, {189, 115}, {190, 127}, {128, 191}, {120, 192}, {193, 115}, {194, 127}, {128, 195}, {120, 196}, {197, 115}, {198, 127}, {128, 199}, {120, 200}, {201, 115}, {202, 127}, {128, 203}, {120, 204}, {205, 115}, {206, 127}, {128, 207}, {120, 208}, {209, 125}, {210, 127}, {128, 211}, {130, 212}, {213, 125}, {214, 137}, {138, 215}, {130, 216}, {217, 125}, {218, 137}, {138, 219}, {130, 220}, {221, 125}, {222, 137}, {138, 223}, {130, 224}, {225, 125}, {226, 137}, {138, 227}, {130, 228}, {229, 125}, {230, 137}, {138, 231}, {130, 232}, {233, 125}, {234, 137}, {138, 235}, {130, 236}, {237, 125}, {238, 137}, {138, 239}, {130, 240}, {241, 125}, {242, 137}, {138, 243}, {130, 244}, {245, 135}, {246, 137}, {138, 247}, {140, 248}, {249, 135}, {250, 69}, {80, 251}, {140, 252}, {249, 135}, {250, 69}, {80, 251}, {140, 252}, {0, 0}, {0, 0}, {0, 0}};

#define nex(state, sel) State_table[state][sel]

// ============================================================
// StateMap
// ============================================================

__device__ int StateMap_dt[1024];

class StateMap
{
protected:
    const int N;
    int cntxt;
    U32 *prediction_table;

public:
    __device__ StateMap(int n = 256);
    __device__ void update(int y, int limit = 255);
    __device__ int predict_next_bit(int cntx);
};

__device__ StateMap::StateMap(int n)
    : N(n), cntxt(0)
{
    allocator->alloc(prediction_table, N);

    for (int i = 0; i < N; i++)
        prediction_table[i] = 2147483648U;

    if (StateMap_dt[0] == 0)
    {
        for (int i = 0; i < 1024; i++)
            StateMap_dt[i] = 16384 / (i + i + 3);
    }
}

__device__ void StateMap::update(int y, int limit)
{
    assert(cntxt >= 0 && cntxt < N);

    int n = prediction_table[cntxt] & 1023;
    int p = prediction_table[cntxt] >> 10;

    if (n < limit)
        prediction_table[cntxt]++;
    else
        prediction_table[cntxt] =
            (prediction_table[cntxt] & 0xfffffc00) | limit;

    prediction_table[cntxt] +=
        (((y << 22) - p) >> 3) *
            StateMap_dt[n] &
        0xfffffc00;
}

__device__ int StateMap::predict_next_bit(int cntx)
{
    assert(cntx >= 0 && cntx < N);

    return prediction_table[cntxt = cntx] >> 20;
}

// ============================================================
// Mix / APM
// ============================================================

class Mix
{
protected:
    const int N;
    int *wt;
    int x1, x2;
    int context;
    int last_prediction;

public:
    __device__ Mix(int n = 512);
    __device__ int prediction(int p1, int p2, int cntxt);
    __device__ void update(int y);
};

__device__ Mix::Mix(int n)
    : N(n),
      x1(0),
      x2(0),
      context(0),
      last_prediction(0)
{
    allocator->alloc(wt, n * 2);

    for (int i = 0; i < N * 2; i++)
        wt[i] = 1 << 23;
}

__device__ int Mix::prediction(int p1, int p2, int cntxt)
{
    assert(cntxt >= 0 && cntxt < N);

    context = cntxt * 2;

    return last_prediction =
               ((x1 = p1) * (wt[context] >> 16) +
                (x2 = p2) * (wt[context + 1] >> 16) +
                128) >>
               8;
}

__device__ void Mix::update(int y)
{
    assert(y == 0 || y == 1);

    int error =
        ((y << 12) -
         squash->operator()(last_prediction));

    if ((wt[context] & 3) < 3)
    {
        error *=
            4 - (++wt[context] & 3);
    }

    error = (error + 8) >> 4;

    wt[context] += x1 * error & -4;
    wt[context + 1] += x2 * error;
}

class APM : public Mix
{
public:
    __device__ APM(int n);
};

__device__ APM::APM(int n)
    : Mix(n)
{
    for (int i = 0; i < n; i++)
        wt[2 * i] = 0;
}

// ============================================================
// HashTable
// ============================================================

template <int B>
class HashTable
{
    U8 *table;
    const U32 N;

public:
    __device__ HashTable(int n);
    __device__ ~HashTable();
    __device__ U8 *operator[](U32 i);
};

template <int B>
__device__ HashTable<B>::HashTable(int n)
    : table(0), N(n)
{
    assert(B >= 2 && (B & (B - 1)) == 0);
    assert(N >= B * 4 && (N & (N - 1)) == 0);

    allocator->alloc(
        table,
        N + B * 4 + 64);

    table +=
        64 - int(((long)table) & 63);
}

template <int B>
__device__ U8 *HashTable<B>::operator[](U32 i)
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

    if (table[i + 1] >
            table[i + 1 ^ B] ||
        table[i + 1] >
            table[i + 1 ^ B * 2])
    {
        i ^= B;
    }

    if (table[i + 1] >
        table[i + 1 ^ B ^ B * 2])
    {
        i ^= B ^ B * 2;
    }

    memset(table + i, 0, B);

    table[i] = chk;

    return table + i;
}

template <int B>
__device__ HashTable<B>::~HashTable()
{
    int c = 0;
    int c0 = 0;

    for (U32 i = 0; i < N; ++i)
    {
        if (table[i])
        {
            ++c;

            if (i % B == 0)
                ++c0;
        }
    }

    printf(
        "HashTable<%d> %1.4f%% full, %1.4f%% utilized of %d KiB\n",
        B,
        100.0 * c0 * B / N,
        100.0 * c / N,
        N >> 10);
}

// ============================================================
// LZP
// ============================================================

__device__ U32 MEM = 1 << 22;

__device__ inline bool isalpha_device(char ch)
{
    return (ch >= 'A' && ch <= 'Z') ||
           (ch >= 'a' && ch <= 'z');
}

__device__ inline char tolower_device(char ch)
{
    if (ch >= 'A' && ch <= 'Z')
        ch += 'a' - 'A';

    return ch;
}

class LZP
{
private:
    const int N, H;

    enum
    {
        MINLEN = 12
    };

    U8 *buf;
    U32 *table;

    int match;
    int len;
    int pos;

    U32 hash;
    U32 hash1;
    U32 hash2;

    StateMap statemap1;

    APM apm1;
    APM apm2;
    APM apm3;

    int literals;
    int matches;

public:
    U32 word0, word1;

    __device__ LZP();
    __device__ ~LZP();

    __device__ int predict_char();
    __device__ int context(int i);

    __device__ int context4()
    {
        return hash2;
    }

    __device__ int context8()
    {
        return hash1;
    }

    __device__ int probability();
    __device__ void update(int ch);
};

__device__ LZP::LZP()
    : N(MEM / 8),
      H(MEM / 32),
      match(-1),
      len(0),
      pos(0),
      hash(0),
      hash1(0),
      hash2(0),
      statemap1(0x200),
      apm1(0x10000),
      apm2(0x40000),
      apm3(0x100000),
      literals(0),
      matches(0),
      word0(0),
      word1(0)
{
    assert(MEM > 0);
    assert(H > 0);

    allocator->alloc(buf, N);
    allocator->alloc(table, H);
}

__device__ LZP::~LZP()
{
    int c = 0;

    for (int i = 0; i < H; ++i)
        c += (table[i] != 0);

    printf(
        "LZP hash table %1.4f%% full of %d KiB\n"
        "LZP buffer %1.4f%% full of %d KiB\n",
        100.0 * c / H,
        H >> 8,
        pos < N ? 100.0 * pos / N : 100.0,
        N >> 10);

    printf(
        "LZP %d literals, %d matches (%1.4f%% matched)\n",
        literals,
        matches,
        literals + matches > 0
            ? 100.0 * matches / (literals + matches)
            : 0.0);
}

__device__ int LZP::predict_char()
{
    return len >= MINLEN
               ? buf[match & (N - 1)]
               : -1;
}

__device__ int LZP::context(int i)
{
    assert(i > 0);

    return buf[(pos - i) &
               (N - 1)];
}

__device__ int LZP::probability()
{
    if (len < MINLEN)
        return 0;

    int cxt = len;

    if (len > 28)
        cxt =
            28 +
            (len >= 32) +
            (len >= 64) +
            (len >= 128);

    int pc = predict_char();

    int pr =
        statemap1.predict_next_bit(cxt);

    pr =
        stretch->operator()(pr);

    pr =
        apm1.prediction(
            2048,
            pr * 2,
            hash2 * 256 + pc & 0xffff) *
                3 +
            pr >>
        2;

    pr =
        apm2.prediction(
            2048,
            pr * 2,
            hash1 * (11 << 6) +
                    pc &
                0x3ffff) *
                3 +
            pr >>
        2;

    pr =
        apm3.prediction(
            2048,
            pr * 2,
            hash1 * (7 << 4) +
                    pc &
                0xfffff) *
                3 +
            pr >>
        2;

    pr =
        squash->operator()(pr);

    return pr;
}

__device__ void LZP::update(int ch)
{
    int y =
        predict_char() == ch;

    hash1 =
        hash1 * (3 << 4) +
        ch + 1;

    hash2 =
        hash2 << 8 |
        ch;

    hash =
        hash * (5 << 2) +
            ch + 1 &
        H - 1;

    if (len >= MINLEN)
    {
        statemap1.update(y);

        apm1.update(y);
        apm2.update(y);
        apm3.update(y);
    }

    if (isalpha_device(ch))
    {
        word0 =
            word0 * (29 << 2) +
            tolower_device(ch);
    }
    else if (word0)
    {
        word1 = word0;
        word0 = 0;
    }

    buf[pos &
        (N - 1)] = ch;

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

        if (!((match ^ pos) &
              (N - 1)))
            --match;

        while (
            len <= 128 &&
            buf[(match - len) &
                (N - 1)] ==
                buf[(pos - len) &
                    (N - 1)])
        {
            ++len;
        }

        --len;
    }

    table[hash] = pos;
}

// ============================================================
// FIX: per-thread LZP pointer
// ============================================================

__device__ LZP *lzp_pool[MAX_DEVICE_THREADS];

#define lzp (lzp_pool[device_thread_id()])

// ============================================================
// Predictor
// ============================================================

class Predictor
{
    enum
    {
        N = 11
    };

    int c0;
    int nibble;
    int bcount;

    HashTable<16> hashtable;
    StateMap statemap[N];

    U8 *cp[N];
    U8 *sp[N];

    Mix mix[N - 1];

    APM apm1;
    APM apm2;
    APM apm3;

    U8 *context1;

public:
    __device__ Predictor();
    __device__ int predict_next_bit();
    __device__ void update(int y);
};

__device__ Predictor::Predictor()
    : c0(0),
      nibble(1),
      bcount(0),
      hashtable(MEM / 2),
      apm1(0x10000),
      apm2(0x10000),
      apm3(0x10000)
{
    allocator->alloc(
        context1,
        0x40000);

    for (int i = 0; i < N; ++i)
        sp[i] =
            cp[i] =
                context1;
}

__device__ void Predictor::update(int y)
{
    assert(y == 0 || y == 1);
    assert(bcount >= 0 && bcount < 8);
    assert(c0 >= 0 && c0 < 256);
    assert(nibble >= 1 && nibble <= 15);

    if (c0 == 0)
    {
        c0 = 1 - y;
    }
    else
    {
        *sp[0] =
            nex(*sp[0], y);

        statemap[0].update(y);

        for (int i = 1; i < N; ++i)
        {
            *sp[i] =
                nex(*sp[i], y);

            statemap[i].update(y);
            statemap[i - 1].update(y);
        }

        c0 += c0 + y;

        if (++bcount == 8)
            bcount = c0 = 0;

        if ((nibble += nibble + y) >= 16)
            nibble = 1;

        apm1.update(y);
        apm2.update(y);
        apm3.update(y);
    }
}

__device__ int Predictor::predict_next_bit()
{
    assert(lzp);

    if (c0 == 0)
    {
        return lzp->probability();
    }
    else
    {
        int pc =
            lzp->predict_char();

        int r =
            pc + 256 >>
                8 - bcount ==
            c0;

        U32 c4 =
            lzp->context4();

        U32 c8 =
            (lzp->context8() << 4) - 1;

        if ((bcount & 3) == 0)
        {
            pc &= -r;

            U32 c4p =
                c4 << 8;

            if (bcount == 0)
            {
                cp[0] =
                    context1 +
                    (c4 >> 16 & 0xff00);

                cp[1] =
                    context1 +
                    (c4 >> 8 & 0xff00) +
                    0x10000;

                cp[2] =
                    context1 +
                    (c4 & 0xff00) +
                    0x20000;

                cp[3] =
                    context1 +
                    (c4 << 8 & 0xff00) +
                    0x30000;
            }

            cp[4] =
                hashtable[(c4p & 0xffff00) -
                          c0];

            cp[5] =
                hashtable[(c4p & 0xffffff00) *
                              3 +
                          c0];

            cp[6] =
                hashtable[c4 * 7 +
                          c0];

            cp[7] =
                hashtable[(c8 * 5 & 0xfffffc) +
                          c0];

            cp[8] =
                hashtable[(c8 * 11 & 0xffffff0) +
                          c0 +
                          pc * 13];

            cp[9] =
                hashtable[lzp->word0 * 5 +
                          c0 +
                          pc * 17];

            cp[10] =
                hashtable[lzp->word1 * 7 +
                          lzp->word0 * 11 +
                          c0 +
                          pc * 37];
        }

        r <<= 8;

        sp[0] =
            &cp[0][c0];

        int pr =
            stretch->operator()(
                statemap[0]
                    .predict_next_bit(
                        *sp[0]));

        for (int i = 1; i < N; ++i)
        {
            sp[i] =
                &cp[i][i < 4
                           ? c0
                           : nibble];

            int st = *sp[i];

            pr =
                mix[i - 1].prediction(
                    pr,
                    stretch->operator()(
                        statemap[i]
                            .predict_next_bit(st)),
                    st + r) *
                        3 +
                    pr >>
                2;
        }

        pr =
            apm1.prediction(
                512,
                pr * 2,
                c0 + pc * 256 &
                    0xffff) *
                    3 +
                pr >>
            2;

        pr =
            apm2.prediction(
                512,
                pr * 2,
                c4 << 8 & 0xff00 |
                    c0) *
                    3 +
                pr >>
            2;

        pr =
            apm3.prediction(
                512,
                pr * 2,
                c4 * 3 + c0 &
                    0xffff) *
                    3 +
                pr >>
            2;

        return squash->operator()(pr);
    }
}

// ============================================================
// FIX: per-thread Predictor pointer
// ============================================================

__device__ Predictor *predictor_pool[MAX_DEVICE_THREADS];

#define predictor (predictor_pool[device_thread_id()])

// ============================================================
// Encoder
// ============================================================

class Encoder
{
private:
    const int mode;

    char *inout;
    int total_size;

    U32 x1, x2;
    U32 x;

    enum
    {
        BUFSIZE = 0x20000
    };

    unsigned char *buf;

    int usize, csize;

    double usum, csum;

public:
    int iterator_size;

    __device__ Encoder(
        int m,
        char *temp,
        int tsz,
        int itr);

    __device__ void flush();
    __device__ void put4(U32 c);

    __device__ int code(int y = 0)
    {
        assert(predictor);

        int p =
            predictor->predict_next_bit();

        assert(p >= 0 && p < 4096);

        p += p < 2048;

        U32 xmid =
            x1 +
            (x2 - x1 >> 12) *
                p +
            ((x2 - x1 & 0xfff) *
                 p >>
             12);

        assert(
            xmid >= x1 &&
            xmid < x2);

        if (mode == DECOMPRESS)
            y = x <= xmid;

        y
            ? (x2 = xmid)
            : (x1 = xmid + 1);

        predictor->update(y);

        while (
            ((x1 ^ x2) &
             0xff000000) == 0)
        {
            if (mode == COMPRESS)
                buf[csize++] =
                    x2 >> 24;

            x1 <<= 8;

            x2 =
                (x2 << 8) + 255;

            if (mode == DECOMPRESS)
            {
                x =
                    (x << 8) +
                    inout[iterator_size++];
            }
        }

        return y;
    }

    __device__ void count()
    {
        assert(mode == COMPRESS);

        ++usize;

        if (csize > BUFSIZE - 256)
            flush();
    }
};

__device__ Encoder::Encoder(
    int m,
    char *temp,
    int tsz,
    int itr)
    : mode(m),
      inout(temp),
      total_size(tsz),
      iterator_size(itr),
      x1(0),
      x2(0xffffffff),
      x(0),
      buf(nullptr),
      usize(0),
      csize(0),
      usum(0),
      csum(0)
{
    if (mode == DECOMPRESS)
    {
        for (int i = 0; i < 4; ++i)
        {
            x =
                (x << 8) +
                (inout[iterator_size++] &
                 255);
        }

        csize = 4;
    }
    else
    {
        // FIX:
        // Every Encoder now has its own buffer.
        allocator->alloc(
            buf,
            BUFSIZE);
    }
}

__device__ void Encoder::put4(U32 c)
{
    inout[iterator_size++] =
        char(c >> 24);

    inout[iterator_size++] =
        char(c >> 16);

    inout[iterator_size++] =
        char(c >> 8);

    inout[iterator_size++] =
        char(c);
}

__device__ void Encoder::flush()
{
    if (mode == COMPRESS)
    {
        buf[csize++] =
            x1 >> 24;

        buf[csize++] = 255;
        buf[csize++] = 255;
        buf[csize++] = 255;

        inout[iterator_size++] = 0;
        inout[iterator_size++] = 'c';

        put4(usize);
        put4(csize);

        for (int i = 0; i < csize; i++)
            inout[iterator_size++] =
                buf[i];

        usum += usize;
        csum += csize + 10;

        x1 = 0;
        x = 0;
        usize = 0;
        csize = 0;
        x2 = 0xffffffff;
    }
}

__device__ int get4(
    int &itr,
    const char *in)
{
    int r =
        (unsigned char)
            in[itr++];

    r =
        r * 256 +
        (unsigned char)
            in[itr++];

    r =
        r * 256 +
        (unsigned char)
            in[itr++];

    r =
        r * 256 +
        (unsigned char)
            in[itr++];

    return r;
}

// ============================================================
// init
// ============================================================

__global__ void init()
{
    if (blockIdx.x == 0 &&
        threadIdx.x == 0)
    {
        squash =
            new Squash();

        stretch =
            new Stretch();

        // FIX:
        // This initializes only thread 0's allocator slot.
        allocator =
            new Alloc();

        ilog =
            new Ilog();
    }
}

// ============================================================
// paq9 kernel
// ============================================================

__global__ void paq9_cuda(
    int *input_size,
    char **input,
    int *output_size,
    char **output,
    int num_of_chunks,
    int mode,
    int memory_level)
{
    int i =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (i >= num_of_chunks)
        return;

    // FIX:
    // Each thread gets its own independent allocator,
    // predictor and LZP objects.
    allocator =
        new Alloc();

    if (!allocator)
    {
        printf(
            "Thread %d: allocator allocation failed\n",
            i);
        output_size[i] = 0;
        return;
    }

    predictor =
        new Predictor();

    if (!predictor)
    {
        printf(
            "Thread %d: predictor allocation failed\n",
            i);
        output_size[i] = 0;
        return;
    }

    lzp =
        new LZP();

    if (!lzp)
    {
        printf(
            "Thread %d: LZP allocation failed\n",
            i);
        output_size[i] = 0;
        return;
    }

    output_size[i] = 0;

    // ========================================================
    // Compress
    // ========================================================

    if (mode == COMPRESS)
    {
        int itr = 0;

        Encoder encoder(
            mode,
            output[i],
            input_size[i],
            itr);

        int ch;

        itr = 0;

        while (itr < input_size[i])
        {
            ch =
                input[i][itr];

            ++itr;

            int cp =
                lzp->predict_char();

            if (ch == cp)
            {
                encoder.code(1);
            }
            else
            {
                for (int j = 8;
                     j >= 0;
                     --j)
                {
                    encoder.code(
                        ch >> j & 1);
                }
            }

            encoder.count();

            lzp->update(ch);
        }

        encoder.flush();

        output_size[i] =
            encoder.iterator_size;
    }
    else
    {
        // ====================================================
        // Decompress
        // ====================================================

        int itr = 0;
        int itr2 = 0;

        int usize =
            get4(
                itr,
                input[i]);

        get4(
            itr,
            input[i]);

        Encoder encoder(
            mode,
            input[i],
            input_size[i],
            itr);

        while (usize--)
        {
            int cp =
                lzp->predict_char();

            if (encoder.code() == 0)
            {
                cp = 1;

                while (cp < 256)
                {
                    cp +=
                        cp +
                        encoder.code();
                }

                cp &= 255;
            }

            output[i][itr2++] =
                cp;

            lzp->update(cp);
        }

        output_size[i] =
            itr2;
    }
}

// ============================================================
// Host compression
// ============================================================

void compress(
    char *destination_file,
    char *source_file)
{
    constexpr size_t MB = 1024 * 1024;

    size_t chunk_size =
        memory_chunk_size * MB;

    std::ifstream source(
        source_file,
        std::ios::binary);

    if (!source)
    {
        std::cerr
            << "Cannot open "
            << source_file
            << std::endl;

        exit(1);
    }

    source.seekg(
        0,
        std::ios::end);

    size_t total_size =
        source.tellg();

    size_t num_of_chunks =
        (total_size +
         chunk_size -
         1) /
        chunk_size;

    char **src_file =
        new char *[num_of_chunks];

    source.clear();

    source.seekg(
        0,
        std::ios::beg);

    std::vector<int> input_size(
        num_of_chunks);

    for (size_t i = 0;
         i < num_of_chunks;
         i++)
    {
        size_t current_size =
            std::min(
                chunk_size,
                total_size -
                    i * chunk_size);

        src_file[i] =
            new char[current_size];

        source.read(
            src_file[i],
            current_size);

        input_size[i] =
            (int)current_size;
    }

    source.close();

    // ========================================================
    // Device pointer arrays
    // ========================================================

    char **d_input;
    char **d_output;

    cudaMalloc(
        &d_input,
        num_of_chunks *
            sizeof(char *));

    cudaMalloc(
        &d_output,
        num_of_chunks *
            sizeof(char *));

    // ========================================================
    // Size arrays
    // ========================================================

    int *d_input_size;
    int *d_output_size;

    cudaMalloc(
        &d_input_size,
        num_of_chunks *
            sizeof(int));

    cudaMalloc(
        &d_output_size,
        num_of_chunks *
            sizeof(int));

    cudaMemcpy(
        d_input_size,
        input_size.data(),
        num_of_chunks *
            sizeof(int),
        cudaMemcpyHostToDevice);

    char **temp_d_input =
        new char *[num_of_chunks];

    char **temp_d_output =
        new char *[num_of_chunks];

    // ========================================================
    // FIX:
    // Output capacity must be larger than input.
    //
    // Worst case of your encoder is approximately 9 bits
    // for every input byte, plus arithmetic coder overhead.
    // 25% extra is sufficient for that case.
    // ========================================================

    std::vector<size_t> output_capacity(
        num_of_chunks);

    for (size_t i = 0;
         i < num_of_chunks;
         i++)
    {
        size_t in_size =
            (size_t)input_size[i];

        output_capacity[i] =
            in_size +
            in_size / 4 +
            4096;

        cudaError_t err =
            cudaMalloc(
                &temp_d_input[i],
                in_size);

        if (err != cudaSuccess)
        {
            std::cerr
                << "cudaMalloc input failed for chunk "
                << i
                << ": "
                << cudaGetErrorString(err)
                << '\n';

            exit(1);
        }

        err =
            cudaMalloc(
                &temp_d_output[i],
                output_capacity[i]);

        if (err != cudaSuccess)
        {
            std::cerr
                << "cudaMalloc output failed for chunk "
                << i
                << ": "
                << cudaGetErrorString(err)
                << '\n';

            exit(1);
        }

        cudaMemcpy(
            temp_d_input[i],
            src_file[i],
            in_size,
            cudaMemcpyHostToDevice);
    }

    cudaMemcpy(
        d_input,
        temp_d_input,
        num_of_chunks *
            sizeof(char *),
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_output,
        temp_d_output,
        num_of_chunks *
            sizeof(char *),
        cudaMemcpyHostToDevice);

    int threads = 256;

    int blocks =
        (num_of_chunks +
         threads - 1) /
        threads;

    // ========================================================
    // FIX:
    // Your original heap was 2 GB.
    // A 256-thread kernel creates 256 independent PAQ9 models.
    //
    // Each model requires roughly 12-15 MB of device heap.
    // ========================================================

    size_t heapSize =
        6ULL *
        1024ULL *
        1024ULL *
        1024ULL;

    cudaError_t err =
        cudaDeviceSetLimit(
            cudaLimitMallocHeapSize,
            heapSize);

    if (err != cudaSuccess)
    {
        std::cerr
            << "cudaDeviceSetLimit failed: "
            << cudaGetErrorString(err)
            << '\n';

        exit(1);
    }

    size_t actualHeap = 0;

    err =
        cudaDeviceGetLimit(
            &actualHeap,
            cudaLimitMallocHeapSize);

    if (err == cudaSuccess)
    {
        std::cout
            << "Device malloc heap: "
            << actualHeap /
                   (1024ULL * 1024ULL)
            << " MB\n";
    }

    // ========================================================
    // Initialize global shared tables
    // ========================================================

    init<<<1, 1>>>();

    err =
        cudaGetLastError();

    if (err != cudaSuccess)
    {
        std::cerr
            << "init launch error: "
            << cudaGetErrorString(err)
            << '\n';

        exit(1);
    }

    err =
        cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        std::cerr
            << "init kernel error: "
            << cudaGetErrorString(err)
            << '\n';

        exit(1);
    }

    // ========================================================
    // PAQ9 CUDA call
    // ========================================================

    int mode = COMPRESS;

    // Keep your original one-block launch.
    // This means the current code processes at most 256 chunks.
    paq9_cuda<<<1, threads>>>(
        d_input_size,
        d_input,
        d_output_size,
        d_output,
        (int)num_of_chunks,
        mode,
        memory_level);

    err =
        cudaGetLastError();

    if (err != cudaSuccess)
    {
        std::cerr
            << "paq9 launch error: "
            << cudaGetErrorString(err)
            << '\n';

        exit(1);
    }

    err =
        cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        std::cerr
            << "paq9 kernel error: "
            << cudaGetErrorString(err)
            << '\n';

        exit(1);
    }

    // ========================================================
    // Copy output sizes
    // ========================================================

    int *output_size =
        (int *)malloc(
            num_of_chunks *
            sizeof(int));

    cudaMemcpy(
        output_size,
        d_output_size,
        num_of_chunks *
            sizeof(int),
        cudaMemcpyDeviceToHost);

    // ========================================================
    // Copy output chunks
    // ========================================================

    char **output =
        new char *[num_of_chunks];

    for (size_t i = 0;
         i < num_of_chunks;
         i++)
    {
        if (output_size[i] < 0)
            output_size[i] = 0;

        output[i] =
            new char[output_size[i]];

        cudaMemcpy(
            output[i],
            temp_d_output[i],
            output_size[i] *
                sizeof(char),
            cudaMemcpyDeviceToHost);
    }

    // ========================================================
    // Free device chunk memory
    // ========================================================

    for (size_t i = 0;
         i < num_of_chunks;
         i++)
    {
        cudaFree(
            temp_d_input[i]);

        cudaFree(
            temp_d_output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);

    cudaFree(d_input_size);
    cudaFree(d_output_size);

    delete[] temp_d_input;
    delete[] temp_d_output;

    // ========================================================
    // Write output
    // ========================================================

    std::ofstream dest(
        destination_file,
        std::ios::binary);

    if (!dest)
    {
        std::cerr
            << "Cannot open destination file\n";

        exit(1);
    }

    dest.write("pQ9", 3);
    dest.put(1);

    for (size_t i = 0;
         i < num_of_chunks;
         i++)
    {
        dest.write(
            output[i],
            output_size[i]);

        std::cout
            << input_size[i]
            << " "
            << output_size[i]
            << std::endl;

        delete[] output[i];
        delete[] src_file[i];
    }

    dest.close();

    delete[] output;
    delete[] src_file;
    free(output_size);
}

// ============================================================
// main
// ============================================================

int main(
    int argc,
    char **args)
{
    auto start =
        std::chrono::steady_clock::now();

    std::cout
        << "CUDA version of PAQ9 started successfully.\n\n";

    if (argc < 3)
    {
        std::cout
            << "Run again and provide proper arguments.\n";

        exit(1);
    }

    int mode;

    char *destination_file_name;
    char *source_file_name;

    if (args[1][0] == '-')
    {
        if (args[1][1] == 'c')
            mode = COMPRESS;
        else if (args[1][1] == 'd')
            mode = DECOMPRESS;
        else
        {
            std::cout
                << "Run again and provide arguments in correct way.\n";

            exit(1);
        }
    }
    else
    {
        std::cout
            << "Run again and provide arguments in correct way.\n";

        exit(1);
    }

    int ind = 2;

    if (mode == COMPRESS)
    {
        if (args[ind][0] == '-')
        {
            std::string temp;

            int len =
                strlen(args[ind]);

            for (int i = 1;
                 i < len;
                 i++)
            {
                if (isdigit(
                        args[ind][i]))
                {
                    temp +=
                        args[ind][i];
                }
                else
                {
                    std::cout
                        << "Run again and provide arguments in correct way.\n";

                    exit(1);
                }
            }

            memory_chunk_size =
                stoi(temp);

            ind++;
        }

        if (ind < argc)
        {
            destination_file_name =
                args[ind];

            ind++;
        }
        else
        {
            std::cout
                << "Run again and provide arguments in correct way.\n";

            exit(1);
        }

        if (ind < argc &&
            args[ind][0] == '-')
        {
            std::string temp;

            int len =
                strlen(args[ind]);

            for (int i = 1;
                 i < len;
                 i++)
            {
                if (isdigit(
                        args[ind][i]))
                {
                    temp +=
                        args[ind][i];
                }
                else
                {
                    std::cout
                        << "Run again and provide arguments in correct way.\n";

                    exit(1);
                }
            }

            memory_level =
                stoi(temp);

            ind++;
        }
        else if (ind >= argc)
        {
            std::cout
                << "Run again and provide arguments in correct way.\n";

            exit(1);
        }

        if (ind < argc)
        {
            source_file_name =
                args[ind];

            ind++;
        }
        else
        {
            std::cout
                << "Run again and provide arguments in correct way.\n";

            exit(1);
        }

        compress(
            destination_file_name,
            source_file_name);
    }
    else
    {
        if (ind < argc)
        {
            source_file_name =
                args[ind];

            ind++;
        }
        else
        {
            std::cout
                << "Run again and provide arguments in correct way.\n";

            exit(1);
        }

        if (ind < argc)
        {
            destination_file_name =
                args[ind];

            ind++;
        }
        else
        {
            std::cout
                << "Run again and provide arguments in correct way.\n";

            exit(1);
        }
    }

    cudaDeviceSynchronize();

    auto end =
        std::chrono::steady_clock::now();

    double seconds =
        std::chrono::duration<double>(
            end - start)
            .count();

    std::cout
        << "Total wall time: "
        << seconds
        << " seconds\n";

    return 0;
}