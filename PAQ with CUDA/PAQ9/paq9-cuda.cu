#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <chrono>
#include <cstring> // For strlen
#include <assert.h>

#define COMPRESS 0   // Compression mode
#define DECOMPRESS 1 // Decompression mode
int mode;
int memory_level = 7;      // default memory level
int memory_chunk_size = 1; // default memory chunks

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
        p = new (std::nothrow) T[allocate_size]();
        if (!p)
        {
            printf("Error: Out of memory (failed to allocate %d elements)\n", allocate_size * sizeof(T));
            exit(1);
        }
        total_allocated_size += allocate_size * sizeof(T);
    }
};

__device__ Alloc allocator;

// 8, 16, 32 bit unsigned types (adjust as appropriate)
typedef unsigned char U8;
typedef unsigned short U16;
typedef unsigned int U32;

///////////////////////////// Squash //////////////////////////////

// return p = 1/(1 + exp(-d)), d scaled by 8 bits, p scaled by 12 bits
class Squash
{
    short tab[4096];

public:
    __device__ Squash();
    __device__ int operator()(int d);
};

// intialize the sonstructor and method of Squash
__device__ Squash::Squash()
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

// global instance of squash
__device__ Squash squash = Squash();

//////////////////////////// Stretch ///////////////////////////////

// Inverse of squash. stretch(d) returns ln(p/(1-p)), d scaled by 8 bits,
// p by 12 bits.  d has range -2047 to 2047 representing -8 to 8.
// p has range 0 to 4095 representing 0 to 1.

class Stretch
{
    short t[4096];

public:
    __device__ Stretch();
    __device__ int operator()(int p) const;
};
// intialize the sonstructor and method of Stretch

__device__ Stretch::Stretch()
{
    int pi = 0;
    for (int x = -2047; x <= 2047; ++x)
    { // invert squash()
        int i = squash(x);
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

// global instance of Stretch
__device__ Stretch stretch = Stretch();

///////////////////////////// ilog //////////////////////////////

// ilog(x) = round(log2(x) * 16), 0 <= x < 64K

class Ilog
{
    U8 *t;

public:
    __device__ Ilog();
    __device__ int operator()(U16 x) const;
    __device__ int operator()(U32 x) const;
};

// intialize the sonstructor and method of Ilog

__device__ Ilog::Ilog()
{
    allocator.alloc(t, 65536);
    U32 x = 14155776;
    for (int i = 2; i < 65536; ++i)
    {
        x += 774541002 / (i * 2 - 1); // numerator is 2^29/ln 2
        t[i] = x >> 24;
    }
}
__device__ int Ilog::operator()(U16 x) const
{
    return t[x];
}
__device__ int Ilog::operator()(U32 x) const
{
    if (x >= 0x1000000)
        return 256 + ilog(x >> 16);
    else if (x >= 0x10000)
        return 128 + ilog(x >> 8);
    else
        return ilog(x);
}
// global instance of Ilog

__device__ Ilog ilog = Ilog();

///////////////////////// state table ////////////////////////

// State table:
//   nex(state, 0) = next state if bit y is 0, 0 <= state < 256
//   nex(state, 1) = next state if bit y is 1
//
// States represent a bit history within some context.
// State 0 is the starting state (no bits seen).
// States 1-30 represent all possible sequences of 1-4 bits.
// States 31-252 represent a pair of counts, (n0,n1), the number
//   of 0 and 1 bits respectively.  If n0+n1 < 16 then there are
//   two states for each pair, depending on if a 0 or 1 was the last
//   bit seen.
// If n0 and n1 are too large, then there is no state to represent this
// pair, so another state with about the same ratio of n0/n1 is substituted.
// Also, when a bit is observed and the count of the opposite bit is large,
// then part of this count is discarded to favor newer data over old.

__device__ static const U8 State_table[256][2] = {
    {1, 2}, {3, 5}, {4, 6}, {7, 10}, {8, 12}, {9, 13}, {11, 14}, // 0
    {15, 19},
    {16, 23},
    {17, 24},
    {18, 25},
    {20, 27},
    {21, 28},
    {22, 29}, // 7
    {26, 30},
    {31, 33},
    {32, 35},
    {32, 35},
    {32, 35},
    {32, 35},
    {34, 37}, // 14
    {34, 37},
    {34, 37},
    {34, 37},
    {34, 37},
    {34, 37},
    {36, 39},
    {36, 39}, // 21
    {36, 39},
    {36, 39},
    {38, 40},
    {41, 43},
    {42, 45},
    {42, 45},
    {44, 47}, // 28
    {44, 47},
    {46, 49},
    {46, 49},
    {48, 51},
    {48, 51},
    {50, 52},
    {53, 43}, // 35
    {54, 57},
    {54, 57},
    {56, 59},
    {56, 59},
    {58, 61},
    {58, 61},
    {60, 63}, // 42
    {60, 63},
    {62, 65},
    {62, 65},
    {50, 66},
    {67, 55},
    {68, 57},
    {68, 57}, // 49
    {70, 73},
    {70, 73},
    {72, 75},
    {72, 75},
    {74, 77},
    {74, 77},
    {76, 79}, // 56
    {76, 79},
    {62, 81},
    {62, 81},
    {64, 82},
    {83, 69},
    {84, 71},
    {84, 71}, // 63
    {86, 73},
    {86, 73},
    {44, 59},
    {44, 59},
    {58, 61},
    {58, 61},
    {60, 49}, // 70
    {60, 49},
    {76, 89},
    {76, 89},
    {78, 91},
    {78, 91},
    {80, 92},
    {93, 69}, // 77
    {94, 87},
    {94, 87},
    {96, 45},
    {96, 45},
    {48, 99},
    {48, 99},
    {88, 101}, // 84
    {88, 101},
    {80, 102},
    {103, 69},
    {104, 87},
    {104, 87},
    {106, 57},
    {106, 57}, // 91
    {62, 109},
    {62, 109},
    {88, 111},
    {88, 111},
    {80, 112},
    {113, 85},
    {114, 87}, // 98
    {114, 87},
    {116, 57},
    {116, 57},
    {62, 119},
    {62, 119},
    {88, 121},
    {88, 121}, // 105
    {90, 122},
    {123, 85},
    {124, 97},
    {124, 97},
    {126, 57},
    {126, 57},
    {62, 129}, // 112
    {62, 129},
    {98, 131},
    {98, 131},
    {90, 132},
    {133, 85},
    {134, 97},
    {134, 97}, // 119
    {136, 57},
    {136, 57},
    {62, 139},
    {62, 139},
    {98, 141},
    {98, 141},
    {90, 142}, // 126
    {143, 95},
    {144, 97},
    {144, 97},
    {68, 57},
    {68, 57},
    {62, 81},
    {62, 81}, // 133
    {98, 147},
    {98, 147},
    {100, 148},
    {149, 95},
    {150, 107},
    {150, 107},
    {108, 151}, // 140
    {108, 151},
    {100, 152},
    {153, 95},
    {154, 107},
    {108, 155},
    {100, 156},
    {157, 95}, // 147
    {158, 107},
    {108, 159},
    {100, 160},
    {161, 105},
    {162, 107},
    {108, 163},
    {110, 164}, // 154
    {165, 105},
    {166, 117},
    {118, 167},
    {110, 168},
    {169, 105},
    {170, 117},
    {118, 171}, // 161
    {110, 172},
    {173, 105},
    {174, 117},
    {118, 175},
    {110, 176},
    {177, 105},
    {178, 117}, // 168
    {118, 179},
    {110, 180},
    {181, 115},
    {182, 117},
    {118, 183},
    {120, 184},
    {185, 115}, // 175
    {186, 127},
    {128, 187},
    {120, 188},
    {189, 115},
    {190, 127},
    {128, 191},
    {120, 192}, // 182
    {193, 115},
    {194, 127},
    {128, 195},
    {120, 196},
    {197, 115},
    {198, 127},
    {128, 199}, // 189
    {120, 200},
    {201, 115},
    {202, 127},
    {128, 203},
    {120, 204},
    {205, 115},
    {206, 127}, // 196
    {128, 207},
    {120, 208},
    {209, 125},
    {210, 127},
    {128, 211},
    {130, 212},
    {213, 125}, // 203
    {214, 137},
    {138, 215},
    {130, 216},
    {217, 125},
    {218, 137},
    {138, 219},
    {130, 220}, // 210
    {221, 125},
    {222, 137},
    {138, 223},
    {130, 224},
    {225, 125},
    {226, 137},
    {138, 227}, // 217
    {130, 228},
    {229, 125},
    {230, 137},
    {138, 231},
    {130, 232},
    {233, 125},
    {234, 137}, // 224
    {138, 235},
    {130, 236},
    {237, 125},
    {238, 137},
    {138, 239},
    {130, 240},
    {241, 125}, // 231
    {242, 137},
    {138, 243},
    {130, 244},
    {245, 135},
    {246, 137},
    {138, 247},
    {140, 248}, // 238
    {249, 135},
    {250, 69},
    {80, 251},
    {140, 252},
    {249, 135},
    {250, 69},
    {80, 251}, // 245
    {140, 252},
    {0, 0},
    {0, 0},
    {0, 0}}; // 252

#define nex(state, sel) State_table[state][sel]

//////////////////////////// StateMap //////////////////////////

// A StateMap maps a context to a probability.  Methods:
//
// Statemap sm(n) creates a StateMap with n contexts using 4*n bytes memory.
// sm.p(cx, limit) converts state cx (0..n-1) to a probability (0..4095)
//     that the next updated bit y=1.
//     limit (1..1023, default 255) is the maximum count for computing a
//     prediction.  Larger values are better for stationary sources.
// sm.update(y) updates the model with actual bit y (0..1).

class StateMap
{
protected:
    const int N;           // Number of contexts
    int cntxt;             // Context of last prediction
    U32 *prediction_table; // cntxt -> prediction in high 22 bits, count in low 10 bits
    static int dt[1024];   // i-> 16K/(i+3)

public:
    __device__ StateMap(int n = 256);

    // update bit y (0..1)
    __device__ void update(int y, int limit = 255);

    // predict next bit in context cntx

    __device__ int predict_next_bit(int cntx);
};

// Initialization
__device__ int StateMap::dt[1024] = {0};

__device__ StateMap::StateMap(int n) : N(n), cntxt(0)
{
    allocator.alloc(prediction_table, N);
    for (int i = 0; i < N; i++)
        prediction_table[i] = 2147483648U; // 1<<31
    if (dt[0] == 0)
        for (int i = 0; i < 1024; i++)
            dt[i] = 16384 / (i + i + 3);
}

__device__ void StateMap::update(int y, int limit = 255)
{
    assert(cntxt >= 0 && cntxt < N);
    int n = prediction_table[cntxt] & 1023, p = prediction_table[cntxt] >> 10; // count, prediction

    if (n < limit)
        prediction_table[cntxt]++;
    else
        prediction_table[cntxt] = prediction_table[cntxt] & 0xfffffc00 | limit;

    prediction_table[cntxt] += (((y << 22) - p) >> 3) * dt[n] & 0xfffffc00;
}

__device__ int StateMap::predict_next_bit(int cntx)
{
    assert(cntx >= 0 && cntx < N);
    return prediction_table[cntxt = cntx] >> 20;
}


__global__ void paq9_cuda(
    int *input_size,
    char **input,
    int *output_size,
    char **output,
    int num_of_chunks, int mode, int memory_level)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_of_chunks)
        return;
    allocator = Alloc();
    if (mode == COMPRESS)
        ;

    for (int j = 0; j < input_size[i]; j++)
    {
        output[i][j] = input[i][j];
    }

    output_size[i] = input_size[i];
}

void compress(char *destination_file, char *source_file)
{

    constexpr size_t MB = 1024 * 1024;
    size_t chunk_size = memory_chunk_size * MB;

    std::ifstream source(source_file, std::ios::binary);
    if (!source)
    {
        std::cerr << "Cannot open " << source_file << std::endl;
        exit(1);
    }
    source.seekg(0, std::ios::end);
    size_t total_size = source.tellg();

    size_t num_of_chunks =
        (total_size + chunk_size - 1) / chunk_size;

    char **src_file = new char *[num_of_chunks];
    source.clear();
    source.seekg(0, std::ios::beg);

    std::vector<int> input_size(num_of_chunks);

    for (size_t i = 0; i < num_of_chunks; i++)
    {
        size_t current_size =
            min(chunk_size, total_size - i * chunk_size);

        src_file[i] = new char[current_size];

        source.read(src_file[i], current_size);
        input_size[i] = current_size;
    }
    source.close();

    // preparing for calling device function

    // --------------------------------------------------
    // Device pointer arrays
    // --------------------------------------------------

    char **d_input;
    char **d_output;

    cudaMalloc(&d_input, num_of_chunks * sizeof(char *));
    cudaMalloc(&d_output, num_of_chunks * sizeof(char *));

    // --------------------------------------------------
    // Size arrays
    // --------------------------------------------------

    int *d_input_size;
    int *d_output_size;

    cudaMalloc(&d_input_size,
               num_of_chunks * sizeof(int));

    cudaMalloc(&d_output_size,
               num_of_chunks * sizeof(int));
    // --------------------------------------------------
    // Copy input sizes: HOST -> DEVICE
    // --------------------------------------------------

    cudaMemcpy(
        d_input_size,
        input_size.data(),
        num_of_chunks * sizeof(int),
        cudaMemcpyHostToDevice);

    // --------------------------------------------------
    // Temporary host arrays containing device pointers
    // --------------------------------------------------

    char **temp_d_input =
        new char *[num_of_chunks];

    char **temp_d_output =
        new char *[num_of_chunks];

    // --------------------------------------------------
    // Allocate each chunk on DEVICE
    // --------------------------------------------------

    for (int i = 0; i < num_of_chunks; i++)
    {
        // Input
        cudaMalloc(
            &temp_d_input[i],
            input_size[i] * sizeof(char));

        // Output
        //
        // Currently output size == input size
        // because your kernel only copies data.
        cudaMalloc(
            &temp_d_output[i],
            input_size[i] * sizeof(char));

        // --------------------------------------------------
        // Copy input chunk: HOST -> DEVICE
        // --------------------------------------------------

        cudaMemcpy(
            temp_d_input[i],
            src_file[i],
            input_size[i] * sizeof(char),
            cudaMemcpyHostToDevice);
    }

    // --------------------------------------------------
    // Copy DEVICE POINTER ARRAYS to DEVICE
    // --------------------------------------------------

    cudaMemcpy(
        d_input,
        temp_d_input,
        num_of_chunks * sizeof(char *),
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_output,
        temp_d_output,
        num_of_chunks * sizeof(char *),
        cudaMemcpyHostToDevice);

    // --------------------------------------------------
    // Launch kernel
    // --------------------------------------------------

    int threads = 256;

    int blocks =
        (num_of_chunks + threads - 1) / threads;

    // paq9_cuda<<<blocks, threads>>>(
    //     d_input_size,
    //     d_input,
    //     d_output_size,
    //     d_output,
    //     num_of_chunks, mode, memory_level);

    cudaDeviceSynchronize();

    // --------------------------------------------------
    // Copy output sizes: DEVICE -> HOST
    // --------------------------------------------------

    // FIX: Allocate memory for the host integer array before copying
    int *output_size = new int[num_of_chunks];
    cudaMemcpy(
        output_size,
        d_output_size,
        num_of_chunks * sizeof(int),
        cudaMemcpyDeviceToHost);

    // --------------------------------------------------
    // Copy output chunks: DEVICE -> HOST
    // --------------------------------------------------

    // FIX: Allocate memory for the array of host pointers before copying
    char **output = new char *[num_of_chunks];
    for (int i = 0; i < num_of_chunks; i++)
    {
        // FIX: Allocate memory for each specific chunk array before copying
        output[i] = new char[output_size[i]];
        cudaMemcpy(
            output[i],
            temp_d_output[i],
            output_size[i] * sizeof(char),
            cudaMemcpyDeviceToHost);
    }

    // --------------------------------------------------
    // Free DEVICE chunk memory
    // --------------------------------------------------

    for (int i = 0; i < num_of_chunks; i++)
    {
        cudaFree(temp_d_input[i]);
        cudaFree(temp_d_output[i]);
    }

    // --------------------------------------------------
    // Free DEVICE arrays
    // --------------------------------------------------

    cudaFree(d_input);
    cudaFree(d_output);

    cudaFree(d_input_size);
    cudaFree(d_output_size);

    delete[] temp_d_input;
    delete[] temp_d_output;

    std::ofstream dest(destination_file, std::ios::binary);

    for (size_t i = 0; i < num_of_chunks; i++)
    {
        size_t current_size =
            min(chunk_size, total_size - i * chunk_size);

        dest.write(src_file[i], input_size[i]);
        // std::cout << input_size[i] << " " << output_size[i] << std::endl;
    }
    dest.close();
}

int main(int argc, char **args)
{
    auto start = std::chrono::steady_clock::now();
    std::cout << "CUDA version of PAQ9 started successfully.\n\n";
    if (argc < 3)
    {
        std::cout << "Run again and provide proper arguments.\n";
        exit(1);
    }

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
            std::cout << "Run again and provide arguments in correct way.\n";
            exit(1);
        }
    }
    else
    {
        std::cout << "Run again and provide arguments in correct way.\n";
        exit(1);
    }
    int ind = 2;
    if (mode == COMPRESS)
    {
        if (args[ind][0] == '-')
        {
            std::string temp;

            int len = strlen(args[ind]);
            for (int i = 1; i < len; i++)
            {
                if (isdigit(args[ind][i]))
                    temp += args[ind][i];
                else
                {
                    std::cout << "Run again and provide arguments in correct way.\n";
                    exit(1);
                }
            }
            memory_chunk_size = stoi(temp);
            ind++;
        }

        if (ind < argc)
        {
            destination_file_name = args[ind];
            ind++;
        }
        else
        {
            std::cout << "Run again and provide arguments in correct way.\n";
            exit(1);
        }

        if (ind < argc && args[ind][0] == '-')
        {
            std::string temp;

            int len = strlen(args[ind]);
            for (int i = 1; i < len; i++)
            {
                if (isdigit(args[ind][i]))
                    temp += args[ind][i];
                else
                {
                    std::cout << "Run again and provide arguments in correct way.\n";
                    exit(1);
                }
            }
            memory_level = stoi(temp);
            ind++;
        }
        else if (ind >= argc)
        {
            std::cout << "Run again and provide arguments in correct way.\n";
            exit(1);
        }

        if (ind < argc)
        {
            source_file_name = args[ind];
            ind++;
        }
        else
        {
            std::cout << "Run again and provide arguments in correct way.\n";
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
            std::cout << "Run again and provide arguments in correct way.\n";
            exit(1);
        }
        if (ind < argc)
        {
            destination_file_name = args[ind];
            ind++;
        }
        else
        {
            std::cout << "Run again and provide arguments in correct way.\n";
            exit(1);
        }
    }

    cudaDeviceSynchronize(); // GPU কাজ শেষ হওয়া নিশ্চিত

    auto end = std::chrono::steady_clock::now();

    double seconds =
        std::chrono::duration<double>(end - start).count();

    std::cout << "Total wall time: "
              << seconds << " seconds\n";

    return 0;
}