#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <cstring> // For strlen

#define COMPRESS 0   // Compression mode
#define DECOMPRESS 1 // Decompression mode
int mode;
int memory_level = 7;      // default memory level
int memory_chunk_size = 1; // default memory chunks

template <class T>
__device__ void alloc(T *&p, int allocate_size, int &total_allocated_size)
{
    p = new (std::nothrow) T[allocate_size]();
    if (!p)
    {
        printf("Error: Out of memory (failed to allocate %d elements)\n", allocate_size * sizeof(T));
        exit(1);
    }
    total_allocated_size += allocate_size * sizeof(T);
}

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

// intialize the method of Squash
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

__global__ void paq9_cuda(
    int *input_size,
    char **input,
    int *output_size,
    char **output,
    int num_of_chunks, int mode, int memory_level)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (mode == COMPRESS)
        ;
    if (i >= num_of_chunks)
        return;

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

    paq9_cuda<<<blocks, threads>>>(
        d_input_size,
        d_input,
        d_output_size,
        d_output,
        num_of_chunks, mode, memory_level);

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

        dest.write(output[i], output_size[i]);
        std::cout << input_size[i] << " " << output_size[i] << std::endl;
    }
    dest.close();
}

int main(int argc, char **args)
{
    clock_t start = clock();
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

    std::cout << "Successfuly finished with exit code 0\n";
}