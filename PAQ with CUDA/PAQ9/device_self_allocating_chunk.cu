#include <iostream>
#include <cuda_runtime.h>

using namespace std;

class Chunk
{
private:
    int siam;
public:
    char *data;
    int size;

    __device__ Chunk() : data(nullptr), size(0) {}

    __device__ void init(int n)
    {
        size = n;
        data = new char[n]; // device heap allocation, still on-device
    }

    __device__ void fillWith(char value)
    {
        for (int i = 0; i < size; i++)
            data[i] = value;
    }

    __device__ void copyOut(char *dst)
    {
        for (int i = 0; i < size; i++)
            dst[i] = data[i];
    }

    __device__ ~Chunk()
    {
        delete[] data; // device-side free
    }
};

// Host passes in only raw sizes and a plain output buffer -
// no Chunk*, no Chunk-sized cudaMalloc, no Chunk construction on host.
__global__ void processChunks(int *sizes, char *outFlat, int chunkSize, int numChunks)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numChunks)
        return;
    Chunk chn=Chunk();

    // allocated, constructed, used, and destroyed - all on-device,
    // all within this one thread's own scope
    Chunk myChunk;
    myChunk.init(sizes[i]);
    myChunk.fillWith('A' + (i % 26));
    myChunk.copyOut(outFlat + i * chunkSize);
    // myChunk destructor runs automatically at end of scope, freeing device heap memory
}

int main()
{
    const int numChunks = 4;
    const int chunkSize = 8;

    // Optional: grow the device heap if you plan to allocate a lot from kernels
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 32 * 1024 * 1024);

    int h_sizes[numChunks];
    for (int i = 0; i < numChunks; i++)
        h_sizes[i] = chunkSize;

    int *d_sizes;
    cudaMalloc((void **)&d_sizes, numChunks * sizeof(int)); // just plain ints, not Chunk objects
    cudaMemcpy(d_sizes, h_sizes, numChunks * sizeof(int), cudaMemcpyHostToDevice);

    char *d_outFlat;
    cudaMalloc((void **)&d_outFlat, numChunks * chunkSize * sizeof(char)); // plain bytes, not Chunk objects

    int threads = 256;
    int blocks = (numChunks + threads - 1) / threads;
    processChunks<<<blocks, threads>>>(d_sizes, d_outFlat, chunkSize, numChunks);
    cudaDeviceSynchronize();

    char *h_out = new char[numChunks * chunkSize];
    cudaMemcpy(h_out, d_outFlat, numChunks * chunkSize * sizeof(char), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numChunks; i++)
    {
        cout << "Chunk " << i << ": ";
        for (int j = 0; j < chunkSize; j++)
            cout << h_out[i * chunkSize + j];
        cout << "\n";
    }

    delete[] h_out;
    cudaFree(d_sizes);
    cudaFree(d_outFlat);

    return 0;
}
