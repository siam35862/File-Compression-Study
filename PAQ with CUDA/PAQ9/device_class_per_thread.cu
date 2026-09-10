#include <iostream>
#include <cuda_runtime.h>

using namespace std;

class Chunk
{
public:
    char *data;
    int size;

    __device__ Chunk() : data(nullptr), size(0) {}

    __device__ void init(char *buf, int n)
    {
        data = buf;
        size = n;
    }

    __device__ void fillWith(char value)
    {
        for (int i = 0; i < size; i++)
            data[i] = value;
    }
};

// Thread i constructs chunks[i] exactly once here...
__global__ void initChunks(char **buffers, int *sizes, Chunk *chunks, int numChunks)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numChunks)
        return;

    // placement-new into THIS thread's own slot only
    Chunk *myChunk = new (&chunks[i]) Chunk();
    myChunk->init(buffers[i], sizes[i]);
}

// ...and a LATER, separate kernel launch can have thread i go back
// to that same chunks[i] instance - never chunks[j] for j != i.
__global__ void fillChunks(Chunk *chunks, int numChunks)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numChunks)
        return;

    chunks[i].fillWith('A' + (i % 26)); // still operating only on this thread's own instance
}

int main()
{
    const int numChunks = 4;
    const int chunkSize = 8;

    char *d_buffers_individual[numChunks];
    for (int i = 0; i < numChunks; i++)
        cudaMalloc((void **)&d_buffers_individual[i], chunkSize * sizeof(char));

    char **d_buffers;
    cudaMalloc((void **)&d_buffers, numChunks * sizeof(char *));
    cudaMemcpy(d_buffers, d_buffers_individual, numChunks * sizeof(char *), cudaMemcpyHostToDevice);

    int h_sizes[numChunks];
    for (int i = 0; i < numChunks; i++)
        h_sizes[i] = chunkSize;
    int *d_sizes;
    cudaMalloc((void **)&d_sizes, numChunks * sizeof(int));
    cudaMemcpy(d_sizes, h_sizes, numChunks * sizeof(int), cudaMemcpyHostToDevice);

    Chunk *d_chunks;
    cudaMalloc((void **)&d_chunks, numChunks * sizeof(Chunk));

    int threads = 256;
    int blocks = (numChunks + threads - 1) / threads;

    // Two SEPARATE kernel launches. Thread i still only ever
    // touches chunks[i] in both of them.
    initChunks<<<blocks, threads>>>(d_buffers, d_sizes, d_chunks, numChunks);
    cudaDeviceSynchronize();

    fillChunks<<<blocks, threads>>>(d_chunks, numChunks);
    cudaDeviceSynchronize();

    for (int i = 0; i < numChunks; i++)
    {
        char *h_buf = new char[chunkSize];
        cudaMemcpy(h_buf, d_buffers_individual[i], chunkSize * sizeof(char), cudaMemcpyDeviceToHost);
        cout << "Chunk " << i << ": ";
        for (int j = 0; j < chunkSize; j++)
            cout << h_buf[j];
        cout << "\n";
        delete[] h_buf;
        cudaFree(d_buffers_individual[i]);
    }

    cudaFree(d_buffers);
    cudaFree(d_sizes);
    cudaFree(d_chunks);

    return 0;
}
