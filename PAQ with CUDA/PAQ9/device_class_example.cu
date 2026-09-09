#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// A class that owns a device-side buffer and runs entirely on the GPU.
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

// Kernel that constructs the object on the device (placement new),
// then uses it. This is what you launch to "build" the object.
__global__ void makeAndUseChunk(char *buffer, int n, Chunk *outObj)
{
    // placement-new: construct a Chunk in memory we already own on device
    Chunk *c = new (outObj) Chunk();
    c->init(buffer, n);
    c->fillWith('A');
}

int main()
{
    const int n = 16;

    char *d_buffer;
    Chunk *d_chunk;

    cudaMalloc((void **)&d_buffer, n * sizeof(char));
    cudaMalloc((void **)&d_chunk, sizeof(Chunk));

    makeAndUseChunk<<<1, 1>>>(d_buffer, n, d_chunk);
    cudaDeviceSynchronize();

    char *h_buffer = new char[n];
    cudaMemcpy(h_buffer, d_buffer, n * sizeof(char), cudaMemcpyDeviceToHost);

    cout << "Buffer after Chunk::fillWith: ";
    for (int i = 0; i < n; i++)
        cout << h_buffer[i];
    cout << "\n";

    delete[] h_buffer;
    cudaFree(d_buffer);
    cudaFree(d_chunk);

    return 0;
}
