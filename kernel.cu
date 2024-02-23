
#include "common.h"

#include "timer.h"

__global__ void histogram_private_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {

    __shared__ unsigned int private_bins[256];

    // Initialize private histogram bins to zero
    for (int i = 0; i < 256; ++i) {
        private_bins[i] = 0;
    }

    __syncthreads(); 

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * height) {
        unsigned char b = image[i];
        atomicAdd(&private_bins[b], 1);
    }

    __syncthreads();

    // Commit non-zero bin counts to the global copy of the histogram
    if (threadIdx.x < 256) {
        atomicAdd(&bins[threadIdx.x], private_bins[threadIdx.x]);
    }




}

void histogram_gpu_private(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (width * height + numThreadsPerBlock - 1) / numThreadsPerBlock;
    histogram_private_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, bins_d, width, height);

}
__global__ void histogram_private_coarse_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    __shared__ unsigned int private_bins[256]; // Private histogram bins for each block

    // Initialize private histogram bins to zero
    for (int j = 0; j < 256; j++) {
        private_bins[j] = 0;
    }

    __syncthreads(); // Ensure all threads have initialized private histogram bins

    unsigned int coarsening_factor = 64; // Coarsening factor variable

    unsigned int tid = blockIdx.x * (blockDim.x * coarsening_factor) + threadIdx.x; // Adjusted tid calculation for coarsening factor
    for (int i = 0; i < coarsening_factor; ++i) { // Iterate over pixels based on the coarsening factor
        unsigned int index = tid + i * blockDim.x;
        if (index < width * height) {
            unsigned char b = image[index];
            atomicAdd(&private_bins[b], 1); // Update private histogram bins atomically
        }
    }

    __syncthreads(); // Ensure all threads have finished updating private histogram bins

    // Commit non-zero bin counts to the global copy of the histogram in parallel
    if (threadIdx.x < 256) {
        atomicAdd(&bins[threadIdx.x], private_bins[threadIdx.x]);
    }
}

void histogram_gpu_private_coarse(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {
    unsigned int coarsening_factor = 64; // Define the coarsening factor


    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (width * height + numThreadsPerBlock * coarsening_factor - 1) / (numThreadsPerBlock * coarsening_factor);
    histogram_private_coarse_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, bins_d, width, height);
}

