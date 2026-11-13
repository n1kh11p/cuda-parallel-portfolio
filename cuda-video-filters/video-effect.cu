#include <vector>   // vector

/*  use this to set the block size of the kernel launches.
    CUDA kernels will be launched with block size blockDimSize by blockDimSize. */
constexpr int blockDimSize = 8;

/*  your job is to write convolveGPU:
    convolveGPU will be called with blockSize blockDimSize x blockDimSize 
    and gridsize ⌈height/blockDimSize⌉x⌈width/blockDimSize⌉.
    Each thread may have to compute more than one pixel. You will need to stride the computation.
    Look at convolveCPU below for more info.
*/
__global__ void convolveGPU(float const* in, float *out, int width, int height, float const* kernel, int kernelWidth, int kernelHeight) {
    // Calculate half kernel dimensions for border handling
    const int halfKernelHeight = kernelHeight / 2;
    const int halfKernelWidth = kernelWidth / 2;
    const int redChannel = 2;
    const int greenChannel = 1;
    const int blueChannel = 0;
    
    int cCol = blockIdx.x * blockDim.x + threadIdx.x;
    
    int cRow = blockIdx.y * blockDim.y + threadIdx.y;
    int xStr = gridDim.x * blockDim.x;
    int yStr = gridDim.y * blockDim.y;
    
    for (int pixR = cRow; pixR < height; pixR += yStr) {
        for (int pixC = cCol; pixC < width; pixC += xStr) {
            
            if (pixR < halfKernelHeight || pixR >= height - halfKernelHeight) {
                continue;
            }
            if (pixC < halfKernelWidth || pixC >= width - halfKernelWidth) {
                continue;
            }
            float redDot = 0.0f ,greenDot = 0.0f ,blueDot = 0.0f;
            
            for (int k=-halfKernelHeight; k<= halfKernelHeight; k++) {
                for (int l=-halfKernelWidth; l<= halfKernelWidth;l++) {
                    
                    int kRow =k+ halfKernelHeight;
                    int kCol=l+ halfKernelWidth;

                    int kernelInd =kRow *kernelWidth+ kCol;
                    float kernelValue=kernel[kernelInd];

                    int inpR=pixR+ k;

                    int inpC=pixC +l;
                    int inpInd= (inpR*width*3)+ (inpC*3);
                    
                    int greenInd=inpInd+ greenChannel;
                    int blueInd=inpInd+blueChannel;

                    int redInd= inpInd+redChannel;
                    greenDot +=in[greenInd]*kernelValue;
                    blueDot+=in[blueInd] *kernelValue;

                    redDot+= in[redInd]* kernelValue;
                }
            }
            
            int outputInd=pixR*width*3 + pixC*3;

            int greenInd =outputInd+greenChannel;

            int blueInd= outputInd+blueChannel;
            int redInd =outputInd+redChannel;
            out[redInd]=redDot;
            out[greenInd] =greenDot;

            out[blueInd]= blueDot;
        }
    }
}

/* A CPU example of the convolve kernel */
void convolveCPU(float const* in, float *out, int width, int height, float const* kernel, int kernelWidth, int kernelHeight) {
    const int halfKernelHeight = kernelHeight/2;
    const int halfKernelWidth = kernelWidth/2;
    const int redChannel = 2;
    const int greenChannel = 1;
    const int blueChannel = 0;

    /* point-wise loop over the image pixels */
    for (int i = halfKernelHeight; i < height-halfKernelHeight; i += 1) {
        for (int j = halfKernelWidth; j < width-halfKernelWidth; j += 1) {

            /* compute dot product of kernel and sub-image */
            float redDot = 0.0f, greenDot = 0.0f, blueDot = 0.0f;
            for (int k = -halfKernelHeight; k <= halfKernelHeight; k += 1) {
                for (int l = -halfKernelWidth; l <= halfKernelWidth; l += 1) {

                    /* add in[i+k][j+l]*kernel[k][l] to dot product for red, green, and blue */
                    redDot += in[(i+k)*width*3 + (j+l)*3 + redChannel] * kernel[(k+halfKernelHeight)*kernelWidth + (l+halfKernelWidth)];
                    greenDot += in[(i+k)*width*3 + (j+l)*3 + greenChannel] * kernel[(k+halfKernelHeight)*kernelWidth + (l+halfKernelWidth)];
                    blueDot += in[(i+k)*width*3 + (j+l)*3 + blueChannel] * kernel[(k+halfKernelHeight)*kernelWidth + (l+halfKernelWidth)];
                
                }
            }

            /* set out[i][j] to dot product */
            out[i*width*3 + j*3 + redChannel] = redDot;
            out[i*width*3 + j*3 + greenChannel] = greenDot;
            out[i*width*3 + j*3 + blueChannel] = blueDot;

        }
    }
}

/* call the convolveGPU function on each frame */
float convolveFrames(std::vector<float *> const& framesIn, std::vector<float *> &framesOut, int width, int height, float const* kernel, int kernelWidth, int kernelHeight,
    cudaStream_t *streams, int numStreams, int gridSizeX, int gridSizeY) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blockSize (blockDimSize, blockDimSize);
    dim3 gridSize (gridSizeX, gridSizeY);

    cudaEventRecord(start, 0);
    for (int i = 0; i < framesIn.size(); i += 1) {
        convolveGPU<<<gridSize, blockSize, 0, streams[i % numStreams]>>>(framesIn.at(i), framesOut.at(i), width, height, kernel, kernelWidth, kernelHeight);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (elapsed / 1000.0f);
}
