// WE'RE FIXING THIS VERSION: 
// - get rid of expensive i/o at each iteration
// - avoid recreating bvh at each iteration
// - make a class with a C interface for python access
// readFile, writeFile are the problem
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <typeinfo>
#include <vector>

#include <stdint.h>
#include "bvh_structure.h"
#include "rsi_geometry.h"

using namespace std;
using namespace lib_bvh;
using namespace lib_rsi;

#ifdef __cplusplus
extern "C" {
#endif

// Pure C-style declarations here
void* setup_RSI(float* vertices, int* triangles, int num_vertices, int num_triangles, int num_rays);
void detect_RSI(void* rsi_obj, float* rayFrom, float* rayTo, int** out_intersectTriangle, float** out_baryT);
void destroy_RSI(void* rsi_obj);
static void HandleError(cudaError_t err, const char *file, int line);


#ifdef __cplusplus
}
#endif

//-------------------------------------------------
// This implementation corresponds to version v3
// with support for barycentric mode and the
// intercept_count experimental feature
//-------------------------------------------------

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line);
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void CheckSyncAsyncErrors(const char* file, int line)
{
    // Inspired from https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/
    cudaError_t errSync = cudaGetLastError(); // returns the value of the latest asynchronous error and also resets it to cudaSuccess.
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess)
    {
        printf("Sync kernel error\n");
        HandleError(errSync, file, line);
    }
    if (errAsync != cudaSuccess)
    {
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
        HandleError(errAsync, file, line);
    }
}

#define CUDA_SYNCHRO_CHECK() (CheckSyncAsyncErrors(__FILE__, __LINE__))

template <class T>
int readData(string fname, vector<T> &v, int dim=1, bool silent=false)
{
    ifstream infile(fname.c_str(), ios::binary | ios::ate);
    if (! infile) {
        cerr << "File " << fname << " not found" << endl;
        exit(1);
    }
    ifstream::pos_type nbytes = infile.tellg();
    infile.seekg(0, infile.beg);
    const int elements = nbytes / sizeof(T);
    v.resize(elements);
    infile.read(reinterpret_cast<char*>(v.data()), nbytes);
    if (! silent) {
        cout << fname << " contains " << nbytes << " bytes, "
             << v.size() << " <" << typeid(v.front()).name() << ">, "
             << v.size() / dim << " elements" << endl;
    }
    return elements / dim;
}

template <class T>
void writeData(string fname, vector<T> &v)
{
    ofstream outfile(fname.c_str(), ios::out | ios::binary);
    if (! outfile) {
        cerr << "Cannot create " << fname << " for writing" << endl;
        exit(1);
    }
    outfile.write(reinterpret_cast<char*>(v.data()), v.size() * sizeof(T));
    outfile.close();
}


// OsamaDabb functions etc.
inline void safeCudaFree(void* ptr)
{
    if (ptr != nullptr)
    {
        HANDLE_ERROR(cudaFree(ptr));
        ptr = nullptr;  // Optional: prevents double-free later
    }
}

// End OsamaDabb


class RSI {
    public:
        const bool checkEnabled;
        const float largePosVal;
        const bool barycentric;
        const bool quietMode;

        vector<float> h_vertices;
        vector<int>   h_triangles;
        vector<float> h_rayFrom;
        vector<float> h_rayTo;
        vector<int>   h_crossingDetected;
        vector<int>   h_intersectTriangle;
        vector<float> h_baryT, h_baryU, h_baryV;
        vector<uint64_t> h_morton;
        vector<int> h_sortedTriangleIDs;
        int nVertices, nTriangles, nRays;

        /// Device pointers
        float* d_vertices = nullptr;
        int* d_triangles = nullptr;
        float* d_rayFrom = nullptr;
        float* d_rayTo = nullptr;
        AABB* d_rayBox = nullptr;
        int* d_crossingDetected = nullptr;
        int* d_intersectTriangle = nullptr;
        float* d_baryT = nullptr;
        float* d_baryU = nullptr;
        float* d_baryV = nullptr;

        BVHNode* d_leafNodes = nullptr;
        BVHNode* d_internalNodes = nullptr;
        uint64_t* d_morton = nullptr;
        int* d_sortedTriangleIDs = nullptr;
        CollisionList* d_hitIDs = nullptr;
        InterceptDistances* d_interceptDists = nullptr;

        // Grid sizes
        int blockX;
        int gridXr;
        int gridXt;
        int gridXLambda;

        // CUDA timing
        cudaEvent_t start, end;

        // Sizes
        int sz_vertices;
        int sz_triangles;
        int sz_rays;
        int sz_rbox;
        int sz_id;
        int sz_bary;
        int sz_morton;
        int sz_sortedIDs;
        int sz_hitIDs;
        int sz_interceptDists;

        // Extents
        float minval[3];
        float maxval[3];
        float half_delta[3];
        float inv_delta[3];


        // Constructor to initialize consts
        RSI()
            : checkEnabled(true), largePosVal(2.5e8f), barycentric(true),
            quietMode(true),
            d_vertices(nullptr),d_triangles(nullptr),
            d_rayFrom(nullptr),d_rayTo(nullptr),
            d_rayBox(nullptr),
            d_crossingDetected(nullptr),
            d_intersectTriangle(nullptr),
            d_baryT(nullptr),
            d_baryU(nullptr),
            d_baryV(nullptr),
            d_leafNodes(nullptr),
            d_internalNodes(nullptr),
            d_morton(nullptr),
            d_sortedTriangleIDs(nullptr),
            d_hitIDs(nullptr),
            d_interceptDists(nullptr),
            blockX(0),gridXr(0),gridXt(0),
            gridXLambda(0),
            sz_vertices(0),
            sz_triangles(0),
            sz_rays(0),
            sz_rbox(0),
            sz_id(0),
            sz_bary(0),
            sz_morton(0),
            sz_sortedIDs(0),
            sz_hitIDs(0),
            sz_interceptDists(0)
        {
            // Initialize minval, maxval, half_delta, inv_delta arrays
            for (int i = 0; i < 3; ++i) {
                minval[i] = 0.0f;
                maxval[i] = 0.0f;
                half_delta[i] = 0.0f;
                inv_delta[i] = 0.0f;
            }
        }

        void setup(float* in_vertices, int* in_triangles, int num_vertices, int num_triangles, int num_rays){
            
            h_vertices.assign(in_vertices, in_vertices + 3 * num_vertices);
            h_triangles.assign(in_triangles, in_triangles + 3*num_triangles);
            nVertices = num_vertices;
            nTriangles = num_triangles;
            nRays = num_rays;

            sz_vertices = 3 * nVertices * sizeof(float);
            sz_triangles = 3 * nTriangles * sizeof(int);
            sz_rays = 3 * nRays * sizeof(float);
            sz_rbox = nRays * sizeof(AABB);
            sz_id = nRays * sizeof(int);
            sz_bary = nRays * sizeof(float);

            h_crossingDetected.resize(nRays);
         
            HANDLE_ERROR(cudaMalloc(&d_vertices, sz_vertices));
            HANDLE_ERROR(cudaMalloc(&d_triangles, sz_triangles));
            HANDLE_ERROR(cudaMalloc(&d_rayFrom, sz_rays));
            HANDLE_ERROR(cudaMalloc(&d_rayTo, sz_rays));
            HANDLE_ERROR(cudaMalloc(&d_rayBox, sz_rbox));
        
            h_intersectTriangle.resize(nRays);
            h_baryT.resize(nRays);
            h_baryU.resize(nRays);
            h_baryV.resize(nRays);
            HANDLE_ERROR(cudaMalloc(&d_intersectTriangle, sz_id));
            HANDLE_ERROR(cudaMalloc(&d_baryT, sz_bary));
            HANDLE_ERROR(cudaMalloc(&d_baryU, sz_bary));
            HANDLE_ERROR(cudaMalloc(&d_baryV, sz_bary));
                
            HANDLE_ERROR(cudaMemcpy(d_vertices, h_vertices.data(), sz_vertices, cudaMemcpyHostToDevice));
            HANDLE_ERROR(cudaMemcpy(d_triangles, h_triangles.data(), sz_triangles, cudaMemcpyHostToDevice));

            //grid partitions
            blockX = 1024,
            gridXr = (int)ceil((float)nRays / blockX),
            gridXt = (int)ceil((float)nTriangles / blockX),
            gridXLambda = 16; //N_{grids}
            if (! quietMode) {
                cout << blockX << " threads/block, grids: {triangles: "
                    << gridXt << ", rays: " << gridXLambda << "}" << endl;
            }

            //order triangles using Morton code
            //- normalise surface vertices to canvas coords
            getMinMaxExtentOfSurface<float>(h_vertices, minval, maxval, half_delta,
                inv_delta, nVertices, quietMode);
            //- convert centroid of triangles to morton code
            createMortonCode<float, uint64_t>(h_vertices, h_triangles,
                        minval, half_delta, inv_delta,
                        h_morton, nTriangles);
            //- sort before constructing binary radix tree
            sortMortonCode<uint64_t>(h_morton, h_sortedTriangleIDs);
            if (!quietMode && checkEnabled) {
                cout << "checking sortMortonCode" << endl;
                for (int j = 0; j < min(12, nTriangles); j++) {
                    cout << j << ": (" << h_sortedTriangleIDs[j] << ") "
                    << h_morton[j] << endl;
                }
            }


            sz_morton = nTriangles * sizeof(uint64_t);
            sz_sortedIDs = nTriangles * sizeof(int);
            sz_hitIDs = gridXLambda * blockX * sizeof(CollisionList);
            sz_interceptDists = gridXLambda * blockX * sizeof(InterceptDistances);
            //data structures used in agglomerative LBVH construction
            HANDLE_ERROR(cudaMalloc(&d_leafNodes, nTriangles * sizeof(BVHNode)));
            HANDLE_ERROR(cudaMalloc(&d_internalNodes, nTriangles * sizeof(BVHNode)));
            HANDLE_ERROR(cudaMalloc(&d_morton, sz_morton));
            HANDLE_ERROR(cudaMalloc(&d_sortedTriangleIDs, sz_sortedIDs));
            HANDLE_ERROR(cudaMalloc(&d_hitIDs, sz_hitIDs));

            HANDLE_ERROR(cudaMemcpy(d_morton, h_morton.data(), sz_morton, cudaMemcpyHostToDevice));
            HANDLE_ERROR(cudaMemcpy(d_sortedTriangleIDs, h_sortedTriangleIDs.data(), sz_sortedIDs, cudaMemcpyHostToDevice));
            std::vector<uint64_t>().swap(h_morton);
            std::vector<int>().swap(h_sortedTriangleIDs);

            bvhResetKernel<<<gridXt, blockX>>>(d_vertices, d_triangles,
                d_internalNodes, d_leafNodes,
                d_sortedTriangleIDs, nTriangles);
            HANDLE_ERROR(cudaDeviceSynchronize());

            bvhConstruct<uint64_t><<<gridXt, blockX>>>(d_internalNodes, d_leafNodes,
                                    d_morton, nTriangles);
            HANDLE_ERROR(cudaDeviceSynchronize());
            CUDA_SYNCHRO_CHECK();

        }

    void detect(float* rayFrom, float* rayTo){

        h_rayFrom.assign(rayFrom, rayFrom + 3*nRays);
        h_rayTo.assign(rayTo, rayTo + 3 * nRays);
        
        HANDLE_ERROR(cudaMemcpy(d_rayFrom, h_rayFrom.data(), sz_rays, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_rayTo, h_rayTo.data(), sz_rays, cudaMemcpyHostToDevice));

        // BVH BUILDING CODE, maybe relies on ray data to create

        //initialise arrays
        initArrayKernel<<<gridXr, blockX>>>(d_intersectTriangle, -1, nRays);
        initArrayKernel<<<gridXr, blockX>>>(d_baryT, largePosVal, nRays);

        HANDLE_ERROR(cudaDeviceSynchronize());

        //compute ray-segment bounding boxes
        rbxKernel<<<gridXr, blockX>>>(d_rayFrom, d_rayTo, d_rayBox, nRays);
        HANDLE_ERROR(cudaDeviceSynchronize());

        // END BVH CODE

        bvhIntersectionKernel<<<gridXLambda, blockX>>>(
                    d_vertices, d_triangles, d_rayFrom, d_rayTo,
                    d_internalNodes, d_rayBox, d_hitIDs,
                    d_intersectTriangle, d_baryT, d_baryU, d_baryV,
                    nTriangles, nRays);
                    
        HANDLE_ERROR(cudaDeviceSynchronize());
    
        HANDLE_ERROR(cudaMemcpy(h_intersectTriangle.data(), d_intersectTriangle,
                                sz_id, cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(h_baryT.data(), d_baryT, sz_bary, cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(h_baryU.data(), d_baryU, sz_bary, cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(h_baryV.data(), d_baryV, sz_bary, cudaMemcpyDeviceToHost));

        // printf("Detected triangle ID = %d\n", h_intersectTriangle[0]);
    
    }

    void destroy(){
        
        safeCudaFree(d_vertices);
        safeCudaFree(d_triangles);
        safeCudaFree(d_rayFrom);
        safeCudaFree(d_rayTo);
        safeCudaFree(d_rayBox);
        safeCudaFree(d_intersectTriangle);
        safeCudaFree(d_baryT);
        safeCudaFree(d_baryU);
        safeCudaFree(d_baryV);
        safeCudaFree(d_leafNodes);
        safeCudaFree(d_internalNodes);
        safeCudaFree(d_morton);
        safeCudaFree(d_sortedTriangleIDs);
        safeCudaFree(d_hitIDs);

    }
};

// int main(){
//     RSI* rsi = new RSI();

//     // 9 vertices (3 triangles, 3 vertices each)
//     float vertices_array[27] = {
//         // Triangle 0
//         0.0f, 0.0f, 0.0f,   // Vertex 0
//         1.0f, 0.0f, 0.0f,   // Vertex 1
//         0.0f, 1.0f, 0.0f,   // Vertex 2

//         // Triangle 1
//         2.0f, 0.0f, 0.0f,   // Vertex 3
//         3.0f, 0.0f, 0.0f,   // Vertex 4
//         2.0f, 1.0f, 0.0f,   // Vertex 5

//         // Triangle 2
//         4.0f, 0.0f, 0.0f,   // Vertex 6
//         5.0f, 0.0f, 0.0f,   // Vertex 7
//         4.0f, 1.0f, 0.0f    // Vertex 8
//     };

//     int triangles_array[9] = {
//         0, 1, 2,  // Triangle 0
//         3, 4, 5,  // Triangle 1
//         6, 7, 8   // Triangle 2
//     };

//     // Pointers
//     float* vertices = vertices_array;
//     int* triangles = triangles_array;

//     // Number of primitives
//     int num_vertices = 9;    // 9 vertices
//     int num_triangles = 3;   // 3 triangles
//     int num_rays = 1;        // 1 ray

//     // Define a ray that will hit the first triangle
//     float rayFrom_array[3] = { 0.5f, 0.5f, 1.0f };   // Above Triangle 0
//     float rayTo_array[3]   = { 0.5f, 0.5f, -1.0f };  // Going down through Triangle 0

//     // Setup
//     rsi->setup(vertices, triangles, num_vertices, num_triangles, num_rays);

//     // Detect
//     rsi->detect(rayFrom_array, rayTo_array);

//     // Optionally you might want to inspect rsi->h_intersectTriangle, rsi->h_baryT here
//     // Example:
//     printf("Intersected Triangle ID: %d\n", rsi->h_intersectTriangle[0]);
//     printf("Barycentric T: %f\n", rsi->h_baryT[0]);

//     // Destroy
//     rsi->destroy();

//     delete rsi;
//     return 0;
// }


// Create RSI object and call setup
extern "C" void* setup_RSI(float* vertices, int* triangles, int num_vertices, int num_triangles, int num_rays) {
    RSI* rsi = new RSI();
    rsi->setup(vertices, triangles, num_vertices, num_triangles, num_rays);
    return static_cast<void*>(rsi);
}

// Call detect on an existing RSI object
extern "C" void detect_RSI(void* rsi_obj, float* rayFrom, float* rayTo, int** out_intersectTriangle, float** out_baryT) {
    RSI* rsi = static_cast<RSI*>(rsi_obj);
    rsi->detect(rayFrom, rayTo);

    // printf("rayFrom = (%f, %f, %f)\n", rayFrom[0], rayFrom[1], rayFrom[2]);
    // printf("rayTo = (%f, %f, %f)\n", rayTo[0], rayTo[1], rayTo[2]);
    
    *out_intersectTriangle = rsi->h_intersectTriangle.data();
    *out_baryT = rsi->h_baryT.data();
}

// Destroy RSI object
extern "C" void destroy_RSI(void* rsi_obj) {
    RSI* rsi = static_cast<RSI*>(rsi_obj);
    delete rsi;
}


    