// streams, events, async copies, host callbacks-free ordering, stream capture + graph launch, explicit graph API
#include "common.h"
__global__ void addc(float* p, float c, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) p[i] += c; }
__global__ void mulc(float* p, float c, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) p[i] *= c; }
int main() {
  const int n = 1 << 16; size_t sz = n * 4;
  float* hp; CK(cudaMallocHost(&hp, sz)); for (int i = 0; i < n; i++) hp[i] = 1.f;
  float* d; CK(cudaMalloc(&d, sz));
  cudaStream_t s1, s2; CK(cudaStreamCreate(&s1)); CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
  cudaEvent_t e0, e1, ev; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  CK(cudaEventRecord(e0, s1));
  CK(cudaMemcpyAsync(d, hp, sz, cudaMemcpyHostToDevice, s1));
  addc<<<n / 256, 256, 0, s1>>>(d, 2.f, n);                    // 3
  CK(cudaEventRecord(ev, s1)); CK(cudaStreamWaitEvent(s2, ev, 0));
  mulc<<<n / 256, 256, 0, s2>>>(d, 4.f, n);                    // 12
  CK(cudaMemcpyAsync(hp, d, sz, cudaMemcpyDeviceToHost, s2));
  CK(cudaEventRecord(e1, s2)); CK(cudaStreamSynchronize(s2));
  float ms = -1; CK(cudaEventElapsedTime(&ms, e0, e1));
  bool ok = true; for (int i = 0; i < n; i++) ok &= hp[i] == 12.f;
  EXPECT(ok, "streams_event_dependency"); EXPECT(ms >= 0.f, "event_elapsed_time"); EXPECT(cudaEventQuery(e1) == cudaSuccess, "event_query");
  CK(cudaMemsetAsync(d, 0, sz, s1)); CK(cudaStreamSynchronize(s1));
  // stream capture graph: d = ((0+1)*3)+5 per replay iteration order
  cudaGraph_t g; cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal));
  addc<<<n / 256, 256, 0, s1>>>(d, 1.f, n); mulc<<<n / 256, 256, 0, s1>>>(d, 3.f, n); addc<<<n / 256, 256, 0, s1>>>(d, 5.f, n);
  CK(cudaStreamEndCapture(s1, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
  for (int r = 0; r < 3; r++) CK(cudaGraphLaunch(ge, s1));
  CK(cudaStreamSynchronize(s1)); CK(cudaMemcpy(hp, d, sz, cudaMemcpyDeviceToHost));
  float v = 0; for (int r = 0; r < 3; r++) v = (v + 1.f) * 3.f + 5.f;
  ok = true; for (int i = 0; i < n; i++) ok &= hp[i] == v; EXPECT(ok, "stream_capture_graph_replay");
  size_t nodes = 0; { cudaError_t e = cudaGraphGetNodes(g, nullptr, &nodes); if (e == cudaErrorNotSupported || e == cudaErrorCallRequiresNewerDriver) printf("SKIP graph_node_count (%s)\n", cudaGetErrorString(e)); else { CK(e); EXPECT(nodes == 3, "graph_node_count"); } }
  // explicit graph API
  cudaGraph_t g2; { cudaError_t ge0 = cudaGraphCreate(&g2, 0); if (ge0 == cudaErrorNotSupported || ge0 == cudaErrorCallRequiresNewerDriver) { printf("SKIP explicit_kernel_node_graph (%s)\n", cudaGetErrorString(ge0)); printf(g_fail ? "RESULT FAIL\n" : "RESULT OK\n"); return g_fail; } CK(ge0); } CK(cudaMemset(d, 0, sz));
  void* args[3]; float c1 = 7.f; int nn = n; args[0] = &d; args[1] = &c1; args[2] = &nn;
  cudaKernelNodeParams kp = {}; kp.func = (void*)addc; kp.gridDim = dim3(n / 256); kp.blockDim = dim3(256); kp.kernelParams = args;
  cudaGraphNode_t nd; CK(cudaGraphAddKernelNode(&nd, g2, nullptr, 0, &kp));
  cudaGraphExec_t ge2; CK(cudaGraphInstantiate(&ge2, g2, 0)); CK(cudaGraphLaunch(ge2, s1)); CK(cudaStreamSynchronize(s1));
  CK(cudaMemcpy(hp, d, sz, cudaMemcpyDeviceToHost)); ok = true; for (int i = 0; i < n; i++) ok &= hp[i] == 7.f; EXPECT(ok, "explicit_kernel_node_graph");
  CK(cudaDeviceSynchronize()); printf(g_fail ? "RESULT FAIL\n" : "RESULT OK\n"); return g_fail; }
