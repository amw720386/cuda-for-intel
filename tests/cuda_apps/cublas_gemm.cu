// cublasSgemm (col-major, transposes), cublasGemmEx fp32 and fp16-in/fp32-acc, strided batched
#include "common.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { printf("FAIL %s:%d %s status=%d\n", __FILE__, __LINE__, #x, (int)s_); exit(1);} } while (0)
static void ref_gemm(int m, int n, int k, const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) { // col-major, no trans
  for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) { float s = 0; for (int p = 0; p < k; p++) s += A[p * m + i] * B[j * k + p]; C[j * m + i] = s; } }
int main() {
  cublasHandle_t h; CB(cublasCreate(&h));
  int m = 48, n = 40, k = 32; std::vector<float> A(m * k), B(k * n), C(m * n), R(m * n);
  for (int i = 0; i < m * k; i++) A[i] = ((i * 13) % 17) / 8.f - 1.f; for (int i = 0; i < k * n; i++) B[i] = ((i * 7) % 11) / 5.f - 1.f;
  ref_gemm(m, n, k, A, B, R);
  float *dA, *dB, *dC; CK(cudaMalloc(&dA, m * k * 4)); CK(cudaMalloc(&dB, k * n * 4)); CK(cudaMalloc(&dC, m * n * 4));
  CK(cudaMemcpy(dA, A.data(), m * k * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB, B.data(), k * n * 4, cudaMemcpyHostToDevice));
  float al = 1.f, be = 0.f;
  CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, dA, m, dB, k, &be, dC, m));
  CK(cudaMemcpy(C.data(), dC, m * n * 4, cudaMemcpyDeviceToHost)); bool ok = true; for (int i = 0; i < m * n; i++) ok &= close_f(C[i], R[i]);
  EXPECT(ok, "cublasSgemm_NN");
  // alpha/beta: C = 2*A*B + 0.5*C_prev
  { al = 2.f; be = 0.5f; CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, dA, m, dB, k, &be, dC, m));
    CK(cudaMemcpy(C.data(), dC, m * n * 4, cudaMemcpyDeviceToHost)); ok = true; for (int i = 0; i < m * n; i++) ok &= close_f(C[i], 2.5f * R[i]); EXPECT(ok, "cublasSgemm_alpha_beta"); }
  { // transposed A: pass At (k x m col-major) with OP_T
    std::vector<float> At(k * m); for (int i = 0; i < m; i++) for (int p = 0; p < k; p++) At[i * k + p] = A[p * m + i];
    float* dAt; CK(cudaMalloc(&dAt, k * m * 4)); CK(cudaMemcpy(dAt, At.data(), k * m * 4, cudaMemcpyHostToDevice)); al = 1.f; be = 0.f;
    CB(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &al, dAt, k, dB, k, &be, dC, m));
    CK(cudaMemcpy(C.data(), dC, m * n * 4, cudaMemcpyDeviceToHost)); ok = true; for (int i = 0; i < m * n; i++) ok &= close_f(C[i], R[i]); EXPECT(ok, "cublasSgemm_TN"); }
  CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, dA, CUDA_R_32F, m, dB, CUDA_R_32F, k, &be, dC, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  CK(cudaMemcpy(C.data(), dC, m * n * 4, cudaMemcpyDeviceToHost)); ok = true; for (int i = 0; i < m * n; i++) ok &= close_f(C[i], R[i]); EXPECT(ok, "cublasGemmEx_f32");
  { std::vector<__half> hA(m * k), hB(k * n); std::vector<float> qA(m * k), qB(k * n), RR(m * n);
    for (int i = 0; i < m * k; i++) { hA[i] = __float2half(A[i]); qA[i] = __half2float(hA[i]); } for (int i = 0; i < k * n; i++) { hB[i] = __float2half(B[i]); qB[i] = __half2float(hB[i]); }
    ref_gemm(m, n, k, qA, qB, RR); __half *pA, *pB; CK(cudaMalloc(&pA, m * k * 2)); CK(cudaMalloc(&pB, k * n * 2));
    CK(cudaMemcpy(pA, hA.data(), m * k * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(pB, hB.data(), k * n * 2, cudaMemcpyHostToDevice));
    CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, pA, CUDA_R_16F, m, pB, CUDA_R_16F, k, &be, dC, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CK(cudaMemcpy(C.data(), dC, m * n * 4, cudaMemcpyDeviceToHost)); ok = true; for (int i = 0; i < m * n; i++) ok &= close_f(C[i], RR[i], 2e-3f); EXPECT(ok, "cublasGemmEx_f16in_f32out"); }
  { int bc = 3; std::vector<float> Ab(m * k * bc), Bb(k * n * bc), Cb(m * n * bc);
    for (int b = 0; b < bc; b++) for (int i = 0; i < m * k; i++) Ab[b * m * k + i] = A[i] * (b + 1); for (int b = 0; b < bc; b++) for (int i = 0; i < k * n; i++) Bb[b * k * n + i] = B[i];
    float *bA, *bB, *bC; CK(cudaMalloc(&bA, Ab.size() * 4)); CK(cudaMalloc(&bB, Bb.size() * 4)); CK(cudaMalloc(&bC, Cb.size() * 4));
    CK(cudaMemcpy(bA, Ab.data(), Ab.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(bB, Bb.data(), Bb.size() * 4, cudaMemcpyHostToDevice));
    CB(cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, bA, m, m * k, bB, k, k * n, &be, bC, m, m * n, bc));
    CK(cudaMemcpy(Cb.data(), bC, Cb.size() * 4, cudaMemcpyDeviceToHost)); ok = true;
    for (int b = 0; b < bc; b++) for (int i = 0; i < m * n; i++) ok &= close_f(Cb[b * m * n + i], R[i] * (b + 1)); EXPECT(ok, "cublasSgemmStridedBatched"); }
  CB(cublasDestroy(h)); CK(cudaDeviceSynchronize()); printf(g_fail ? "RESULT FAIL\n" : "RESULT OK\n"); return g_fail; }
