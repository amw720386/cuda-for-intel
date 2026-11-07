// cublasLtMatmul fp32 with bias epilogue, and RELU_BIAS epilogue
#include "common.h"
#include <cublasLt.h>
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { printf("FAIL %s:%d %s status=%d\n", __FILE__, __LINE__, #x, (int)s_); exit(1);} } while (0)
static bool run(cublasLtHandle_t lt, cublasLtEpilogue_t epi, const char* name) {
  int m = 64, n = 32, k = 48; std::vector<float> A(m * k), B(k * n), bias(m), C(m * n);
  for (int i = 0; i < m * k; i++) A[i] = ((i * 13) % 17) / 8.f - 1.f; for (int i = 0; i < k * n; i++) B[i] = ((i * 7) % 11) / 5.f - 1.f;
  for (int i = 0; i < m; i++) bias[i] = (i % 5) - 2.f;
  float *dA, *dB, *dC, *db; CK(cudaMalloc(&dA, m * k * 4)); CK(cudaMalloc(&dB, k * n * 4)); CK(cudaMalloc(&dC, m * n * 4)); CK(cudaMalloc(&db, m * 4));
  CK(cudaMemcpy(dA, A.data(), m * k * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB, B.data(), k * n * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(db, bias.data(), m * 4, cudaMemcpyHostToDevice));
  cublasLtMatmulDesc_t d; CB(cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  CB(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
  CB(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &db, sizeof(db)));
  cublasLtMatrixLayout_t la, lb, lc; CB(cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, m, k, m)); CB(cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, k, n, k)); CB(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, m, n, m));
  cublasLtMatmulPreference_t pref; CB(cublasLtMatmulPreferenceCreate(&pref)); size_t wsz = 4 << 20; void* ws; CK(cudaMalloc(&ws, wsz));
  CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));
  cublasLtMatmulHeuristicResult_t hr; int found = 0; CB(cublasLtMatmulAlgoGetHeuristic(lt, d, la, lb, lc, lc, pref, 1, &hr, &found));
  if (!found) { printf("FAIL %s no algo\n", name); return false; }
  float al = 1.f, be = 0.f;
  CB(cublasLtMatmul(lt, d, &al, dA, la, dB, lb, &be, dC, lc, dC, lc, &hr.algo, ws, wsz, 0));
  CK(cudaDeviceSynchronize()); CK(cudaMemcpy(C.data(), dC, m * n * 4, cudaMemcpyDeviceToHost)); bool ok = true;
  for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) { float s = bias[i]; float a = 0; for (int p = 0; p < k; p++) a += A[p * m + i] * B[j * k + p]; s += a;
    if (epi == CUBLASLT_EPILOGUE_RELU_BIAS && s < 0) s = 0; ok &= close_f(C[j * m + i], s); }
  EXPECT(ok, name); return ok; }
int main() {
  cublasLtHandle_t lt; CB(cublasLtCreate(&lt));
  run(lt, CUBLASLT_EPILOGUE_BIAS, "cublasLtMatmul_bias_epilogue");
  run(lt, CUBLASLT_EPILOGUE_RELU_BIAS, "cublasLtMatmul_relu_bias_epilogue");
  CB(cublasLtDestroy(lt)); printf(g_fail ? "RESULT FAIL\n" : "RESULT OK\n"); return g_fail; }
