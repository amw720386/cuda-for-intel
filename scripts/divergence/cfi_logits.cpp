// cfi_logits: dump per-step next-token logits of a GGUF model through llama.cpp (any ggml backend).
//
// usage: cfi_logits <model.gguf> <ngl> <gen|force> <n_gen> <prompt.txt> <ids.txt|-> <out_prefix>
//   gen   : greedy free run for n_gen steps (stops at an end-of-generation token)
//   force : teacher-forcing; <ids.txt> holds the generated token ids (whitespace separated), one logits row per id
// The prompt is evaluated as ONE batch (like a server prefill), every later token as a batch of one (decode),
// so the prefill (mmq) and decode (mmvq) kernel paths of a real generation are both exercised.
// Output: <out_prefix>.logits.bin  float32 [steps][n_vocab]
//         <out_prefix>.ids.txt     line 1: n_vocab n_prompt; line 2: prompt ids; line 3: generated ids
// Step 0 = logits after the prompt (predict gen[0]); step i = logits after feeding gen[i-1] (predict gen[i]).
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iterator>

int main(int argc, char ** argv) {
    if (argc < 8) { fprintf(stderr, "usage: %s model ngl gen|force n_gen prompt.txt ids.txt|- out_prefix\n", argv[0]); return 2; }
    const std::string model_path = argv[1];
    const int ngl = atoi(argv[2]);
    const bool force = std::string(argv[3]) == "force";
    int n_gen = atoi(argv[4]);
    std::ifstream pf(argv[5], std::ios::binary);
    std::string prompt((std::istreambuf_iterator<char>(pf)), std::istreambuf_iterator<char>());
    std::vector<llama_token> forced;
    if (force) {
        std::ifstream f(argv[6]); long long t;
        while (f >> t) forced.push_back((llama_token) t);
        n_gen = (int) forced.size();
    }
    const std::string out = argv[7];

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> ptoks(prompt.size() + 8);
    int n = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), ptoks.data(), (int) ptoks.size(), llama_vocab_get_add_bos(vocab), false);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    ptoks.resize(n);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.n_threads = 4; cp.n_threads_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }

    FILE * lf = fopen((out + ".logits.bin").c_str(), "wb");
    std::vector<llama_token> gen;
    llama_batch b = llama_batch_get_one(ptoks.data(), (int) ptoks.size());
    for (int step = 0; step < n_gen; ++step) {
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed at step %d\n", step); return 1; }
        const float * lg = llama_get_logits_ith(ctx, -1);
        fwrite(lg, sizeof(float), n_vocab, lf);
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) if (lg[i] > lg[best]) best = i;
        llama_token next = force ? forced[step] : (llama_token) best;
        gen.push_back(next);
        if (!force && llama_vocab_is_eog(vocab, next)) break;
        b = llama_batch_get_one(&gen.back(), 1);
    }
    fclose(lf);
    std::ofstream idf(out + ".ids.txt");
    idf << n_vocab << " " << ptoks.size() << "\n";
    for (auto t : ptoks) idf << t << " ";
    idf << "\n";
    for (auto t : gen) idf << t << " ";
    idf << "\n";
    llama_free(ctx); llama_model_free(model);
    return 0;
}
