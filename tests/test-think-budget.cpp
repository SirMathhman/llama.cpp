// Standalone test for llama_sampler_init_think_budget.
// Runs the sampler chain directly on synthetic logits (llama_sampler_apply
// takes just a llama_sampler + llama_token_data_array -- no llama_context
// or loaded model needed), so this validates the sampler's logic without
// requiring a GGUF file.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static const int32_t N_VOCAB      = 8;
static const llama_token EOT      = 3; // pretend token 3 is </think>

static llama_token_data_array make_candidates(std::vector<llama_token_data> & buf) {
    buf.clear();
    for (int32_t i = 0; i < N_VOCAB; ++i) {
        // flat-ish logits, EOT deliberately NOT the argmax, so we can see
        // whether the bias is what pushes it to the top rather than luck
        float logit = (i == EOT) ? 0.0f : 1.0f;
        buf.push_back({ i, logit, 0.0f });
    }
    return llama_token_data_array{ buf.data(), buf.size(), -1, false };
}

static llama_token argmax(const llama_token_data_array & arr) {
    size_t best = 0;
    for (size_t i = 1; i < arr.size; ++i) {
        if (arr.data[i].logit > arr.data[best].logit) best = i;
    }
    return arr.data[best].id;
}

int main() {
    int failures = 0;

    // --- test 1: below ramp_start, no bias applied, EOT should not win ---
    {
        auto * smpl = llama_sampler_init_think_budget(
            /* n_budget     */ 100,
            /* ramp_start   */ 0.5f,
            /* max_bias     */ 10.0f,
            /* exponent     */ 2.0f,
            /* target_token */ EOT,
            /* hard_force   */ true);

        std::vector<llama_token_data> buf;
        auto cur_p = make_candidates(buf);

        // simulate 10 accepted tokens (10% of budget, below 50% ramp_start)
        for (int i = 0; i < 10; ++i) llama_sampler_accept(smpl, 0);

        llama_sampler_apply(smpl, &cur_p);
        llama_token top = argmax(cur_p);
        printf("[test 1] below ramp_start: argmax=%d (expect != %d)\n", top, EOT);
        if (top == EOT) { printf("  FAIL\n"); failures++; }

        llama_sampler_free(smpl);
    }

    // --- test 2: near budget (95%), ramp bias should push EOT to argmax ---
    {
        auto * smpl = llama_sampler_init_think_budget(
            100, 0.5f, 10.0f, 2.0f, EOT, /* hard_force */ false);

        std::vector<llama_token_data> buf;
        auto cur_p = make_candidates(buf);

        for (int i = 0; i < 95; ++i) llama_sampler_accept(smpl, 0);

        llama_sampler_apply(smpl, &cur_p);
        llama_token top = argmax(cur_p);
        printf("[test 2] near budget (95%%): argmax=%d (expect == %d)\n", top, EOT);
        if (top != EOT) { printf("  FAIL\n"); failures++; }

        llama_sampler_free(smpl);
    }

    // --- test 3: at/over budget with hard_force, must collapse onto EOT ---
    {
        auto * smpl = llama_sampler_init_think_budget(
            100, 0.5f, 10.0f, 2.0f, EOT, /* hard_force */ true);

        std::vector<llama_token_data> buf;
        auto cur_p = make_candidates(buf);

        for (int i = 0; i < 100; ++i) llama_sampler_accept(smpl, 0);

        llama_sampler_apply(smpl, &cur_p);
        bool others_neg_inf = true;
        for (size_t i = 0; i < cur_p.size; ++i) {
            if (cur_p.data[i].id != EOT && cur_p.data[i].logit != -INFINITY) {
                others_neg_inf = false;
            }
        }
        printf("[test 3] hard force at budget: all-others -inf = %s (expect true)\n",
               others_neg_inf ? "true" : "false");
        if (!others_neg_inf) { printf("  FAIL\n"); failures++; }

        llama_sampler_free(smpl);
    }

    // --- test 4: ordering hazard -- if EOT already pruned by an earlier
    // sampler in the chain, this sampler must be a silent no-op, not crash ---
    {
        auto * smpl = llama_sampler_init_think_budget(
            100, 0.5f, 10.0f, 2.0f, EOT, /* hard_force */ true);

        std::vector<llama_token_data> buf;
        auto cur_p = make_candidates(buf);
        // simulate top-k=2 having already pruned EOT out of the candidate set
        buf.erase(buf.begin() + EOT);
        cur_p.data = buf.data();
        cur_p.size = buf.size();

        for (int i = 0; i < 100; ++i) llama_sampler_accept(smpl, 0);

        bool crashed = false;
        llama_sampler_apply(smpl, &cur_p); // should just return, not touch memory out of bounds
        printf("[test 4] EOT pre-pruned: no crash = %s (expect true)\n", crashed ? "false" : "true");

        llama_sampler_free(smpl);
    }

    // --- test 5: reset() zeroes n_generated, confirming per-sequence reuse works ---
    {
        auto * smpl = llama_sampler_init_think_budget(
            100, 0.5f, 10.0f, 2.0f, EOT, /* hard_force */ false);

        for (int i = 0; i < 95; ++i) llama_sampler_accept(smpl, 0);
        llama_sampler_reset(smpl);

        std::vector<llama_token_data> buf;
        auto cur_p = make_candidates(buf);
        llama_sampler_apply(smpl, &cur_p);
        llama_token top = argmax(cur_p);
        printf("[test 5] after reset: argmax=%d (expect != %d)\n", top, EOT);
        if (top == EOT) { printf("  FAIL\n"); failures++; }

        llama_sampler_free(smpl);
    }

    // --- test 6 (regression): after target_token is actually emitted once,
    // the bias must turn off for the rest of generation. without this, the
    // sampler permanently drags every subsequent token toward target_token
    // once ramp_start is crossed, which is what caused a real runaway
    // repeated-</think> loop in production. ---
    {
        auto * smpl = llama_sampler_init_think_budget(
            100, 0.5f, 10.0f, 2.0f, EOT, /* hard_force */ false);

        for (int i = 0; i < 95; ++i) llama_sampler_accept(smpl, 0); // ramp to 95%
        llama_sampler_accept(smpl, EOT);                            // model naturally closes thinking

        // simulate several more tokens generated AFTER </think> (i.e. the
        // answer phase) and confirm EOT is no longer being pulled to the top
        // on any of them, the way it would if the bias were still armed
        bool leaked = false;
        for (int i = 0; i < 5; ++i) {
            std::vector<llama_token_data> buf;
            auto cur_p = make_candidates(buf); // EOT starts as a logit-loser again
            llama_sampler_apply(smpl, &cur_p);
            if (argmax(cur_p) == EOT) leaked = true;
            llama_sampler_accept(smpl, 1); // some ordinary answer token, not EOT
        }
        printf("[test 6] bias disarmed after natural </think>: leaked=%s (expect false)\n",
               leaked ? "true" : "false");
        if (leaked) { printf("  FAIL\n"); failures++; }

        llama_sampler_free(smpl);
    }

    if (failures == 0) {
        printf("\nALL TESTS PASSED\n");
    } else {
        printf("\n%d TEST(S) FAILED\n", failures);
    }
    return failures == 0 ? 0 : 1;
}
