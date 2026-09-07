// Soft "reasoning budget" sampler.
//
// Instead of llama.cpp's existing --reasoning-budget behavior (hard truncate
// at N tokens, then splice in the closing think token regardless of where
// generation was), this sampler *biases* generation toward a target token
// (typically the model's "end of thinking" token, e.g. </think>) as the
// token count approaches the budget, on a ramp, rather than forcing it
// abruptly at the boundary. The bias is 0 until `ramp_start` (a fraction of
// the budget), then grows from 0 to `max_bias` as budget is exhausted.
//
// This does NOT make the model "know" it is running low the way an
// in-context textual reminder would -- it only reshapes the sampling
// distribution. It's meant to pair with in-context budget hints (which
// should do the actual behavioral work) as a soft safety net, with an
// optional hard force still available as a last-resort ceiling.

#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>

struct llama_sampler_think_budget {
    const int32_t     n_budget;     // total token budget for the "thinking" phase
    const float        ramp_start;   // fraction of budget (0..1) at which bias starts ramping
    const float         max_bias;     // logit bias applied at 100% of budget (before hard force)
    const float         exponent;     // ramp shape: 1 = linear, >1 = back-loaded (stays low longer)
    const llama_token target_token;  // token to bias toward, e.g. the model's </think> id
    const bool     hard_force;   // if true, force target_token once n_generated >= n_budget

    int32_t n_generated; // mutable: count of tokens accepted since last reset
    bool    armed;       // mutable: false once target_token has actually been emitted --
                          // without this, the bias never turns off and permanently drags
                          // every subsequent token (including deep into the answer) toward
                          // target_token once the ramp threshold is crossed, causing a
                          // runaway repetition loop. armed mirrors rbudget's COUNTING->DONE
                          // transition: once the target is naturally emitted, this sampler
                          // goes inert for the rest of generation, same as rbudget does.
};

static const char * llama_sampler_think_budget_name(const struct llama_sampler * /*smpl*/) {
    return "think-budget";
}

static void llama_sampler_think_budget_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (llama_sampler_think_budget *) smpl->ctx;
    ctx->n_generated++;
    if (token == ctx->target_token) {
        ctx->armed = false; // thinking closed naturally -- stop biasing for the rest of generation
    }
}

static void llama_sampler_think_budget_reset(struct llama_sampler * smpl) {
    auto * ctx = (llama_sampler_think_budget *) smpl->ctx;
    ctx->n_generated = 0;
    ctx->armed = true;
}

static void llama_sampler_think_budget_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_think_budget *) smpl->ctx;

    if (ctx->n_budget <= 0 || !ctx->armed) {
        return;
    }

    const float frac = (float) ctx->n_generated / (float) ctx->n_budget;

    // find the target token's current slot in cur_p (its position may have
    // been shuffled by an earlier sampler in the chain, e.g. top-k/top-p, so
    // we can't assume idx == id the way logit_bias's fast path does)
    size_t idx = cur_p->size;
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (cur_p->data[i].id == ctx->target_token) {
            idx = i;
            break;
        }
    }

    if (idx == cur_p->size) {
        // target token was pruned by an earlier sampler in the chain (e.g.
        // top-k already cut it) -- nothing we can bias. this is a real
        // ordering hazard: this sampler MUST run before any truncating
        // sampler (top-k/top-p/min-p) or it can silently become a no-op.
        return;
    }

    if (ctx->hard_force && ctx->n_generated >= ctx->n_budget) {
        // force: collapse the distribution onto the target token
        for (size_t i = 0; i < cur_p->size; ++i) {
            cur_p->data[i].logit = (i == idx) ? 0.0f : -INFINITY;
        }
        return;
    }

    if (frac < ctx->ramp_start) {
        return;
    }

    // normalize into [0, 1] over the ramp window, apply shape, scale by max_bias
    float t = (frac - ctx->ramp_start) / std::max(1e-6f, (1.0f - ctx->ramp_start));
    t = std::clamp(t, 0.0f, 1.0f);
    t = std::pow(t, ctx->exponent);

    cur_p->data[idx].logit += t * ctx->max_bias;
}

static struct llama_sampler * llama_sampler_think_budget_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const llama_sampler_think_budget *) smpl->ctx;
    auto * result = new llama_sampler_think_budget(*ctx);
    return llama_sampler_init(smpl->iface, result);
}

static void llama_sampler_think_budget_free(struct llama_sampler * smpl) {
    delete (llama_sampler_think_budget *) smpl->ctx;
}

static struct llama_sampler_i llama_sampler_think_budget_i = {
    /* .name              = */ llama_sampler_think_budget_name,
    /* .accept            = */ llama_sampler_think_budget_accept,
    /* .apply             = */ llama_sampler_think_budget_apply,
    /* .reset             = */ llama_sampler_think_budget_reset,
    /* .clone             = */ llama_sampler_think_budget_clone,
    /* .free              = */ llama_sampler_think_budget_free,
    /* .backend_init      = */ nullptr, // CPU-only: no GPU graph path implemented (see notes)
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};

struct llama_sampler * llama_sampler_init_think_budget(
        int32_t     n_budget,
        float       ramp_start,
        float       max_bias,
        float       exponent,
        llama_token target_token,
        bool        hard_force) {
    return llama_sampler_init(
        /* .iface = */ &llama_sampler_think_budget_i,
        /* .ctx   = */ new llama_sampler_think_budget {
            /* .n_budget     = */ n_budget,
            /* .ramp_start   = */ ramp_start,
            /* .max_bias     = */ max_bias,
            /* .exponent     = */ exponent,
            /* .target_token = */ target_token,
            /* .hard_force   = */ hard_force,
            /* .n_generated  = */ 0,
            /* .armed        = */ true,
        }
    );
}
