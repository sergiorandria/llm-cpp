// tests/test_train_mamba.cpp — Trainer wiring for the Mamba path.
// Exercises the full stack that Gap 1 enabled: Trainer::train_step /
// train_step_accum -> GPT::backward (mamba) -> Adam step, plus grad-clip
// accounting and the NaN-rollback API. Explicit checks (no assert-only;
// Release NDEBUG elides assert bodies).
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "llm/dataset.h"
#include "llm/loss.h"
#include "llm/model.h"
#include "llm/optimizer.h"
#include "llm/scheduler.h"
#include "llm/trainer.h"

#define CHECK(cond, msg)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            std::cerr << "FAIL: " << msg << "\n";         \
            return 1;                                     \
        }                                                 \
    } while (0)

static llm::Config mamba_cfg() {
    llm::Config cfg;
    cfg.vocab_size = 16;
    cfg.n_embd = 8;
    cfg.n_heads = 2;
    cfg.n_layers = 1;
    cfg.block_size = 8;
    cfg.weight_tying = true;
    cfg.use_mamba = true;
    cfg.d_inner = 8;
    cfg.d_state = 4;
    cfg.dt_rank = 2;
    cfg.conv_kernel = 2;
    return cfg;
}

int main() {
    using namespace llm;
    std::vector<int> batch = {1, 2, 3, 4, 5, 6, 7, 0};

    // 1. train_step moves weights, loss finite, grad norm clipped.
    {
        GPT model(mamba_cfg());
        Adam optim(0.02f);
        CosineScheduler sched(0.02f, 0, 50);
        TrainConfig tcfg;
        tcfg.max_iters = 50;
        tcfg.grad_clip = 1.0f;
        Trainer trainer(model, tcfg, optim, sched);

        auto ps = model.parameters();
        std::vector<float> before;
        for (auto* p : ps) before.insert(before.end(), p->data.begin(), p->data.end());

        float l0 = trainer.train_step(batch);
        CHECK(std::isfinite(l0), "train_step loss non-finite");
        float diff = 0;
        size_t idx = 0;
        for (auto* p : model.parameters())
            for (float v : p->data) diff += std::abs(v - before[idx++]);
        CHECK(diff > 1e-6f, "trainer did not update mamba weights");
        CHECK(trainer.last_grad_norm() <= tcfg.grad_clip + 1e-3f, "grad norm exceeds clip");
        CHECK(trainer.skipped_steps() == 0, "clean step wrongly skipped");

        // 2. Loss decreases over Adam steps (overfit smoke through Trainer).
        float prev = l0;
        for (int s = 0; s < 15; ++s) {
            float l = trainer.train_step(batch);
            CHECK(std::isfinite(l), "non-finite loss during mamba training");
            prev = l;
        }
        std::cout << "mamba trainer: " << l0 << " -> " << prev << "\n";
        CHECK(prev < l0, "mamba loss did not decrease via Trainer+Adam");
    }

    // 3. train_step_accum: no step before K micros, steps at K.
    {
        GPT model(mamba_cfg());
        Adam optim(0.02f);
        CosineScheduler sched(0.02f, 0, 50);
        TrainConfig tcfg;
        tcfg.grad_accum_steps = 2;
        Trainer trainer(model, tcfg, optim, sched);

        auto snapshot = [&]() {
            std::vector<float> s;
            for (auto* p : model.parameters()) s.insert(s.end(), p->data.begin(), p->data.end());
            return s;
        };
        auto before = snapshot();
        float l1 = trainer.train_step_accum(batch);
        CHECK(std::isfinite(l1), "accum micro loss non-finite");
        auto after_one = snapshot();
        double moved1 = 0;
        for (size_t i = 0; i < before.size(); ++i) moved1 += std::abs(after_one[i] - before[i]);
        CHECK(moved1 == 0.0, "accum stepped before K micros");
        float l2 = trainer.train_step_accum(batch);
        CHECK(std::isfinite(l2), "accum step loss non-finite");
        auto after_two = snapshot();
        double moved2 = 0;
        for (size_t i = 0; i < before.size(); ++i) moved2 += std::abs(after_two[i] - before[i]);
        CHECK(moved2 > 1e-6, "accum did not step at K micros");
        std::cout << "mamba accum ok (loss " << l1 << " -> " << l2 << ")\n";
    }

    // 4. NaN-rollback API on a mamba model (snapshot + restore + halve LR).
    {
        GPT model(mamba_cfg());
        Adam optim(0.02f);
        CosineScheduler sched(0.02f, 0, 50);
        TrainConfig tcfg;
        Trainer trainer(model, tcfg, optim, sched);
        trainer.snapshot_params();
        auto good = model.parameters()[0]->data;
        model.parameters()[0]->data[0] += 5.0f;
        float lr_before = optim.get_lr();
        CHECK(trainer.rollback_if_nonfinite(std::numeric_limits<float>::infinity()),
              "rollback did not trigger on inf loss");
        CHECK(model.parameters()[0]->data == good, "rollback did not restore params");
        CHECK(optim.get_lr() < lr_before, "rollback did not halve LR");
        CHECK(!trainer.rollback_if_nonfinite(1.0f), "rollback triggered on finite loss");
        std::cout << "mamba rollback ok\n";
    }

    std::cout << "test_train_mamba passed\n";
    return 0;
}
