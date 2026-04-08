/**
 * @file signal_demo.cpp
 * @brief Standalone C++26 demo: run all five strategy engines on synthetic data.
 *
 * Demonstrates the online tick() API for each engine and prints a summary
 * table of signals and positions.  This binary is the entry-point for
 * `bazel run //src/cpp:signal_demo`.
 *
 * @author Alpha Research Pod — 2026
 */

// Header search path provided by CMake / Bazel target deps:
//   #include "pdrrm_engine.hpp"   → src/cpp/pdrrm_engine.hpp
//   etc.
#include "math_utils.hpp"
#include "pdrrm_engine.hpp"
#include "strategies_engine.hpp"
#include "portfolio_optimizer.hpp"

#include <array>
#include <cmath>
#include <print>
#include <random>

using namespace alpha;

int main() {
    std::println("╔══════════════════════════════════════════════════════╗");
    std::println("║   Alpha Research — C++26 Signal Engine Demo          ║");
    std::println("╚══════════════════════════════════════════════════════╝\n");

    std::mt19937_64 rng{42};
    std::normal_distribution<double> nd{0.0, 1.0};

    constexpr int T = 300;   // days
    constexpr int N = 5;     // currencies

    // ── Synthetic data ─────────────────────────────────────────────────────
    math::MatrixXd nominal(T, N), be(T, N), surprises(T, N), fp(T, N), ret(T, N);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < N; ++i) {
            nominal(t,i)   = 0.03 + 0.01*i + 0.0003*nd(rng);
            be(t,i)        = 0.02 + 0.0002*nd(rng);
            surprises(t,i) = (t % 63 == 0 && i == 1) ? 0.0025 : 0.0;
            fp(t,i)        = nominal(t,i) - nominal(t,0);
            ret(t,i)       = 0.007 * nd(rng);
        }
    // BOJ hiking trend
    for (int t = 200; t < T; ++t) nominal(t,1) += 0.003 * (t-200)/100.0;

    // ── PDRRM ──────────────────────────────────────────────────────────────
    std::println("── PDRRM Engine ──────────────────────────────────────");
    pdrrm::PDRRMEngine pdrrm_eng;
    auto rrdm_r = pdrrm_eng.compute_rrdm(nominal, be);
    auto pss_r  = pdrrm_eng.compute_pss(surprises);
    auto rac_r  = pdrrm_eng.compute_rac(fp, ret);
    if (rrdm_r && pss_r && rac_r) {
        const auto& rrdm = *rrdm_r;
        const auto& pss  = *pss_r;
        const auto& rac  = *rac_r;

        // Build IS panel (first 60%)
        constexpr int T_IS = static_cast<int>(T * 0.6);
        math::MatrixXd X(T_IS * N, 3);
        math::VectorXd y(T_IS * N);
        for (int t = 0, row = 0; t < T_IS - 21; ++t)
            for (int i = 0; i < N; ++i, ++row) {
                X(row,0) = rrdm(t,i); X(row,1) = pss(t,i); X(row,2) = rac(t,i);
                y(row)   = ret.col(i).segment(t, 21).sum();
            }
        auto w_r = pdrrm_eng.update_weights(X, y);
        if (w_r) {
            auto& w = *w_r;
            std::println("  Weights: α_RRDM={:.4f}  α_PSS={:.4f}  α_RAC={:.4f}",
                         w[0], w[1], w[2]);
        }
        // Latest signals
        std::array<std::string_view,N> ccy{"JPY","EUR","GBP","AUD","CAD"};
        std::println("  Last-day signals:");
        for (int i = 0; i < N; ++i)
            std::println("    {:3s}  RRDM={:+.3f}  PSS={:+.3f}  RAC={:+.3f}",
                         ccy[i], rrdm(T-1,i), pss(T-1,i), rac(T-1,i));
    }

    // ── TPMCR ──────────────────────────────────────────────────────────────
    std::println("\n── TPMCR Engine ──────────────────────────────────────");
    strategies::TPMCREngine tpmcr_eng;
    strategies::TPMCREngine::Output tpmcr_out{};
    for (int t = 0; t < T; ++t) {
        strategies::TPMCREngine::DayData d{
            .tp_acm_10y     = 0.5 + 0.8*t/T + 0.05*nd(rng),
            .yield_2y       = 0.048 - 0.005*t/T,
            .yield_10y      = 0.045 - 0.003*t/T,
            .yield_30y      = 0.047,
            .swap_spread_30y= -0.002,
            .cds_5y_proxy   = 0.003,
            .returns        = {0.001*nd(rng),0.001*nd(rng),0.001*nd(rng),
                               0.001*nd(rng),0.001*nd(rng)},
        };
        tpmcr_out = tpmcr_eng.tick(d);
    }
    std::array<std::string_view,5> rates{"ZN","ZB","RX","G","TN"};
    std::println("  Regime: {}  TPM_z={:+.3f}  FSO_z={:+.3f}",
                 tpmcr_out.curve_regime, tpmcr_out.tpm_z, tpmcr_out.fiscal_stress_z);
    for (int i = 0; i < 5; ++i)
        std::println("    {:3s}  sig={:+.3f}  pos={:+.4f}",
                     rates[i], tpmcr_out.signals[i], tpmcr_out.positions[i]);

    // ── VSRA ───────────────────────────────────────────────────────────────
    std::println("\n── VSRA Engine ───────────────────────────────────────");
    strategies::VSRAEngine vsra_eng;
    strategies::VSRAEngine::Output vsra_out{};
    for (int t = 0; t < T; ++t) {
        vsra_out = vsra_eng.tick({
            .iv_atm   = 0.22 + 0.02*std::sin(t*0.1),
            .rv_21d   = 0.15 + 0.01*nd(rng),
            .vix_spot = 22.0 + 2*nd(rng),
            .vix_3m   = 24.0,
            .put_skew = 0.07 + 0.01*nd(rng),
            .rv_skew  = -0.25,
            .vx_return= 0.02*nd(rng),
        });
    }
    std::println("  VRP_z={:+.3f}  TSS_z={:+.3f}  SKA_z={:+.3f}  pos={:+.4f}",
                 vsra_out.vrp_z, vsra_out.tss_z, vsra_out.ska_z, vsra_out.position);

    // ── BacktestEngine ─────────────────────────────────────────────────────
    std::println("\n── BacktestEngine KPIs ──────────────────────────────");
    portfolio::BacktestEngine bt;
    math::MatrixXd sig_panel(T, N), ret_panel = ret;
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < N; ++i)
            sig_panel(t,i) = (*rrdm_r)(t,i);

    auto kpi_r = bt.run(sig_panel, ret_panel);
    if (kpi_r) {
        auto& k = *kpi_r;
        std::println("  Sharpe:  {:.3f}", k.sharpe_ratio);
        std::println("  Sortino: {:.3f}", k.sortino_ratio);
        std::println("  MDD:     {:.2f}%", k.max_drawdown*100);
        std::println("  Mean IC: {:.4f}", k.mean_ic);
        std::println("  ICIR:    {:.3f}", k.icir);
        std::println("  HL(days):{:.1f}  DecayAlert:{}", k.ic_half_life_days, k.decay_alert);
    }

    std::println("\n✅ Demo complete.");
    return 0;
}
