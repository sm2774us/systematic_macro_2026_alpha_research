/**
 * @file bindings.cpp — nanobind Python/C++23 bridge.
 *
 * Fixes vs original:
 *  1. nb::dict built via d["k"]=v  (pybind11-style dict(key_a=val) is NOT nanobind API)
 *  2. tick() lambdas accept numpy arrays (reduces param count below analyze_method limit)
 *  3. __repr__ uses snprintf (avoids GCC-14 consteval/lambda conflict with std::format)
 *  4. RingBuffer<N>::convolve now uses size_t K (fixed in math_utils.hpp)
 *
 * @author Alpha Research Pod — 2026
 */

#include "math_utils.hpp"
#include "pdrrm_engine.hpp"
#include "portfolio_optimizer.hpp"
#include "strategies_engine.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <cstdio>

namespace nb = nanobind;
using namespace nb::literals;

using NpArray2D = nb::ndarray<double, nb::ndim<2>, nb::c_contig>;
using NpArray1D = nb::ndarray<double, nb::ndim<1>, nb::c_contig>;

static Eigen::MatrixXd np2mat(const NpArray2D& a) {
    return Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,
        Eigen::RowMajor>>(a.data(), (int)a.shape(0), (int)a.shape(1));
}
static Eigen::VectorXd np2vec(const NpArray1D& a) {
    return Eigen::Map<const Eigen::VectorXd>(a.data(), (int)a.shape(0));
}
static nb::object mat2np(const Eigen::MatrixXd& m) {
    auto* d = new double[m.size()];
    Eigen::Map<Eigen::MatrixXd>(d, m.rows(), m.cols()) = m;
    nb::capsule del(d, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return nb::cast(nb::ndarray<nb::numpy,double,nb::ndim<2>>(
        d, {(size_t)m.rows(),(size_t)m.cols()}, del));
}
static nb::object vec2np(const Eigen::VectorXd& v) {
    auto* d = new double[v.size()];
    Eigen::Map<Eigen::VectorXd>(d, v.size()) = v;
    nb::capsule del(d, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return nb::cast(nb::ndarray<nb::numpy,double,nb::ndim<1>>(d, {(size_t)v.size()}, del));
}

NB_MODULE(alpha_cpp, m) {
    m.doc() = "alpha_cpp: C++23 signal engines for systematic macro research.";

    // KPIBundle -----------------------------------------------------------
    nb::class_<alpha::portfolio::KPIBundle>(m, "KPIBundle")
        .def(nb::init<>())
        .def_rw("sharpe_ratio",       &alpha::portfolio::KPIBundle::sharpe_ratio)
        .def_rw("sortino_ratio",      &alpha::portfolio::KPIBundle::sortino_ratio)
        .def_rw("calmar_ratio",       &alpha::portfolio::KPIBundle::calmar_ratio)
        .def_rw("cagr",               &alpha::portfolio::KPIBundle::cagr)
        .def_rw("total_return",       &alpha::portfolio::KPIBundle::total_return)
        .def_rw("max_drawdown",       &alpha::portfolio::KPIBundle::max_drawdown)
        .def_rw("annual_vol",         &alpha::portfolio::KPIBundle::annual_vol)
        .def_rw("var_95",             &alpha::portfolio::KPIBundle::var_95)
        .def_rw("expected_shortfall", &alpha::portfolio::KPIBundle::expected_shortfall)
        .def_rw("mean_ic",            &alpha::portfolio::KPIBundle::mean_ic)
        .def_rw("icir",               &alpha::portfolio::KPIBundle::icir)
        .def_rw("hit_rate",           &alpha::portfolio::KPIBundle::hit_rate)
        .def_rw("avg_turnover",       &alpha::portfolio::KPIBundle::avg_turnover)
        .def_rw("ic_half_life_days",  &alpha::portfolio::KPIBundle::ic_half_life_days)
        .def_rw("decay_alert",        &alpha::portfolio::KPIBundle::decay_alert)
        .def_rw("gross_alpha_bps",    &alpha::portfolio::KPIBundle::gross_alpha_bps)
        .def_rw("tc_drag_bps",        &alpha::portfolio::KPIBundle::tc_drag_bps)
        .def_rw("net_alpha_bps",      &alpha::portfolio::KPIBundle::net_alpha_bps)
        // snprintf avoids GCC-14 consteval issue triggered by std::format in lambdas
        .def("__repr__", [](const alpha::portfolio::KPIBundle& k) -> std::string {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "KPIBundle(sharpe=%.3f sortino=%.3f calmar=%.3f "
                "mdd=%.2f%% ic=%.4f icir=%.3f hl=%.1fd)",
                k.sharpe_ratio, k.sortino_ratio, k.calmar_ratio,
                k.max_drawdown*100.0, k.mean_ic, k.icir, k.ic_half_life_days);
            return std::string(buf);
        });

    // PDRRMConfig ---------------------------------------------------------
    nb::class_<alpha::pdrrm::PDRRMConfig>(m, "PDRRMConfig")
        .def(nb::init<>())
        .def_rw("rrdm_momentum_window", &alpha::pdrrm::PDRRMConfig::rrdm_momentum_window)
        .def_rw("pss_ewma_halflife",    &alpha::pdrrm::PDRRMConfig::pss_ewma_halflife)
        .def_rw("rac_vol_window",       &alpha::pdrrm::PDRRMConfig::rac_vol_window)
        .def_rw("ridge_lambda",         &alpha::pdrrm::PDRRMConfig::ridge_lambda)
        .def_rw("target_vol",           &alpha::pdrrm::PDRRMConfig::target_vol)
        .def_rw("max_position",         &alpha::pdrrm::PDRRMConfig::max_position);

    // PDRRMEngine ---------------------------------------------------------
    nb::class_<alpha::pdrrm::PDRRMEngine>(m, "PDRRMEngine")
        .def(nb::init<alpha::pdrrm::PDRRMConfig>(), "cfg"_a = alpha::pdrrm::PDRRMConfig{})
        .def("compute_rrdm",
             [](alpha::pdrrm::PDRRMEngine& self, const NpArray2D& nom, const NpArray2D& be) {
                 auto r = self.compute_rrdm(np2mat(nom), np2mat(be));
                 if (!r) throw std::runtime_error(r.error().message);
                 return mat2np(*r);
             }, "nominal_rates"_a, "breakevens"_a)
        .def("compute_pss",
             [](alpha::pdrrm::PDRRMEngine& self, const NpArray2D& s) {
                 auto r = self.compute_pss(np2mat(s));
                 if (!r) throw std::runtime_error(r.error().message);
                 return mat2np(*r);
             }, "cb_surprises"_a)
        .def("compute_rac",
             [](alpha::pdrrm::PDRRMEngine& self, const NpArray2D& fp, const NpArray2D& ret) {
                 auto r = self.compute_rac(np2mat(fp), np2mat(ret));
                 if (!r) throw std::runtime_error(r.error().message);
                 return mat2np(*r);
             }, "forward_premium"_a, "returns"_a)
        .def("update_weights",
             [](alpha::pdrrm::PDRRMEngine& self, const NpArray2D& X, const NpArray1D& y) {
                 auto r = self.update_weights(np2mat(X), np2vec(y));
                 if (!r) throw std::runtime_error(r.error().message);
                 return vec2np(*r);
             }, "X_is"_a, "y_is"_a);

    // BacktestEngine ------------------------------------------------------
    nb::class_<alpha::portfolio::BacktestEngine>(m, "BacktestEngine")
        .def(nb::init<>())
        .def("run",
             [](const alpha::portfolio::BacktestEngine& self,
                const NpArray2D& sig, const NpArray2D& ret,
                double tc, double ann) {
                 auto r = self.run(np2mat(sig), np2mat(ret), tc, ann);
                 if (!r) throw std::runtime_error(r.error().message);
                 return *r;
             }, "signals"_a, "returns"_a, "tc_rate"_a=0.0001, "ann_days"_a=252.0);

    // SignalDecayMonitor --------------------------------------------------
    nb::class_<alpha::portfolio::SignalDecayMonitor>(m, "SignalDecayMonitor")
        .def(nb::init<int>(), "window"_a = 60)
        .def("update",          &alpha::portfolio::SignalDecayMonitor::update, "ic"_a)
        .def("half_life",       &alpha::portfolio::SignalDecayMonitor::half_life)
        .def("rolling_mean_ic", &alpha::portfolio::SignalDecayMonitor::rolling_mean_ic)
        .def("alert",           &alpha::portfolio::SignalDecayMonitor::alert);

    // TPMCREngine ---------------------------------------------------------
    // tick(data[6], returns[5]) -> dict
    // data = [tp_acm_10y, yield_2y, yield_10y, yield_30y, swap_spread_30y, cds_5y_proxy]
    nb::class_<alpha::strategies::TPMCREngine>(m, "TPMCREngine")
        .def(nb::init<alpha::strategies::TPMCRConfig>(),
             "cfg"_a = alpha::strategies::TPMCRConfig{})
        .def("tick",
             [](alpha::strategies::TPMCREngine& self,
                const NpArray1D& data, const NpArray1D& rets) {
                 alpha::strategies::TPMCREngine::DayData d{};
                 d.tp_acm_10y      = data.data()[0];
                 d.yield_2y        = data.data()[1];
                 d.yield_10y       = data.data()[2];
                 d.yield_30y       = data.data()[3];
                 d.swap_spread_30y = data.data()[4];
                 d.cds_5y_proxy    = data.data()[5];
                 for (int i = 0; i < 5; ++i) d.returns[i] = rets.data()[i];
                 auto out = self.tick(d);
                 nb::dict result;
                 result["curve_regime"] = out.curve_regime;
                 result["tpm_z"]        = out.tpm_z;
                 result["fso_z"]        = out.fiscal_stress_z;
                 result["signals"]      = out.signals;
                 result["positions"]    = out.positions;
                 return result;
             }, "data"_a, "returns"_a);

    // VSRAEngine ----------------------------------------------------------
    // tick(data[7]) -> dict
    // data = [iv_atm, rv_21d, vix_spot, vix_3m, put_skew, rv_skew, vx_return]
    nb::class_<alpha::strategies::VSRAEngine>(m, "VSRAEngine")
        .def(nb::init<alpha::strategies::VSRAConfig>(),
             "cfg"_a = alpha::strategies::VSRAConfig{})
        .def("tick",
             [](alpha::strategies::VSRAEngine& self, const NpArray1D& data) {
                 alpha::strategies::VSRAEngine::DayData d{};
                 d.iv_atm    = data.data()[0];
                 d.rv_21d    = data.data()[1];
                 d.vix_spot  = data.data()[2];
                 d.vix_3m    = data.data()[3];
                 d.put_skew  = data.data()[4];
                 d.rv_skew   = data.data()[5];
                 d.vx_return = data.data()[6];
                 auto out = self.tick(d);
                 nb::dict result;
                 result["vrp_z"]    = out.vrp_z;
                 result["tss_z"]    = out.tss_z;
                 result["ska_z"]    = out.ska_z;
                 result["signal"]   = out.signal;
                 result["position"] = out.position;
                 return result;
             }, "data"_a);

    // FDSPEngine ----------------------------------------------------------
    // tick(macro[3], returns[5]) -> dict
    // macro = [swap_spread_30y, cds_5y, tbill_spike]
    nb::class_<alpha::strategies::FDSPEngine>(m, "FDSPEngine")
        .def(nb::init<alpha::strategies::FDSPConfig>(),
             "cfg"_a = alpha::strategies::FDSPConfig{})
        .def("tick",
             [](alpha::strategies::FDSPEngine& self,
                const NpArray1D& macro, const NpArray1D& rets) {
                 alpha::strategies::FDSPEngine::DayData d{};
                 d.swap_spread_30y = macro.data()[0];
                 d.cds_5y          = macro.data()[1];
                 d.tbill_spike     = macro.data()[2];
                 for (int i = 0; i < 5; ++i) d.returns[i] = rets.data()[i];
                 auto out = self.tick(d);
                 nb::dict result;
                 result["fci_z"]     = out.fci_z;
                 result["signals"]   = out.signals;
                 result["positions"] = out.positions;
                 return result;
             }, "macro"_a, "returns"_a);

    // Utility functions ---------------------------------------------------
    m.def("sharpe_ratio", [](const NpArray1D& pnl, double ann) {
        return alpha::math::sharpe_ratio(np2vec(pnl), ann);
    }, "pnl"_a, "ann_factor"_a = 252.0);

    m.def("max_drawdown", [](const NpArray1D& cum_pnl) {
        return alpha::math::max_drawdown(np2vec(cum_pnl));
    }, "cum_pnl"_a);

    m.def("half_life_decay", [](const NpArray1D& series) {
        return alpha::math::half_life_decay(np2vec(series));
    }, "series"_a);
}
