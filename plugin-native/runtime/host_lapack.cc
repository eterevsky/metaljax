// metaljax native engine — the host linalg family, on Accelerate's LAPACK.
//
// `src/metaljax/ops/lapack.py`, handler for handler, with numpy/scipy replaced
// by the LAPACK that ships in Accelerate.framework.  Three things carry over
// verbatim because they are XLA's semantics rather than LAPACK's, and a test
// notices each of them:
//
//   * a singular triangular solve divides THROUGH the zero pivot to +-inf/nan
//     instead of failing (jnp.linalg.det's JVP depends on it),
//   * a cholesky of a non-positive-definite matrix is all NaN,
//   * `perturb_singular` nudges a tiny diagonal so the solve stays finite.
//
// ...and one that is this backend's own policy: LAPACK has no half-precision
// routines, so f16/bf16 operands compute in f32 and the results are cast back
// (`_np_in`).  That is why metaljax's halved linalg works where jax's own CPU
// rules reject it outright.
//
// --- LP64, not ILP64 ------------------------------------------------------
//
// Accelerate ships both interfaces: the default takes 32-bit integers
// (`__LAPACK_int` is `int`), and `ACCELERATE_LAPACK_ILP64` selects the 64-bit
// one under `$ILP64`-suffixed symbols.  This file takes the 32-bit one, with
// `ACCELERATE_NEW_LAPACK` for the modern (LAPACK 3.9.1) prototypes rather than
// the frozen 3.2.1 legacy set.  The reason is that ILP64 buys exactly one
// thing — matrix dimensions past 2^31 — which no program that reaches here can
// have: a single f32 matrix of that order is 17 EB of operand, and every
// dimension is checked (`Fit`) so an impossible one is a loud throw and never
// a truncated `int`.  The batch dimensions, which really can be large, are
// loop counters here and never cross into LAPACK.
//
// --- layout ----------------------------------------------------------------
//
// LAPACK is column-major and everything above this line is row-major, so every
// matrix crosses through `ToCol`/`ToRow`.  Explicitly, rather than by flipping
// `uplo`/`trans` flags to make a transpose free: the flag tricks are correct
// only per routine, and the cost here is a memcpy-sized transpose of a matrix
// that is about to be factorized in O(n^3).

#include "host_lapack.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace metaljax {
namespace {

using Int = __LAPACK_int;
using C = std::complex<float>;

// Accelerate's complex prototypes take its own struct; it is layout-identical
// to std::complex<float>, which is what the rest of this file (and MLX) uses.
inline __LAPACK_float_complex* LC(C* p) {
  return reinterpret_cast<__LAPACK_float_complex*>(p);
}

int64_t Numel(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

Int Fit(int64_t n, const char* what) {
  if (n < 0 || n > static_cast<int64_t>(std::numeric_limits<Int>::max()))
    throw std::invalid_argument(std::string("metaljax: LAPACK dimension ") +
                                what + " does not fit a 32-bit integer");
  return static_cast<Int>(n);
}

inline float Conj(float x) { return x; }
inline C Conj(const C& x) { return std::conj(x); }
inline float Abs(float x) { return std::fabs(x); }
inline float Abs(const C& x) { return std::abs(x); }

// row-major (m x n) -> column-major, and back.
template <class T>
void ToCol(const T* row, T* col, int64_t m, int64_t n) {
  for (int64_t i = 0; i < m; i++)
    for (int64_t j = 0; j < n; j++) col[i + j * m] = row[i * n + j];
}

template <class T>
void ToRow(const T* col, T* row, int64_t m, int64_t n) {
  for (int64_t i = 0; i < m; i++)
    for (int64_t j = 0; j < n; j++) row[i * n + j] = col[i + j * m];
}

// --------------------------------------------------------------------------
// the LAPACK routines, overloaded on the element type
// --------------------------------------------------------------------------
//
// One overload pair per routine so every handler below is written once.  The
// workspace queries (`lwork = -1`) and the extra real workspace the complex
// routines want live in the wrappers, because they are the only difference
// between the two arms of most of them.

void Potrf(const char* uplo, Int n, float* a, Int* info) {
  spotrf_(uplo, &n, a, &n, info);
}
void Potrf(const char* uplo, Int n, C* a, Int* info) {
  cpotrf_(uplo, &n, LC(a), &n, info);
}

void Geqrf(Int m, Int n, float* a, float* tau, Int* info) {
  Int lwork = -1;
  float q = 0;
  sgeqrf_(&m, &n, a, &m, tau, &q, &lwork, info);
  lwork = std::max<Int>(1, static_cast<Int>(q));
  std::vector<float> work(lwork);
  sgeqrf_(&m, &n, a, &m, tau, work.data(), &lwork, info);
}
void Geqrf(Int m, Int n, C* a, C* tau, Int* info) {
  Int lwork = -1;
  C q = 0;
  cgeqrf_(&m, &n, LC(a), &m, LC(tau), LC(&q), &lwork, info);
  lwork = std::max<Int>(1, static_cast<Int>(q.real()));
  std::vector<C> work(lwork);
  cgeqrf_(&m, &n, LC(a), &m, LC(tau), LC(work.data()), &lwork, info);
}

void Orgqr(Int m, Int n, Int k, float* a, float* tau, Int* info) {
  Int lwork = -1;
  float q = 0;
  sorgqr_(&m, &n, &k, a, &m, tau, &q, &lwork, info);
  lwork = std::max<Int>(1, static_cast<Int>(q));
  std::vector<float> work(lwork);
  sorgqr_(&m, &n, &k, a, &m, tau, work.data(), &lwork, info);
}
void Orgqr(Int m, Int n, Int k, C* a, C* tau, Int* info) {
  Int lwork = -1;
  C q = 0;
  cungqr_(&m, &n, &k, LC(a), &m, LC(tau), LC(&q), &lwork, info);
  lwork = std::max<Int>(1, static_cast<Int>(q.real()));
  std::vector<C> work(lwork);
  cungqr_(&m, &n, &k, LC(a), &m, LC(tau), LC(work.data()), &lwork, info);
}

void Evd(const char* uplo, Int n, float* a, float* w, Int* info) {
  Int lwork = -1, liwork = -1, iq = 0;
  float q = 0;
  ssyevd_("V", uplo, &n, a, &n, w, &q, &lwork, &iq, &liwork, info);
  lwork = std::max<Int>(1, static_cast<Int>(q));
  liwork = std::max<Int>(1, iq);
  std::vector<float> work(lwork);
  std::vector<Int> iwork(liwork);
  ssyevd_("V", uplo, &n, a, &n, w, work.data(), &lwork, iwork.data(), &liwork,
          info);
}
void Evd(const char* uplo, Int n, C* a, float* w, Int* info) {
  Int lwork = -1, lrwork = -1, liwork = -1, iq = 0;
  C q = 0;
  float rq = 0;
  cheevd_("V", uplo, &n, LC(a), &n, w, LC(&q), &lwork, &rq, &lrwork, &iq,
          &liwork, info);
  lwork = std::max<Int>(1, static_cast<Int>(q.real()));
  lrwork = std::max<Int>(1, static_cast<Int>(rq));
  liwork = std::max<Int>(1, iq);
  std::vector<C> work(lwork);
  std::vector<float> rwork(lrwork);
  std::vector<Int> iwork(liwork);
  cheevd_("V", uplo, &n, LC(a), &n, w, LC(work.data()), &lwork, rwork.data(),
          &lrwork, iwork.data(), &liwork, info);
}

void Gesdd(const char* jobz, Int m, Int n, float* a, float* s, float* u,
           Int ldu, float* vt, Int ldvt, Int* info) {
  const Int mn = std::min(m, n);
  Int lwork = -1;
  float q = 0;
  std::vector<Int> iwork(std::max<Int>(1, 8 * mn));
  sgesdd_(jobz, &m, &n, a, &m, s, u, &ldu, vt, &ldvt, &q, &lwork, iwork.data(),
          info);
  lwork = std::max<Int>(1, static_cast<Int>(q));
  std::vector<float> work(lwork);
  sgesdd_(jobz, &m, &n, a, &m, s, u, &ldu, vt, &ldvt, work.data(), &lwork,
          iwork.data(), info);
}
void Gesdd(const char* jobz, Int m, Int n, C* a, float* s, C* u, Int ldu,
           C* vt, Int ldvt, Int* info) {
  const Int mn = std::min(m, n), mx = std::max(m, n);
  // cgesdd's real workspace is the one size LAPACK will not tell you: the
  // query returns work[0] and nothing else, so the documented bound is
  // computed here (7*min for jobz='N', the larger expression otherwise).
  const Int lrwork =
      *jobz == 'N'
          ? std::max<Int>(1, 7 * mn)
          : std::max<Int>(1, mn * std::max(5 * mn + 7, 2 * mx + 2 * mn + 1));
  std::vector<float> rwork(lrwork);
  std::vector<Int> iwork(std::max<Int>(1, 8 * mn));
  Int lwork = -1;
  C q = 0;
  cgesdd_(jobz, &m, &n, LC(a), &m, s, LC(u), &ldu, LC(vt), &ldvt, LC(&q),
          &lwork, rwork.data(), iwork.data(), info);
  lwork = std::max<Int>(1, static_cast<Int>(q.real()));
  std::vector<C> work(lwork);
  cgesdd_(jobz, &m, &n, LC(a), &m, s, LC(u), &ldu, LC(vt), &ldvt,
          LC(work.data()), &lwork, rwork.data(), iwork.data(), info);
}

void Geev(const char* jobvl, const char* jobvr, Int n, float* a, float* wr,
          float* wi, float* vl, float* vr, Int* info) {
  Int lwork = -1;
  float q = 0;
  sgeev_(jobvl, jobvr, &n, a, &n, wr, wi, vl, &n, vr, &n, &q, &lwork, info);
  lwork = std::max<Int>(1, static_cast<Int>(q));
  std::vector<float> work(lwork);
  sgeev_(jobvl, jobvr, &n, a, &n, wr, wi, vl, &n, vr, &n, work.data(), &lwork,
         info);
}
void Geev(const char* jobvl, const char* jobvr, Int n, C* a, C* w, C* vl,
          C* vr, Int* info) {
  Int lwork = -1;
  C q = 0;
  std::vector<float> rwork(std::max<Int>(1, 2 * n));
  cgeev_(jobvl, jobvr, &n, LC(a), &n, LC(w), LC(vl), &n, LC(vr), &n, LC(&q),
         &lwork, rwork.data(), info);
  lwork = std::max<Int>(1, static_cast<Int>(q.real()));
  std::vector<C> work(lwork);
  cgeev_(jobvl, jobvr, &n, LC(a), &n, LC(w), LC(vl), &n, LC(vr), &n,
         LC(work.data()), &lwork, rwork.data(), info);
}

void Gees(Int n, float* a, float* vs, Int* info) {
  Int lwork = -1, sdim = 0;
  float q = 0;
  std::vector<float> wr(std::max<Int>(1, n)), wi(std::max<Int>(1, n));
  std::vector<__LAPACK_bool> bwork(std::max<Int>(1, n));
  sgees_("V", "N", nullptr, &n, a, &n, &sdim, wr.data(), wi.data(), vs, &n, &q,
         &lwork, bwork.data(), info);
  lwork = std::max<Int>(1, static_cast<Int>(q));
  std::vector<float> work(lwork);
  sgees_("V", "N", nullptr, &n, a, &n, &sdim, wr.data(), wi.data(), vs, &n,
         work.data(), &lwork, bwork.data(), info);
}
void Gees(Int n, C* a, C* vs, Int* info) {
  Int lwork = -1, sdim = 0;
  C q = 0;
  std::vector<C> w(std::max<Int>(1, n));
  std::vector<float> rwork(std::max<Int>(1, n));
  std::vector<__LAPACK_bool> bwork(std::max<Int>(1, n));
  cgees_("V", "N", nullptr, &n, LC(a), &n, &sdim, LC(w.data()), LC(vs), &n,
         LC(&q), &lwork, rwork.data(), bwork.data(), info);
  lwork = std::max<Int>(1, static_cast<Int>(q.real()));
  std::vector<C> work(lwork);
  cgees_("V", "N", nullptr, &n, LC(a), &n, &sdim, LC(w.data()), LC(vs), &n,
         LC(work.data()), &lwork, rwork.data(), bwork.data(), info);
}

void Gehrd(Int n, float* a, float* tau, Int* info) {
  Int ilo = 1, ihi = n, lwork = -1;
  float q = 0;
  sgehrd_(&n, &ilo, &ihi, a, &n, tau, &q, &lwork, info);
  lwork = std::max<Int>(1, static_cast<Int>(q));
  std::vector<float> work(lwork);
  sgehrd_(&n, &ilo, &ihi, a, &n, tau, work.data(), &lwork, info);
}
void Gehrd(Int n, C* a, C* tau, Int* info) {
  Int ilo = 1, ihi = n, lwork = -1;
  C q = 0;
  cgehrd_(&n, &ilo, &ihi, LC(a), &n, LC(tau), LC(&q), &lwork, info);
  lwork = std::max<Int>(1, static_cast<Int>(q.real()));
  std::vector<C> work(lwork);
  cgehrd_(&n, &ilo, &ihi, LC(a), &n, LC(tau), LC(work.data()), &lwork, info);
}

void Sytrd(const char* uplo, Int n, float* a, float* d, float* e, float* tau,
           Int* info) {
  Int lwork = -1;
  float q = 0;
  ssytrd_(uplo, &n, a, &n, d, e, tau, &q, &lwork, info);
  lwork = std::max<Int>(1, static_cast<Int>(q));
  std::vector<float> work(lwork);
  ssytrd_(uplo, &n, a, &n, d, e, tau, work.data(), &lwork, info);
}
void Sytrd(const char* uplo, Int n, C* a, float* d, float* e, C* tau,
           Int* info) {
  Int lwork = -1;
  C q = 0;
  chetrd_(uplo, &n, LC(a), &n, d, e, LC(tau), LC(&q), &lwork, info);
  lwork = std::max<Int>(1, static_cast<Int>(q.real()));
  std::vector<C> work(lwork);
  chetrd_(uplo, &n, LC(a), &n, d, e, LC(tau), LC(work.data()), &lwork, info);
}

void Getrf(Int m, Int n, float* a, Int* ipiv, Int* info) {
  sgetrf_(&m, &n, a, &m, ipiv, info);
}
void Getrf(Int m, Int n, C* a, Int* ipiv, Int* info) {
  cgetrf_(&m, &n, LC(a), &m, ipiv, info);
}

void Getrs(Int n, Int nrhs, float* a, Int* ipiv, float* b, Int* info) {
  sgetrs_("N", &n, &nrhs, a, &n, ipiv, b, &n, info);
}
void Getrs(Int n, Int nrhs, C* a, Int* ipiv, C* b, Int* info) {
  cgetrs_("N", &n, &nrhs, LC(a), &n, ipiv, LC(b), &n, info);
}

// CBLAS takes the row-major layout directly, so the triangular solve is the
// one routine that needs no transpose.
void Trsm(CBLAS_UPLO uplo, CBLAS_DIAG diag, Int n, Int nrhs, const float* a,
          float* b) {
  cblas_strsm(CblasRowMajor, CblasLeft, uplo, CblasNoTrans, diag, n, nrhs, 1.0f,
              a, n, b, nrhs);
}
void Trsm(CBLAS_UPLO uplo, CBLAS_DIAG diag, Int n, Int nrhs, const C* a,
          C* b) {
  const C one(1.0f, 0.0f);
  cblas_ctrsm(CblasRowMajor, CblasLeft, uplo, CblasNoTrans, diag, n, nrhs, &one,
              a, n, b, nrhs);
}

// --------------------------------------------------------------------------
// host buffers
// --------------------------------------------------------------------------

// One operand, settled and copied to the host in the COMPUTE type.
// `_np_in`: halves compute in f32, so the only two compute types are f32 and
// complex64 and every operand of one call shares one of them.
template <class T>
struct Operand {
  std::vector<T> data;
  std::vector<int64_t> shape;
  int64_t per_item = 0;
  int64_t per() const { return per_item; }   // elements per batch item
  const T* at(int64_t b) const { return data.data() + b * per_item; }
  int64_t dim(int64_t i) const {
    const int64_t r = static_cast<int64_t>(shape.size());
    return i < 0 ? shape[static_cast<size_t>(r + i)]
                 : shape[static_cast<size_t>(i)];
  }
};

template <class T>
Operand<T> Read(const mx::array& x, mx::Dtype compute, int64_t batch) {
  Operand<T> out;
  mx::array c = mx::contiguous(mx::astype(x, compute));
  c.eval();
  out.data.resize(static_cast<size_t>(c.size()));
  if (c.size() > 0)
    std::memcpy(out.data.data(), c.data<T>(), out.data.size() * sizeof(T));
  for (auto d : c.shape()) out.shape.push_back(static_cast<int64_t>(d));
  out.per_item = batch > 0 ? static_cast<int64_t>(c.size()) / batch : 0;
  return out;
}

// One declared result, accumulated on the host.  The buffer's element type is
// the compute type of the DECLARED dtype -- f32 for f32/f16/bf16, complex64
// for complex, int32 for the integer results LU's pivots are -- and `Finish`
// casts it back, which is `_batch_apply`'s trailing `.astype(spec[1])`.
class Out {
 public:
  Out(const HostSpec& spec, int64_t batch) : spec_(spec) {
    total_ = Numel(spec.shape);
    per_ = batch > 0 ? total_ / batch : 0;
    if (is_complex(spec.dtype)) {
      c_.assign(static_cast<size_t>(total_), C(0.0f, 0.0f));
    } else if (is_float(spec.dtype)) {
      f_.assign(static_cast<size_t>(total_), 0.0f);
    } else {
      i_.assign(static_cast<size_t>(total_), 0);
    }
  }

  float* f(int64_t b) { return f_.data() + b * per_; }
  C* c(int64_t b) { return c_.data() + b * per_; }
  int32_t* i(int64_t b) { return i_.data() + b * per_; }
  int64_t per() const { return per_; }

  // Fill one batch item with a quiet NaN (an integer result stays at zero):
  // what `_metal_eigh`/`_metal_svd` do for a non-finite operand and
  // `_cholesky` for a factorization that fails.
  void PoisonItem(int64_t b) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int64_t k = 0; k < per_; k++) {
      if (!f_.empty()) f_[b * per_ + k] = nan;
      if (!c_.empty()) c_[b * per_ + k] = C(nan, nan);
    }
  }

  // The staging block becomes the array's own storage -- one allocation, one
  // copy, freed by MLX when the array dies, which is the ingest path's
  // arrangement (metal_client.cc `BufferFromHostBuffer`).  Zero-size never
  // touches a pointer: MLX 0.32 hands out null-backed empty buffers.
  mx::array Finish() const {
    mx::Shape shp;
    for (int64_t d : spec_.shape) shp.push_back(static_cast<mx::ShapeElem>(d));
    if (total_ == 0) return mx::zeros(shp, spec_.dtype);
    const mx::Dtype src =
        !c_.empty() ? mx::complex64 : (!f_.empty() ? mx::float32 : mx::int32);
    const void* from = !c_.empty() ? static_cast<const void*>(c_.data())
                                   : (!f_.empty()
                                          ? static_cast<const void*>(f_.data())
                                          : static_cast<const void*>(i_.data()));
    const size_t nbytes = static_cast<size_t>(total_) * src.size();
    void* stage = std::malloc(nbytes);
    if (stage == nullptr)
      throw std::runtime_error("metaljax: host linalg could not stage a result");
    std::memcpy(stage, from, nbytes);
    return mx::astype(
        mx::array(stage, shp, src, [](void* p) { std::free(p); }), spec_.dtype);
  }

 private:
  HostSpec spec_;
  int64_t total_ = 0;
  int64_t per_ = 0;
  std::vector<float> f_;
  std::vector<C> c_;
  std::vector<int32_t> i_;
};

std::vector<Out> MakeOuts(const HostLinalgCall& call, int64_t batch) {
  std::vector<Out> outs;
  outs.reserve(call.results.size());
  for (const HostSpec& s : call.results) outs.emplace_back(s, batch);
  return outs;
}

std::vector<mx::array> Finish(const std::vector<Out>& outs) {
  std::vector<mx::array> res;
  res.reserve(outs.size());
  for (const Out& o : outs) res.push_back(o.Finish());
  return res;
}

// The batch shape a call loops over: the operand's leading dimensions, all but
// the trailing `keep`.  Flattened, because every handler indexes items.
int64_t FlatBatch(const mx::array& x, int keep) {
  int64_t n = 1;
  const auto& s = x.shape();
  for (int i = 0; i + keep < static_cast<int>(s.size()); i++) n *= s[i];
  return n;
}

// A declared result's trailing dimension (`back` counts from the end, -1 being
// the last).  Read off the SPEC rather than divided out of the buffer size, so
// it stays right for a zero-size matrix.
int64_t Dim(const HostSpec& s, int back) {
  const int64_t r = static_cast<int64_t>(s.shape.size());
  return r + back >= 0 ? s.shape[static_cast<size_t>(r + back)] : 0;
}

// Item pointer for a result whose compute type is T -- the two arms differ
// only in which of `Out`'s buffers they name.
template <class T>
T* ItemOf(Out& o, int64_t b) {
  if constexpr (std::is_same_v<T, C>) {
    return o.c(b);
  } else {
    return o.f(b);
  }
}

// --------------------------------------------------------------------------
// the handlers
// --------------------------------------------------------------------------

template <class T>
bool AllFinite(const T* p, int64_t n);

template <>
bool AllFinite<float>(const float* p, int64_t n) {
  for (int64_t i = 0; i < n; i++)
    if (!std::isfinite(p[i])) return false;
  return true;
}

template <>
bool AllFinite<C>(const C* p, int64_t n) {
  for (int64_t i = 0; i < n; i++)
    if (!std::isfinite(p[i].real()) || !std::isfinite(p[i].imag())) return false;
  return true;
}

// `stablehlo.cholesky`.  potrf reads the triangle `uplo` names and leaves the
// other one untouched, so it is zeroed here -- numpy's cholesky does the same,
// and XLA's result is triangular.  A factorization that fails is all NaN
// (`np.linalg.LinAlgError` in the Python handler).
template <class T>
void RunCholesky(const HostLinalgCall& call, const Operand<T>& a,
                 int64_t batch, std::vector<Out>& outs) {
  const int64_t n = a.dim(-1);
  const Int ln = Fit(n, "cholesky n");
  std::vector<T> col(static_cast<size_t>(n * n));
  std::vector<T> row(static_cast<size_t>(n * n));
  for (int64_t b = 0; b < batch; b++) {
    ToCol(a.at(b), col.data(), n, n);
    Int info = 0;
    if (n > 0) Potrf(call.lower ? "L" : "U", ln, col.data(), &info);
    if (info != 0) {
      outs[0].PoisonItem(b);
      continue;
    }
    for (int64_t i = 0; i < n; i++)
      for (int64_t j = 0; j < n; j++)
        if (call.lower ? (j > i) : (j < i)) col[i + j * n] = T(0);
    ToRow(col.data(), row.data(), n, n);
    std::memcpy(ItemOf<T>(outs[0], b), row.data(),
                static_cast<size_t>(n * n) * sizeof(T));
  }
}

// @Qr / lapack_?geqrf_ffi: geqrf, whose results are the packed factorization
// and the reflector scalars.
template <class T>
void RunQr(const Operand<T>& a, int64_t batch, std::vector<Out>& outs) {
  const int64_t m = a.dim(-2), n = a.dim(-1), k = std::min(m, n);
  const Int lm = Fit(m, "qr m"), ln = Fit(n, "qr n");
  std::vector<T> col(static_cast<size_t>(m * n));
  std::vector<T> row(static_cast<size_t>(m * n));
  std::vector<T> tau(static_cast<size_t>(std::max<int64_t>(k, 1)), T(0));
  for (int64_t b = 0; b < batch; b++) {
    ToCol(a.at(b), col.data(), m, n);
    Int info = 0;
    if (m > 0 && n > 0) Geqrf(lm, ln, col.data(), tau.data(), &info);
    ToRow(col.data(), row.data(), m, n);
    std::memcpy(ItemOf<T>(outs[0], b), row.data(), row.size() * sizeof(T));
    if (outs.size() > 1 && outs[1].per() > 0)
      std::memcpy(ItemOf<T>(outs[1], b), tau.data(),
                  static_cast<size_t>(outs[1].per()) * sizeof(T));
  }
}

// @ProductOfElementaryHouseholderReflectors / lapack_?orgqr_ffi.
//
// The one place a completion is asked for: `full_matrices` QR wants MORE
// columns of Q than there are reflectors, and LAPACK's orgqr completes an
// orthonormal basis when it is given the room -- a zero tau is an identity
// reflector, so padding the matrix with zero columns and the taus with zeros
// is exactly the completion (`_householder_product`).
template <class T>
void RunOrgqr(const HostLinalgCall& call, const Operand<T>& a,
              const Operand<T>& taus, int64_t batch, std::vector<Out>& outs) {
  const int64_t m = a.dim(-2), acols = a.dim(-1);
  const int64_t k = taus.per();
  const int64_t ncols = Dim(call.results[0], -1);
  const int64_t cols = std::max<int64_t>(k, 1);
  // The completion case hands LAPACK `ncols` reflectors, of which the last
  // `ncols - cols` have a zero tau; the reduced one hands it `k` and slices.
  const int64_t want = ncols > cols ? ncols : cols;
  const int64_t klap = ncols > cols ? ncols : k;
  const int64_t src = std::min(cols, acols);        // columns that come from `a`
  const Int lm = Fit(m, "orgqr m"), ln = Fit(want, "orgqr n"),
            lk = Fit(klap, "orgqr k");
  const size_t area = static_cast<size_t>(std::max<int64_t>(m * want, 1));
  std::vector<T> col(area, T(0));
  std::vector<T> tau(static_cast<size_t>(std::max<int64_t>(want, 1)), T(0));
  std::vector<T> row(area, T(0));
  for (int64_t b = 0; b < batch; b++) {
    std::fill(col.begin(), col.end(), T(0));
    std::fill(tau.begin(), tau.end(), T(0));
    const T* ap = a.at(b);
    for (int64_t i = 0; i < m; i++)
      for (int64_t j = 0; j < src; j++) col[i + j * m] = ap[i * acols + j];
    for (int64_t j = 0; j < k; j++) tau[j] = taus.at(b)[j];
    Int info = 0;
    if (m > 0 && want > 0 && want <= m)
      Orgqr(lm, ln, lk, col.data(), tau.data(), &info);
    ToRow(col.data(), row.data(), m, want);
    // Down to the declared width (the reduced case asks for fewer columns
    // than LAPACK was given).
    T* dst = ItemOf<T>(outs[0], b);
    for (int64_t i = 0; i < m; i++)
      for (int64_t j = 0; j < ncols; j++) dst[i * ncols + j] = row[i * want + j];
  }
}

// metaljax_eigh / lapack_?syevd_ffi / @Eigh -- syevd, heevd.
//
// No explicit symmetrization: `uplo` is what tells LAPACK which triangle to
// read, and after `ToCol` the column-major buffer IS the operand, so "L" reads
// the operand's lower triangle -- which is what
// `np.tril(x) + np.tril(x, -1).conj().T` built for numpy.
template <class T>
void RunEigh(const HostLinalgCall& call, const Operand<T>& a, int64_t batch,
             std::vector<Out>& outs) {
  const bool guard = call.kind == HostLinalg::kEigh;   // metaljax_eigh only
  const int64_t n = a.dim(-1);
  const Int ln = Fit(n, "eigh n");
  std::vector<T> col(static_cast<size_t>(n * n));
  std::vector<T> row(static_cast<size_t>(n * n));
  std::vector<float> w(static_cast<size_t>(std::max<int64_t>(n, 1)), 0.0f);
  for (int64_t b = 0; b < batch; b++) {
    if (guard && !AllFinite<T>(a.at(b), a.per())) {
      for (Out& o : outs) o.PoisonItem(b);
      continue;
    }
    ToCol(a.at(b), col.data(), n, n);
    Int info = 0;
    if (n > 0) Evd(call.lower ? "L" : "U", ln, col.data(), w.data(), &info);
    if (info != 0) {
      for (Out& o : outs) o.PoisonItem(b);
      continue;
    }
    ToRow(col.data(), row.data(), n, n);
    std::memcpy(ItemOf<T>(outs[0], b), row.data(), row.size() * sizeof(T));
    std::memcpy(outs[1].f(b), w.data(), static_cast<size_t>(n) * sizeof(float));
    // The FFI convention's trailing `info` (and anything else it declares)
    // stays at the zeros `Out` starts from.
  }
}

// metaljax_svd / lapack_?gesdd_ffi -- gesdd.  `first` is where the (s, u, vt)
// triple starts: 0 for our own target, 1 for the FFI one, whose first result
// is the workspace copy of the operand.
template <class T>
void RunSvd(const HostLinalgCall& call, const Operand<T>& a, int64_t batch,
            std::vector<Out>& outs, size_t first) {
  const int64_t m = a.dim(-2), n = a.dim(-1), k = std::min(m, n);
  const bool uv = outs.size() > first + 1;
  const int64_t ucols = uv ? Dim(call.results[first + 1], -1) : 0;
  const int64_t vtrows = uv ? Dim(call.results[first + 2], -2) : 0;
  const bool full = uv && ucols == m && vtrows == n;
  const char* jobz = !uv ? "N" : (full ? "A" : "S");
  const int64_t uw = !uv ? 1 : (full ? m : k);      // widths LAPACK writes
  const int64_t vh = !uv ? 1 : (full ? n : k);
  const Int lm = Fit(m, "svd m"), ln = Fit(n, "svd n");
  std::vector<T> col(static_cast<size_t>(std::max<int64_t>(m * n, 1)));
  std::vector<T> u(static_cast<size_t>(std::max<int64_t>(m * uw, 1)), T(0));
  std::vector<T> vt(static_cast<size_t>(std::max<int64_t>(vh * n, 1)), T(0));
  std::vector<T> row(static_cast<size_t>(
      std::max<int64_t>(std::max(m * uw, vh * n), 1)));
  std::vector<float> s(static_cast<size_t>(std::max<int64_t>(k, 1)), 0.0f);
  for (int64_t b = 0; b < batch; b++) {
    if (call.kind == HostLinalg::kSvd &&
        !AllFinite<T>(a.at(b), a.per())) {   // metaljax_svd's guard
      for (Out& o : outs) o.PoisonItem(b);
      continue;
    }
    if (first == 1)   // the FFI's workspace copy is the operand, unchanged
      std::memcpy(ItemOf<T>(outs[0], b), a.at(b),
                  static_cast<size_t>(a.per()) * sizeof(T));
    ToCol(a.at(b), col.data(), m, n);
    std::fill(u.begin(), u.end(), T(0));
    std::fill(vt.begin(), vt.end(), T(0));
    Int info = 0;
    if (m > 0 && n > 0) {
      Gesdd(jobz, lm, ln, col.data(), s.data(), u.data(),
            Fit(std::max<int64_t>(m, 1), "svd ldu"), vt.data(),
            Fit(std::max<int64_t>(vh, 1), "svd ldvt"), &info);
    } else if (uv) {
      // An empty operand still has orthogonal factors: numpy returns the
      // identity, LAPACK returns immediately and leaves the buffers alone.
      for (int64_t i = 0; i < uw && i < m; i++) u[i + i * m] = T(1);
      for (int64_t i = 0; i < vh && i < n; i++) vt[i + i * vh] = T(1);
    }
    if (info != 0) {
      for (Out& o : outs) o.PoisonItem(b);
      continue;
    }
    std::memcpy(outs[first].f(b), s.data(),
                static_cast<size_t>(k) * sizeof(float));
    if (!uv) continue;
    ToRow(u.data(), row.data(), m, uw);
    T* du = ItemOf<T>(outs[first + 1], b);
    for (int64_t i = 0; i < m; i++)
      for (int64_t j = 0; j < ucols; j++) du[i * ucols + j] = row[i * uw + j];
    ToRow(vt.data(), row.data(), vh, n);
    T* dv = ItemOf<T>(outs[first + 2], b);
    std::memcpy(dv, row.data(), static_cast<size_t>(vtrows * n) * sizeof(T));
  }
}

// metaljax_eig / lapack_?geev_ffi -- geev.  The eigenvalues and eigenvectors
// of a general matrix are COMPLEX whatever the operand is, which is why the
// real arm unpacks LAPACK's conjugate-pair packing (columns j and j+1 of vr
// hold the real and imaginary parts of one pair).
//
// Deliberate divergence from `_metal_eig`, which had numpy's constraint that
// this file does not: the left eigenvectors come from geev's own `jobvl`
// rather than from an eigendecomposition of the adjoint whose columns are then
// matched to conj(w) by nearest eigenvalue.  Same vectors, same order as w,
// and the same normalization jax's CPU backend hands back.
template <class T>
void RunEig(const HostLinalgCall& call, const Operand<T>& a, int64_t batch,
            std::vector<Out>& outs, bool ffi) {
  const int64_t n = a.dim(-1);
  const Int ln = Fit(n, "eig n");
  const bool split = ffi && outs.size() == 5;     // (wr, wi, vl, vr, info)
  // Which results hold what.  Our own target lists w, then the vectors the
  // config asked for; the FFI one always lists both, and jax consumes only
  // the right ones (`compute_left = 'N'`), so vl stays zero there.
  const bool want_left = ffi ? false : call.left_vectors;
  const bool want_right = ffi ? true : call.right_vectors;
  const size_t iw = 0;
  const size_t ivl = split ? 2 : 1;
  const size_t ivr = ffi ? ivl + 1 : (call.left_vectors ? 2 : 1);

  std::vector<T> col(static_cast<size_t>(std::max<int64_t>(n * n, 1)));
  std::vector<T> vl(static_cast<size_t>(std::max<int64_t>(n * n, 1)), T(0));
  std::vector<T> vr(static_cast<size_t>(std::max<int64_t>(n * n, 1)), T(0));
  std::vector<float> wr(static_cast<size_t>(std::max<int64_t>(n, 1)), 0.0f);
  std::vector<float> wi(static_cast<size_t>(std::max<int64_t>(n, 1)), 0.0f);
  std::vector<C> w(static_cast<size_t>(std::max<int64_t>(n, 1)), C(0, 0));

  // One packed geev result -> the complex eigenvector matrix, row-major.
  auto unpack = [&](const std::vector<T>& v, C* dst) {
    if constexpr (std::is_same_v<T, C>) {
      for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++) dst[i * n + j] = v[i + j * n];
    } else {
      for (int64_t j = 0; j < n;) {
        if (wi[j] == 0.0f) {
          for (int64_t i = 0; i < n; i++) dst[i * n + j] = C(v[i + j * n], 0);
          j++;
        } else {
          for (int64_t i = 0; i < n; i++) {
            const C z(v[i + j * n], v[i + (j + 1) * n]);
            dst[i * n + j] = z;
            dst[i * n + j + 1] = std::conj(z);
          }
          j += 2;
        }
      }
    }
  };

  for (int64_t b = 0; b < batch; b++) {
    if (n == 0) continue;    // `_metal_eig`'s zero guard: everything is empty
    ToCol(a.at(b), col.data(), n, n);
    Int info = 0;
    if constexpr (std::is_same_v<T, C>) {
      Geev(want_left ? "V" : "N", want_right ? "V" : "N", ln, col.data(),
           w.data(), vl.data(), vr.data(), &info);
    } else {
      Geev(want_left ? "V" : "N", want_right ? "V" : "N", ln, col.data(),
           wr.data(), wi.data(), vl.data(), vr.data(), &info);
      for (int64_t i = 0; i < n; i++) w[i] = C(wr[i], wi[i]);
    }
    if (info != 0) {
      for (Out& o : outs) o.PoisonItem(b);
      continue;
    }
    if (split) {
      for (int64_t i = 0; i < n; i++) {
        outs[0].f(b)[i] = w[i].real();
        outs[1].f(b)[i] = w[i].imag();
      }
    } else {
      std::memcpy(outs[iw].c(b), w.data(), static_cast<size_t>(n) * sizeof(C));
    }
    if (want_left && ivl < outs.size()) unpack(vl, outs[ivl].c(b));
    if (want_right && ivr < outs.size()) unpack(vr, outs[ivr].c(b));
  }
}

// metaljax_lu -- getrf, plus the permutation jax derives from its pivots.
// LAPACK's ipiv is ONE-based; jax (like scipy's wrapper, which the Python
// handler used) wants it zero-based, and the permutation is the swap sweep.
template <class T>
void RunLu(const Operand<T>& a, int64_t batch, std::vector<Out>& outs) {
  const int64_t m = a.dim(-2), n = a.dim(-1), k = std::min(m, n);
  const Int lm = Fit(m, "lu m"), ln = Fit(n, "lu n");
  std::vector<T> col(static_cast<size_t>(std::max<int64_t>(m * n, 1)));
  std::vector<T> row(static_cast<size_t>(std::max<int64_t>(m * n, 1)));
  std::vector<Int> ipiv(static_cast<size_t>(std::max<int64_t>(k, 1)), 0);
  for (int64_t b = 0; b < batch; b++) {
    Int info = 0;
    if (k == 0) {
      std::memcpy(ItemOf<T>(outs[0], b), a.at(b),
                  static_cast<size_t>(a.per()) * sizeof(T));
    } else {
      ToCol(a.at(b), col.data(), m, n);
      Getrf(lm, ln, col.data(), ipiv.data(), &info);
      // info > 0 is a singular factor, which is still a valid factorization
      // (XLA returns it too); only info < 0 -- an illegal argument -- poisons
      // the result.
      if (info < 0) {
        outs[0].PoisonItem(b);
      } else {
        ToRow(col.data(), row.data(), m, n);
        std::memcpy(ItemOf<T>(outs[0], b), row.data(),
                    static_cast<size_t>(m * n) * sizeof(T));
      }
    }
    if (outs.size() > 1)
      for (int64_t i = 0; i < k; i++)
        outs[1].i(b)[i] = static_cast<int32_t>(ipiv[i] - 1);
    if (outs.size() > 2) {
      int32_t* perm = outs[2].i(b);
      for (int64_t i = 0; i < m; i++) perm[i] = static_cast<int32_t>(i);
      for (int64_t i = 0; i < k; i++) {
        const int64_t j = static_cast<int64_t>(ipiv[i]) - 1;
        if (j >= 0 && j < m) std::swap(perm[i], perm[j]);
      }
    }
  }
}

// metaljax_schur -- gees with jobvs='V', sort='N': (T, Z), real Schur form for
// a real operand and the complex one otherwise, exactly as scipy's `schur`
// picks its `output=` from the operand.
template <class T>
void RunSchur(const Operand<T>& a, int64_t batch, std::vector<Out>& outs) {
  const int64_t n = a.dim(-1);
  const Int ln = Fit(n, "schur n");
  std::vector<T> col(static_cast<size_t>(std::max<int64_t>(n * n, 1)));
  std::vector<T> vs(static_cast<size_t>(std::max<int64_t>(n * n, 1)), T(0));
  std::vector<T> row(static_cast<size_t>(std::max<int64_t>(n * n, 1)));
  for (int64_t b = 0; b < batch; b++) {
    ToCol(a.at(b), col.data(), n, n);
    Int info = 0;
    if (n > 0) Gees(ln, col.data(), vs.data(), &info);
    if (info != 0) {
      for (Out& o : outs) o.PoisonItem(b);
      continue;
    }
    ToRow(col.data(), row.data(), n, n);
    std::memcpy(ItemOf<T>(outs[0], b), row.data(),
                static_cast<size_t>(n * n) * sizeof(T));
    if (outs.size() > 1) {
      ToRow(vs.data(), row.data(), n, n);
      std::memcpy(ItemOf<T>(outs[1], b), row.data(),
                  static_cast<size_t>(n * n) * sizeof(T));
    }
  }
}

// metaljax_hessenberg -- gehrd over the whole matrix (ilo = 1, ihi = n), whose
// results are the packed form and the reflector scalars.
template <class T>
void RunHessenberg(const Operand<T>& a, int64_t batch, std::vector<Out>& outs) {
  const int64_t n = a.dim(-1);
  const Int ln = Fit(n, "hessenberg n");
  std::vector<T> col(static_cast<size_t>(std::max<int64_t>(n * n, 1)));
  std::vector<T> row(static_cast<size_t>(std::max<int64_t>(n * n, 1)));
  std::vector<T> tau(static_cast<size_t>(std::max<int64_t>(n, 1)), T(0));
  for (int64_t b = 0; b < batch; b++) {
    ToCol(a.at(b), col.data(), n, n);
    Int info = 0;
    if (n > 0) Gehrd(ln, col.data(), tau.data(), &info);
    ToRow(col.data(), row.data(), n, n);
    std::memcpy(ItemOf<T>(outs[0], b), row.data(),
                static_cast<size_t>(n * n) * sizeof(T));
    if (outs.size() > 1 && outs[1].per() > 0)
      std::memcpy(ItemOf<T>(outs[1], b), tau.data(),
                  static_cast<size_t>(outs[1].per()) * sizeof(T));
  }
}

// metaljax_tridiagonal -- sytrd / hetrd: (packed form, diagonal, off-diagonal,
// reflector scalars).
template <class T>
void RunTridiagonal(const HostLinalgCall& call, const Operand<T>& a,
                    int64_t batch, std::vector<Out>& outs) {
  const int64_t n = a.dim(-1);
  const Int ln = Fit(n, "tridiagonal n");
  std::vector<T> col(static_cast<size_t>(std::max<int64_t>(n * n, 1)));
  std::vector<T> row(static_cast<size_t>(std::max<int64_t>(n * n, 1)));
  std::vector<float> d(static_cast<size_t>(std::max<int64_t>(n, 1)), 0.0f);
  std::vector<float> e(static_cast<size_t>(std::max<int64_t>(n, 1)), 0.0f);
  std::vector<T> tau(static_cast<size_t>(std::max<int64_t>(n, 1)), T(0));
  for (int64_t b = 0; b < batch; b++) {
    ToCol(a.at(b), col.data(), n, n);
    Int info = 0;
    if (n > 0)
      Sytrd(call.lower ? "L" : "U", ln, col.data(), d.data(), e.data(),
            tau.data(), &info);
    ToRow(col.data(), row.data(), n, n);
    std::memcpy(ItemOf<T>(outs[0], b), row.data(),
                static_cast<size_t>(n * n) * sizeof(T));
    if (outs.size() > 1 && outs[1].per() > 0)
      std::memcpy(outs[1].f(b), d.data(),
                  static_cast<size_t>(outs[1].per()) * sizeof(float));
    if (outs.size() > 2 && outs[2].per() > 0)
      std::memcpy(outs[2].f(b), e.data(),
                  static_cast<size_t>(outs[2].per()) * sizeof(float));
    if (outs.size() > 3 && outs[3].per() > 0)
      std::memcpy(ItemOf<T>(outs[3], b), tau.data(),
                  static_cast<size_t>(outs[3].per()) * sizeof(T));
  }
}

// Forward/back substitution for a SINGULAR triangular system (
// `_tri_substitute`).  XLA's triangular solve never fails: a zero pivot simply
// divides through to +-inf/nan, and jnp.linalg.det's JVP depends on that (it
// runs the solve unconditionally and filters the non-finite results with a
// `where`).  LAPACK's trtrs REFUSES such a system and BLAS's trsm is free to
// scale by a reciprocal, so the singular case is computed here instead.
template <class T>
void TriSubstitute(const T* m, int64_t n, T* x, int64_t nrhs, bool lower,
                   bool unit) {
  for (int64_t s = 0; s < n; s++) {
    const int64_t i = lower ? s : n - 1 - s;
    for (int64_t j = 0; j < n; j++) {
      if (lower ? (j >= i) : (j <= i)) continue;
      const T mij = m[i * n + j];
      for (int64_t r = 0; r < nrhs; r++) x[i * nrhs + r] -= mij * x[j * nrhs + r];
    }
    if (unit) continue;
    const T d = m[i * n + i];
    for (int64_t r = 0; r < nrhs; r++) x[i * nrhs + r] = x[i * nrhs + r] / d;
  }
}

// One left solve `m x = b`, row-major, in place on `x`.  trsm unless a pivot
// is exactly zero -- which is the condition LAPACK's own trtrs reports as
// `info > 0`, and where the Python handler falls out of scipy into
// substitution.
template <class T>
void LeftSolve(const T* m, int64_t n, T* x, int64_t nrhs, bool lower,
               bool unit) {
  if (n == 0 || nrhs == 0) return;
  bool singular = false;
  if (!unit)
    for (int64_t i = 0; i < n && !singular; i++)
      singular = m[i * n + i] == T(0);
  if (singular) {
    TriSubstitute(m, n, x, nrhs, lower, unit);
    return;
  }
  Trsm(lower ? CblasLower : CblasUpper, unit ? CblasUnit : CblasNonUnit,
       Fit(n, "trsm n"), Fit(nrhs, "trsm nrhs"), m, x);
}

// stablehlo.triangular_solve / metaljax_triangular_solve.
template <class T>
void RunTriangularSolve(const HostLinalgCall& call, const Operand<T>& a,
                        const Operand<T>& b, int64_t batch,
                        std::vector<Out>& outs) {
  const int64_t n = a.dim(-1);
  const int64_t brows = b.dim(-2), bcols = b.dim(-1);
  const int64_t nrhs = call.left ? bcols : brows;   // after the right-side flip
  std::vector<T> m(static_cast<size_t>(std::max<int64_t>(n * n, 1)));
  std::vector<T> x(static_cast<size_t>(std::max<int64_t>(brows * bcols, 1)));
  std::vector<T> t(static_cast<size_t>(std::max<int64_t>(brows * bcols, 1)));
  for (int64_t bi = 0; bi < batch; bi++) {
    bool lower = call.lower;
    const T* ap = a.at(bi);
    for (int64_t i = 0; i < n * n; i++)
      m[static_cast<size_t>(i)] = call.conjugate ? Conj(ap[i]) : ap[i];
    if (call.transpose) {
      for (int64_t i = 0; i < n; i++)
        for (int64_t j = i + 1; j < n; j++) std::swap(m[i * n + j], m[j * n + i]);
      lower = !lower;
    }
    if (call.perturb && !call.unit && n > 0) {
      // XLA's perturb_singular: nudge tiny diagonal entries so the solve
      // produces finite values instead of inf/nan.
      float big = 1.0f;
      for (int64_t i = 0; i < n; i++) big = std::max(big, Abs(m[i * n + i]));
      const float eps = big * 1e-30f;
      for (int64_t i = 0; i < n; i++)
        if (Abs(m[i * n + i]) < eps) m[i * n + i] = T(eps);
    }
    if (call.left) {
      std::memcpy(x.data(), b.at(bi),
                  static_cast<size_t>(brows * bcols) * sizeof(T));
      LeftSolve(m.data(), n, x.data(), nrhs, lower, call.unit);
    } else {
      // X op(A) = B  <=>  op(A)^T X^T = B^T
      for (int64_t i = 0; i < n; i++)
        for (int64_t j = i + 1; j < n; j++) std::swap(m[i * n + j], m[j * n + i]);
      const T* bp = b.at(bi);
      for (int64_t i = 0; i < brows; i++)
        for (int64_t j = 0; j < bcols; j++) t[j * brows + i] = bp[i * bcols + j];
      LeftSolve(m.data(), n, t.data(), nrhs, !lower, call.unit);
      for (int64_t i = 0; i < brows; i++)
        for (int64_t j = 0; j < bcols; j++) x[i * bcols + j] = t[j * brows + i];
    }
    std::memcpy(ItemOf<T>(outs[0], bi), x.data(),
                static_cast<size_t>(brows * bcols) * sizeof(T));
  }
}

// The Thomas sweep for a SINGULAR tridiagonal system: like the triangular
// case, XLA divides through zero pivots instead of failing (`_thomas`).
template <class T>
void Thomas(const T* dl, const T* d, const T* du, T* x, int64_t n,
            int64_t nrhs) {
  std::vector<T> cp(static_cast<size_t>(std::max<int64_t>(n, 1)), T(0));
  if (n == 0) return;
  T piv = d[0];
  cp[0] = du[0] / piv;
  for (int64_t r = 0; r < nrhs; r++) x[r] = x[r] / piv;
  for (int64_t i = 1; i < n; i++) {
    piv = d[i] - dl[i] * cp[i - 1];
    cp[i] = du[i] / piv;
    for (int64_t r = 0; r < nrhs; r++)
      x[i * nrhs + r] = (x[i * nrhs + r] - dl[i] * x[(i - 1) * nrhs + r]) / piv;
  }
  for (int64_t i = n - 2; i >= 0; i--)
    for (int64_t r = 0; r < nrhs; r++)
      x[i * nrhs + r] = x[i * nrhs + r] - cp[i] * x[(i + 1) * nrhs + r];
}

// metaljax_tridiagonal_solve: the dense solve numpy's `solve` runs (getrf +
// getrs), falling back to the sweep when the factorization reports singular.
template <class T>
void RunTridiagonalSolve(const HostLinalgCall& call, const Operand<T>& dl,
                         const Operand<T>& d, const Operand<T>& du,
                         const Operand<T>& b, int64_t batch,
                         std::vector<Out>& outs) {
  const int64_t n = d.per();
  const int64_t nrhs = b.dim(-1);
  const Int ln = Fit(n, "tridiagonal_solve n");
  std::vector<T> m(static_cast<size_t>(std::max<int64_t>(n * n, 1)));
  std::vector<T> col(static_cast<size_t>(std::max<int64_t>(n * n, 1)));
  std::vector<T> x(static_cast<size_t>(std::max<int64_t>(n * nrhs, 1)));
  std::vector<T> xc(static_cast<size_t>(std::max<int64_t>(n * nrhs, 1)));
  std::vector<T> dd(static_cast<size_t>(std::max<int64_t>(n, 1)));
  std::vector<Int> ipiv(static_cast<size_t>(std::max<int64_t>(n, 1)), 0);
  for (int64_t bi = 0; bi < batch; bi++) {
    std::memcpy(dd.data(), d.at(bi), static_cast<size_t>(n) * sizeof(T));
    if (call.perturb && n > 0) {
      float big = 1.0f;
      for (int64_t i = 0; i < n; i++) big = std::max(big, Abs(dd[i]));
      const float eps = big * 1e-30f;
      for (int64_t i = 0; i < n; i++) if (Abs(dd[i]) < eps) dd[i] = T(eps);
    }
    std::fill(m.begin(), m.end(), T(0));
    for (int64_t i = 0; i < n; i++) m[i * n + i] = dd[i];
    for (int64_t i = 1; i < n; i++) m[i * n + (i - 1)] = dl.at(bi)[i];
    for (int64_t i = 0; i + 1 < n; i++) m[i * n + (i + 1)] = du.at(bi)[i];
    std::memcpy(x.data(), b.at(bi), static_cast<size_t>(n * nrhs) * sizeof(T));
    Int info = 0;
    if (n > 0) {
      ToCol(m.data(), col.data(), n, n);
      // getrs works on the column-major right-hand side, so the block is
      // transposed in and back out.
      for (int64_t i = 0; i < n; i++)
        for (int64_t r = 0; r < nrhs; r++) xc[i + r * n] = x[i * nrhs + r];
      Getrf(ln, ln, col.data(), ipiv.data(), &info);
      if (info == 0) {
        Getrs(ln, Fit(nrhs, "tridiagonal_solve nrhs"), col.data(), ipiv.data(),
              xc.data(), &info);
        for (int64_t i = 0; i < n; i++)
          for (int64_t r = 0; r < nrhs; r++) x[i * nrhs + r] = xc[i + r * n];
      } else {
        Thomas(dl.at(bi), dd.data(), du.at(bi), x.data(), n, nrhs);
      }
    }
    std::memcpy(ItemOf<T>(outs[0], bi), x.data(),
                static_cast<size_t>(n * nrhs) * sizeof(T));
  }
}

// --------------------------------------------------------------------------
// the dispatcher
// --------------------------------------------------------------------------

// Which operand's batch dimensions the call loops over, and how many trailing
// dimensions belong to the matrix.  (`_batch_apply` takes the batch shape from
// the operand the Python handler names; the same choice, in the same order.)
struct BatchOf {
  int operand;
  int keep;
};

BatchOf BatchSource(HostLinalg kind) {
  switch (kind) {
    case HostLinalg::kOrgqr:            return {0, 2};
    case HostLinalg::kTriangularSolve:  return {1, 2};   // `b`, as the Python
    case HostLinalg::kTridiagonalSolve: return {3, 2};
    default:                            return {0, 2};
  }
}

template <class T>
std::vector<mx::array> Run(const HostLinalgCall& call, mx::Dtype compute,
                           const std::vector<mx::array>& ins) {
  const BatchOf src = BatchSource(call.kind);
  const int64_t batch = FlatBatch(ins[src.operand], src.keep);
  std::vector<Out> outs = MakeOuts(call, batch);

  switch (call.kind) {
    case HostLinalg::kCholesky:
      RunCholesky<T>(call, Read<T>(ins[0], compute, batch), batch, outs);
      break;
    case HostLinalg::kQr:
      RunQr<T>(Read<T>(ins[0], compute, batch), batch, outs);
      break;
    case HostLinalg::kOrgqr:
      RunOrgqr<T>(call, Read<T>(ins[0], compute, batch),
                  Read<T>(ins[1], compute, batch), batch, outs);
      break;
    case HostLinalg::kEigh:
      RunEigh<T>(call, Read<T>(ins[0], compute, batch), batch, outs);
      break;
    case HostLinalg::kSvd:
      RunSvd<T>(call, Read<T>(ins[0], compute, batch), batch, outs, 0);
      break;
    case HostLinalg::kEig:
      RunEig<T>(call, Read<T>(ins[0], compute, batch), batch, outs, false);
      break;
    case HostLinalg::kLu:
      RunLu<T>(Read<T>(ins[0], compute, batch), batch, outs);
      break;
    case HostLinalg::kSchur:
      RunSchur<T>(Read<T>(ins[0], compute, batch), batch, outs);
      break;
    case HostLinalg::kHessenberg:
      RunHessenberg<T>(Read<T>(ins[0], compute, batch), batch, outs);
      break;
    case HostLinalg::kTridiagonal:
      RunTridiagonal<T>(call, Read<T>(ins[0], compute, batch), batch, outs);
      break;
    case HostLinalg::kTriangularSolve: {
      // `a` carries the batch dimensions of `b`, broadcast if it has fewer
      // (the Python handler's np.broadcast_to).
      mx::Shape want = ins[1].shape();
      want[want.size() - 2] = ins[0].shape()[ins[0].ndim() - 2];
      want[want.size() - 1] = ins[0].shape()[ins[0].ndim() - 1];
      RunTriangularSolve<T>(call, Read<T>(mx::broadcast_to(ins[0], want),
                                          compute, batch),
                            Read<T>(ins[1], compute, batch), batch, outs);
      break;
    }
    case HostLinalg::kTridiagonalSolve:
      RunTridiagonalSolve<T>(call, Read<T>(ins[0], compute, batch),
                             Read<T>(ins[1], compute, batch),
                             Read<T>(ins[2], compute, batch),
                             Read<T>(ins[3], compute, batch), batch, outs);
      break;
  }
  return Finish(outs);
}

}  // namespace

HostFn MakeHostLinalg(HostLinalgCall call) {
  if (call.results.empty())
    throw std::invalid_argument("metaljax: a host linalg call with no results");
  for (const HostSpec& s : call.results) {
    if (!is_float(s.dtype) && !is_complex(s.dtype) && s.dtype != mx::int32)
      throw std::invalid_argument(
          "metaljax: a host linalg result this file cannot type");
  }
  return [call = std::move(call)](const std::vector<mx::array>& ins)
             -> std::vector<mx::array> {
    if (ins.empty())
      throw std::runtime_error("metaljax: a host linalg call with no operands");
    // f16/bf16 compute in f32 and cast back on the way out (`_np_in`).
    const bool cplx = is_complex(ins[0].dtype());
    return cplx ? Run<C>(call, mx::complex64, ins)
                : Run<float>(call, mx::float32, ins);
  };
}

}  // namespace metaljax
