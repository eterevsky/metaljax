// metaljax native engine — the linalg family that computes on the HOST (P9).
//
// Every factorization below is CPU-bound in every backend (XLA's CPU backend
// calls LAPACK, and MLX's own `mx::linalg` runs on its CPU stream), and unified
// memory makes the round trip cheap.  So the tape holds them as `kHostCall`
// entries: one `HostFn` per call site, bound at LOWERING time by the plugin,
// which is the one place a native run leaves the tape.
//
// This header is the line between the two halves.  The PLUGIN reads the IR —
// which target, which flags, what result types — and asks for a handler; the
// implementation (host_lapack.cc) knows only arrays and calls Accelerate.
// `src/metaljax/ops/lapack.py` (Stage 1, deleted 0.11.6, ef5774d) was the
// specification for both: the semantics
// here are that module's, handler for handler, including the ones that exist
// to match XLA rather than LAPACK (a singular triangular solve divides through
// to +-inf/nan instead of failing, a non-positive-definite cholesky is all
// NaN, and `perturb_singular` nudges tiny pivots).

#ifndef METALJAX_HOST_LAPACK_H_
#define METALJAX_HOST_LAPACK_H_

#include <cstdint>
#include <vector>

#include "program.h"

namespace metaljax {

// Which factorization a call runs.  Named after the StableHLO op or the
// custom-call target jax emits, not after LAPACK's routine: which routine
// serves depends on the element type, which only the handler sees.
enum class HostLinalg {
  kCholesky,          // stablehlo.cholesky                     potrf
  kQr,                // @Qr, lapack_?geqrf_ffi                 geqrf
  kOrgqr,             // @ProductOfElementaryHouseholderReflectors,
                      // lapack_?orgqr_ffi                      orgqr / ungqr
  kEigh,              // metaljax_eigh                          syevd / heevd
  kSvd,               // metaljax_svd                           gesdd
  kEig,               // metaljax_eig                           geev
  kLu,                // metaljax_lu                            getrf
  kSchur,             // metaljax_schur                         gees
  kHessenberg,        // metaljax_hessenberg                    gehrd
  kTridiagonal,       // metaljax_tridiagonal                   sytrd / hetrd
  kTriangularSolve,   // stablehlo.triangular_solve,
                      // metaljax_triangular_solve              trsm
  kTridiagonalSolve,  // metaljax_tridiagonal_solve             getrf/getrs
};

// One declared result of the call, read off the op's result type.  The shape
// carries the leading BATCH dimensions too: LAPACK is a single-matrix library,
// so the handler loops them exactly as `_batch_apply` does.
struct HostSpec {
  std::vector<int64_t> shape;
  mx::Dtype dtype;
};

// The flags a target's `backend_config` (or the op's own attributes) carries.
// Each is read by the handlers named beside it and ignored by the rest.
struct HostLinalgCall {
  HostLinalg kind;
  std::vector<HostSpec> results;
  bool lower = true;            // cholesky, eigh, tridiagonal
  bool left = true;             // triangular solve: side
  bool transpose = false;       //   ...and its four other flags
  bool conjugate = false;
  bool unit = false;
  bool perturb = false;         // triangular solve, tridiagonal solve
  bool left_vectors = false;    // eig: compute_left_eigenvectors
  bool right_vectors = true;    // eig: compute_right_eigenvectors
};

// The handler for one call site.  Throws `std::invalid_argument` at BIND time
// for a combination this file does not serve, so the plugin declines the
// program with a name rather than failing at execute.
HostFn MakeHostLinalg(HostLinalgCall call);

}  // namespace metaljax

#endif  // METALJAX_HOST_LAPACK_H_
