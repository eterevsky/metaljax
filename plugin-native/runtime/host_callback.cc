// metaljax native engine — jax's host callbacks (P13).  See host_callback.h.
//
// `src/metaljax/ops/callbacks.py::_run_callback` is the specification: the
// operands cross as numpy arrays, the callable's results are reshaped and cast
// to the DECLARED result types, and a callback that returns nothing (every
// `jax.debug.print`) hands back an empty list.  The difference here is only in
// who does the converting — there, numpy; here, a staging copy on each side of
// a C function pointer.

#include "host_callback.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace metaljax {
namespace {

CallbackTrampoline g_trampoline = nullptr;

int32_t HostDtypeOf(mx::Dtype dt) {
  if (dt == mx::bool_) return kHostBool;
  if (dt == mx::int8) return kHostInt8;
  if (dt == mx::int16) return kHostInt16;
  if (dt == mx::int32) return kHostInt32;
  if (dt == mx::int64) return kHostInt64;
  if (dt == mx::uint8) return kHostUint8;
  if (dt == mx::uint16) return kHostUint16;
  if (dt == mx::uint32) return kHostUint32;
  if (dt == mx::uint64) return kHostUint64;
  if (dt == mx::float16) return kHostFloat16;
  if (dt == mx::bfloat16) return kHostBfloat16;
  if (dt == mx::float32) return kHostFloat32;
  if (dt == mx::complex64) return kHostComplex64;
  return -1;
}

int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

// A staging block that becomes an array's own storage: one allocation, freed
// by MLX when the array dies (host_lapack.cc `Out::Finish`, same arrangement).
struct Staged {
  void* data = nullptr;
  std::vector<int64_t> dims;

  Staged() = default;
  Staged(const Staged&) = delete;             // it owns `data`
  Staged& operator=(const Staged&) = delete;
  ~Staged() { std::free(data); }
  void* release() {
    void* p = data;
    data = nullptr;
    return p;
  }
};

}  // namespace

void SetCallbackTrampoline(CallbackTrampoline fn) { g_trampoline = fn; }

bool HasCallbackTrampoline() { return g_trampoline != nullptr; }

HostFn MakeHostCallback(int32_t index, std::vector<CallbackSpec> results) {
  if (g_trampoline == nullptr)
    throw std::invalid_argument(
        "metaljax: a host callback with no trampoline installed");
  for (const CallbackSpec& s : results) {
    if (HostDtypeOf(s.dtype) < 0)
      throw std::invalid_argument(
          "metaljax: a host callback result this ABI cannot carry");
  }
  return [index, results = std::move(results)](
             const std::vector<mx::array>& ins) -> std::vector<mx::array> {
    // The operands, settled and made dense: the embedder reads row-major
    // memory and nothing else.  The eval is also the data dependency that puts
    // this call after the device work it reads, which is what sequences a
    // print against the computation whose value it prints.
    std::vector<mx::array> held;
    std::vector<std::vector<int64_t>> in_dims;
    std::vector<MetaljaxHostBuffer> in_bufs;
    held.reserve(ins.size());
    in_dims.reserve(ins.size());
    for (const mx::array& x : ins) {
      const int32_t code = HostDtypeOf(x.dtype());
      if (code < 0)
        throw std::runtime_error(
            "metaljax: a host callback operand this ABI cannot carry");
      // Settle the array ITSELF, then ask its layout -- MLX sets the flags at
      // eval, and `data_size` is the one that matters here: a broadcast view
      // holds ONE element under a full shape, and handing the embedder its
      // pointer with the full shape beside it is a short-buffer overread
      // (CLAUDE.md item 20's conv bug, in a new place).  Same shape as
      // MetalBuffer::Settled, for the same reason.
      mx::array c = x;
      c.eval();
      if (!c.flags().row_contiguous ||
          c.data_size() != static_cast<size_t>(c.size())) {
        c = mx::contiguous(c);
        c.eval();
      }
      held.push_back(c);
      std::vector<int64_t> dims;
      for (auto d : c.shape()) dims.push_back(static_cast<int64_t>(d));
      in_dims.push_back(std::move(dims));
    }
    for (size_t i = 0; i < held.size(); i++) {
      in_bufs.push_back(MetaljaxHostBuffer{
          held[i].size() > 0 ? held[i].data<void>() : nullptr,
          HostDtypeOf(held[i].dtype()),
          static_cast<int32_t>(in_dims[i].size()), in_dims[i].data()});
    }

    // The results, staged zero-filled: a callable that writes nothing into an
    // output leaves zeros rather than whatever the allocator held, and the
    // embedder never sees uninitialised memory it might read back.
    std::vector<Staged> staged(results.size());
    std::vector<MetaljaxHostBuffer> out_bufs;
    for (size_t i = 0; i < results.size(); i++) {
      const int64_t n = Numel(results[i].shape);
      staged[i].dims = results[i].shape;
      const size_t nbytes =
          static_cast<size_t>(n) * results[i].dtype.size();
      if (nbytes > 0) {
        staged[i].data = std::calloc(1, nbytes);
        if (staged[i].data == nullptr)
          throw std::runtime_error(
              "metaljax: could not stage a host callback result");
      }
      out_bufs.push_back(MetaljaxHostBuffer{
          staged[i].data, HostDtypeOf(results[i].dtype),
          static_cast<int32_t>(staged[i].dims.size()), staged[i].dims.data()});
    }

    char error[512];
    error[0] = '\0';
    const int32_t rc = g_trampoline(
        index, static_cast<int32_t>(in_bufs.size()),
        in_bufs.empty() ? nullptr : in_bufs.data(),
        static_cast<int32_t>(out_bufs.size()),
        out_bufs.empty() ? nullptr : out_bufs.data(), error,
        static_cast<int32_t>(sizeof(error)));
    if (rc != 0) {
      error[sizeof(error) - 1] = '\0';
      throw std::runtime_error(
          std::string("metaljax: host callback failed: ") +
          (error[0] != '\0' ? error : "no message"));
    }

    std::vector<mx::array> out;
    out.reserve(results.size());
    for (size_t i = 0; i < results.size(); i++) {
      mx::Shape shp;
      for (int64_t d : results[i].shape)
        shp.push_back(static_cast<mx::ShapeElem>(d));
      if (staged[i].data == nullptr) {
        out.push_back(mx::zeros(shp, results[i].dtype));
        continue;
      }
      out.push_back(mx::array(staged[i].release(), shp, results[i].dtype,
                              [](void* p) { std::free(p); }));
    }
    return out;
  };
}

}  // namespace metaljax
