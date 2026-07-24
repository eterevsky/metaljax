// metaljax PJRT plugin: a thin C shim that trampolines into the Python
// engine (metaljax.engine) living in the same process. jaxlib dlopens this
// dylib and drives it through the PJRT C API; every compile/execute/transfer
// re-enters Python under the GIL and runs on MLX/Metal.
//
// Python symbols resolve at load time from the host process
// (-undefined dynamic_lookup), so this links against nothing.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "pjrt_c_api.h"

// ---------------------------------------------------------------- structs

// PJRT_Error and PJRT_Memory are defined in the header with a leading vtable
// pointer (the new C-ABI function-table style); we extend/instantiate them.

struct MetalError : PJRT_Error {
  std::string message;
  PJRT_Error_Code code;
};

static void ErrVtDestroy(PJRT_Error* e) { delete static_cast<MetalError*>(e); }
static void ErrVtMessage(const PJRT_Error* e, const char** m, size_t* n) {
  const auto* me = static_cast<const MetalError*>(e);
  *m = me->message.c_str();
  *n = me->message.size();
}
static PJRT_Error_Code ErrVtGetCode(const PJRT_Error* e) {
  return static_cast<const MetalError*>(e)->code;
}
static void ErrVtForEachPayload(const PJRT_Error*, PJRT_Error_PayloadVisitor,
                                void*) {}

static const PJRT_Error_FunctionTable g_error_vtable = {
    /*struct_size=*/PJRT_Error_FunctionTable_STRUCT_SIZE,
    /*instance_size=*/sizeof(MetalError),
    /*extension_start=*/nullptr,
    ErrVtDestroy,
    ErrVtMessage,
    ErrVtGetCode,
    ErrVtForEachPayload,
};

struct PJRT_Event {
  PJRT_Error* error = nullptr;  // owned; null means success. Always ready.
};

struct PJRT_DeviceDescription {};
struct PJRT_Device {};

static std::mutex g_memory_user_data_mu;
static std::map<const void*, std::pair<void*, void (*)(void*)>>
    g_memory_user_data;

static void* MemVtGetUserData(PJRT_Memory*, const void* key) {
  std::lock_guard<std::mutex> lock(g_memory_user_data_mu);
  auto it = g_memory_user_data.find(key);
  return it == g_memory_user_data.end() ? nullptr : it->second.first;
}
static void MemVtSetUserData(PJRT_Memory*, const void* key, void* data,
                             void (*dtor)(void*)) {
  std::lock_guard<std::mutex> lock(g_memory_user_data_mu);
  auto& slot = g_memory_user_data[key];
  if (slot.first != nullptr && slot.second != nullptr) slot.second(slot.first);
  slot = {data, dtor};
}

static const PJRT_Memory_FunctionTable g_memory_vtable = {
    /*struct_size=*/PJRT_Memory_FunctionTable_STRUCT_SIZE,
    /*extension_start=*/nullptr,
    /*instance_struct_size=*/sizeof(PJRT_Memory),
    MemVtGetUserData,
    MemVtSetUserData,
};

struct PJRT_Client {
  PyObject* engine = nullptr;  // borrowed from g_engine
};

struct PJRT_Executable {
  PyObject* py_exec = nullptr;  // owned
  int refs = 1;
  std::string name;
  std::string fingerprint;
  size_t num_outputs = 0;
  size_t num_params = 0;
  std::vector<PJRT_Buffer_Type> out_types;
  std::vector<int64_t> flat_dims;
  std::vector<size_t> dim_sizes;
  std::vector<const char*> memory_kinds;
  std::vector<size_t> memory_kind_sizes;
  std::vector<const char*> param_memory_kinds;
  std::vector<size_t> param_memory_kind_sizes;
};

struct PJRT_LoadedExecutable {
  PJRT_Executable* exec = nullptr;
  bool deleted = false;
};

struct PJRT_Buffer {
  PyObject* py_buf = nullptr;  // owned; has .data (mx.array), .type_enum, .dims, .nbytes
  PJRT_Buffer_Type type = PJRT_Buffer_Type_INVALID;
  std::vector<int64_t> dims;
  std::vector<int64_t> minor_to_major;
  size_t nbytes = 0;
  bool deleted = false;
};

// ---------------------------------------------------------------- globals

static PJRT_DeviceDescription g_desc;
static PJRT_Device g_device;
static PJRT_Memory g_memory = {&g_memory_vtable};
static PJRT_Device* g_devices[1] = {&g_device};
static PJRT_Memory* g_memories[1] = {&g_memory};
static PyObject* g_engine = nullptr;
static std::string g_device_kind = "Apple GPU";
static const char kMemoryKind[] = "device";

// ---------------------------------------------------------------- helpers

namespace {

class Gil {
 public:
  Gil() : state_(PyGILState_Ensure()) {}
  ~Gil() { PyGILState_Release(state_); }

 private:
  PyGILState_STATE state_;
};

PJRT_Error* Err(PJRT_Error_Code code, std::string msg) {
  auto* e = new MetalError();
  e->vtable = &g_error_vtable;
  e->message = std::move(msg);
  e->code = code;
  return e;
}

PJRT_Error* CopyErr(const PJRT_Error* e) {
  return e ? new MetalError(*static_cast<const MetalError*>(e)) : nullptr;
}

void DestroyErr(PJRT_Error* e) { delete static_cast<MetalError*>(e); }

PJRT_Event* ReadyEvent(PJRT_Error* err = nullptr) {
  auto* ev = new PJRT_Event();
  ev->error = err;
  return ev;
}

// Formats and clears the pending Python exception. Call with GIL held.
PJRT_Error* PyErrToErr(const char* where) {
  std::string msg = std::string("metaljax: ") + where + " failed";
  PJRT_Error_Code code = PJRT_Error_Code_INTERNAL;
  PyObject *ptype = nullptr, *pval = nullptr, *ptb = nullptr;
  PyErr_Fetch(&ptype, &pval, &ptb);
  PyErr_NormalizeException(&ptype, &pval, &ptb);
  if (ptype != nullptr) {
    PyObject* tname = PyObject_GetAttrString(ptype, "__name__");
    if (tname != nullptr && PyUnicode_Check(tname)) {
      const char* n = PyUnicode_AsUTF8(tname);
      if (n != nullptr && strstr(n, "Unsupported") != nullptr) {
        code = PJRT_Error_Code_UNIMPLEMENTED;
      }
    }
    Py_XDECREF(tname);
  }
  PyObject* formatted = nullptr;
  PyObject* tb_mod = PyImport_ImportModule("traceback");
  if (tb_mod != nullptr && ptype != nullptr) {
    formatted = PyObject_CallMethod(
        tb_mod, "format_exception", "OOO", ptype,
        pval ? pval : Py_None, ptb ? ptb : Py_None);
  }
  if (formatted != nullptr && PyList_Check(formatted)) {
    msg += ":\n";
    for (Py_ssize_t i = 0; i < PyList_Size(formatted); ++i) {
      const char* s = PyUnicode_AsUTF8(PyList_GetItem(formatted, i));
      if (s != nullptr) msg += s;
    }
  } else if (pval != nullptr) {
    PyObject* s = PyObject_Str(pval);
    if (s != nullptr) {
      const char* c = PyUnicode_AsUTF8(s);
      if (c != nullptr) (msg += ": ") += c;
      Py_DECREF(s);
    }
  }
  PyErr_Clear();
  Py_XDECREF(formatted);
  Py_XDECREF(tb_mod);
  Py_XDECREF(ptype);
  Py_XDECREF(pval);
  Py_XDECREF(ptb);
  return Err(code, std::move(msg));
}

// Call with GIL held.
PyObject* EnsureEngine() {
  if (g_engine == nullptr) {
    g_engine = PyImport_ImportModule("metaljax.engine");
  }
  return g_engine;
}

bool GetAttrSizeT(PyObject* obj, const char* name, size_t* out) {
  PyObject* v = PyObject_GetAttrString(obj, name);
  if (v == nullptr) return false;
  long long x = PyLong_AsLongLong(v);
  Py_DECREF(v);
  if (x == -1 && PyErr_Occurred()) return false;
  *out = static_cast<size_t>(x);
  return true;
}

bool GetAttrString(PyObject* obj, const char* name, std::string* out) {
  PyObject* v = PyObject_GetAttrString(obj, name);
  if (v == nullptr) return false;
  const char* s = PyUnicode_AsUTF8(v);
  if (s == nullptr) {
    Py_DECREF(v);
    return false;
  }
  *out = s;
  Py_DECREF(v);
  return true;
}

bool GetAttrInt64List(PyObject* obj, const char* name,
                      std::vector<int64_t>* out) {
  PyObject* v = PyObject_GetAttrString(obj, name);
  if (v == nullptr) return false;
  PyObject* seq = PySequence_Fast(v, "expected a sequence");
  Py_DECREF(v);
  if (seq == nullptr) return false;
  Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
  out->clear();
  for (Py_ssize_t i = 0; i < n; ++i) {
    long long x = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(seq, i));
    if (x == -1 && PyErr_Occurred()) {
      Py_DECREF(seq);
      return false;
    }
    out->push_back(x);
  }
  Py_DECREF(seq);
  return true;
}

// Takes ownership of py_buf (a metaljax.engine.MetalBuffer). Returns null and
// leaves a Python error set on failure.
PJRT_Buffer* WrapPyBuffer(PyObject* py_buf) {
  auto* buf = new PJRT_Buffer();
  buf->py_buf = py_buf;
  size_t type_enum = 0;
  if (!GetAttrSizeT(py_buf, "type_enum", &type_enum) ||
      !GetAttrInt64List(py_buf, "dims", &buf->dims) ||
      !GetAttrSizeT(py_buf, "nbytes", &buf->nbytes)) {
    delete buf;
    return nullptr;
  }
  buf->type = static_cast<PJRT_Buffer_Type>(type_enum);
  for (int64_t i = static_cast<int64_t>(buf->dims.size()) - 1; i >= 0; --i) {
    buf->minor_to_major.push_back(i);
  }
  return buf;
}

size_t ItemSize(PJRT_Buffer_Type t) {
  switch (t) {
    case PJRT_Buffer_Type_PRED:
    case PJRT_Buffer_Type_S8:
    case PJRT_Buffer_Type_U8:
      return 1;
    case PJRT_Buffer_Type_S16:
    case PJRT_Buffer_Type_U16:
    case PJRT_Buffer_Type_F16:
    case PJRT_Buffer_Type_BF16:
      return 2;
    case PJRT_Buffer_Type_S32:
    case PJRT_Buffer_Type_U32:
    case PJRT_Buffer_Type_F32:
      return 4;
    case PJRT_Buffer_Type_S64:
    case PJRT_Buffer_Type_U64:
    case PJRT_Buffer_Type_F64:
      return 8;
    default:
      return 0;
  }
}

}  // namespace

// ---------------------------------------------------------------- errors

static void ErrorDestroy(PJRT_Error_Destroy_Args* args) {
  DestroyErr(args->error);
}

static void ErrorMessage(PJRT_Error_Message_Args* args) {
  const auto* e = static_cast<const MetalError*>(args->error);
  args->message = e->message.c_str();
  args->message_size = e->message.size();
}

static PJRT_Error* ErrorGetCode(PJRT_Error_GetCode_Args* args) {
  args->code = static_cast<const MetalError*>(args->error)->code;
  return nullptr;
}

static PJRT_Error* ErrorForEachPayload(PJRT_Error_ForEachPayload_Args*) {
  return nullptr;  // no payloads
}

// ---------------------------------------------------------------- plugin

static PJRT_Error* PluginInitialize(PJRT_Plugin_Initialize_Args*) {
  return nullptr;
}

static PJRT_Error* PluginAttributes(PJRT_Plugin_Attributes_Args* args) {
  static std::vector<PJRT_NamedValue> attrs;
  static int64_t shlo_version[3] = {0, 0, 0};
  static bool init_done = false;
  if (!init_done) {
    init_done = true;
    Gil gil;
    PyObject* engine = EnsureEngine();
    if (engine != nullptr) {
      PyObject* v = PyObject_CallMethod(engine, "stablehlo_version", nullptr);
      if (v != nullptr && PyList_Check(v) && PyList_Size(v) == 3) {
        for (int i = 0; i < 3; ++i) {
          shlo_version[i] = PyLong_AsLongLong(PyList_GetItem(v, i));
        }
        PJRT_NamedValue nv;
        memset(&nv, 0, sizeof(nv));
        nv.struct_size = PJRT_NamedValue_STRUCT_SIZE;
        nv.name = "stablehlo_current_version";
        nv.name_size = strlen(nv.name);
        nv.type = PJRT_NamedValue_kInt64List;
        nv.int64_array_value = shlo_version;
        nv.value_size = 3;
        attrs.push_back(nv);
      }
      Py_XDECREF(v);
      PyErr_Clear();
    } else {
      PyErr_Clear();
    }
  }
  args->attributes = attrs.data();
  args->num_attributes = attrs.size();
  return nullptr;
}

// ---------------------------------------------------------------- events

static PJRT_Error* EventDestroy(PJRT_Event_Destroy_Args* args) {
  if (args->event->error != nullptr) DestroyErr(args->event->error);
  delete args->event;
  return nullptr;
}

static PJRT_Error* EventIsReady(PJRT_Event_IsReady_Args* args) {
  args->is_ready = true;
  return nullptr;
}

static PJRT_Error* EventError(PJRT_Event_Error_Args* args) {
  return CopyErr(args->event->error);
}

static PJRT_Error* EventAwait(PJRT_Event_Await_Args* args) {
  return CopyErr(args->event->error);
}

static PJRT_Error* EventOnReady(PJRT_Event_OnReady_Args* args) {
  args->callback(CopyErr(args->event->error), args->user_arg);
  return nullptr;
}

// ---------------------------------------------------------------- client

static PJRT_Error* ClientCreate(PJRT_Client_Create_Args* args) {
  Gil gil;
  PyObject* engine = EnsureEngine();
  if (engine == nullptr) return PyErrToErr("import metaljax.engine");
  PyObject* kind = PyObject_CallMethod(engine, "device_kind", nullptr);
  if (kind != nullptr) {
    const char* s = PyUnicode_AsUTF8(kind);
    if (s != nullptr) g_device_kind = s;
    Py_DECREF(kind);
  } else {
    PyErr_Clear();
  }
  auto* client = new PJRT_Client();
  client->engine = engine;
  args->client = client;
  return nullptr;
}

static PJRT_Error* ClientDestroy(PJRT_Client_Destroy_Args* args) {
  delete args->client;
  return nullptr;
}

static PJRT_Error* ClientPlatformName(PJRT_Client_PlatformName_Args* args) {
  args->platform_name = "metal";
  args->platform_name_size = 5;
  return nullptr;
}

static PJRT_Error* ClientProcessIndex(PJRT_Client_ProcessIndex_Args* args) {
  args->process_index = 0;
  return nullptr;
}

static PJRT_Error* ClientPlatformVersion(
    PJRT_Client_PlatformVersion_Args* args) {
  static const char kVersion[] = "metaljax 0.0.1";
  args->platform_version = kVersion;
  args->platform_version_size = sizeof(kVersion) - 1;
  return nullptr;
}

static PJRT_Error* ClientDevices(PJRT_Client_Devices_Args* args) {
  args->devices = g_devices;
  args->num_devices = 1;
  return nullptr;
}

static PJRT_Error* ClientAddressableDevices(
    PJRT_Client_AddressableDevices_Args* args) {
  args->addressable_devices = g_devices;
  args->num_addressable_devices = 1;
  return nullptr;
}

static PJRT_Error* ClientLookupDevice(PJRT_Client_LookupDevice_Args* args) {
  if (args->id != 0) {
    return Err(PJRT_Error_Code_INVALID_ARGUMENT, "metal has a single device 0");
  }
  args->device = &g_device;
  return nullptr;
}

static PJRT_Error* ClientLookupAddressableDevice(
    PJRT_Client_LookupAddressableDevice_Args* args) {
  if (args->local_hardware_id != 0) {
    return Err(PJRT_Error_Code_INVALID_ARGUMENT, "metal has a single device 0");
  }
  args->addressable_device = &g_device;
  return nullptr;
}

static PJRT_Error* ClientAddressableMemories(
    PJRT_Client_AddressableMemories_Args* args) {
  args->addressable_memories = g_memories;
  args->num_addressable_memories = 1;
  return nullptr;
}

static PJRT_Error* ClientDefaultDeviceAssignment(
    PJRT_Client_DefaultDeviceAssignment_Args* args) {
  for (size_t i = 0; i < args->default_assignment_size; ++i) {
    args->default_assignment[i] = 0;
  }
  return nullptr;
}

static PJRT_Error* ClientCompile(PJRT_Client_Compile_Args* args) {
  Gil gil;
  PyObject* engine = EnsureEngine();
  if (engine == nullptr) return PyErrToErr("import metaljax.engine");

  std::string fmt(args->program->format, args->program->format_size);
  PyObject* py_exec = PyObject_CallMethod(
      engine, "compile_program", "y#s", args->program->code,
      static_cast<Py_ssize_t>(args->program->code_size), fmt.c_str());
  if (py_exec == nullptr) return PyErrToErr("compile");

  auto* ex = new PJRT_Executable();
  ex->py_exec = py_exec;
  std::vector<int64_t> out_types;
  bool ok = GetAttrString(py_exec, "name", &ex->name) &&
            GetAttrString(py_exec, "fingerprint", &ex->fingerprint) &&
            GetAttrSizeT(py_exec, "num_outputs", &ex->num_outputs) &&
            GetAttrSizeT(py_exec, "num_params", &ex->num_params) &&
            GetAttrInt64List(py_exec, "out_types", &out_types);
  PyObject* dims_list = ok ? PyObject_GetAttrString(py_exec, "out_dims") : nullptr;
  if (ok && dims_list != nullptr && PyList_Check(dims_list)) {
    for (Py_ssize_t i = 0; i < PyList_Size(dims_list); ++i) {
      PyObject* dl = PyList_GetItem(dims_list, i);
      size_t rank = 0;
      for (Py_ssize_t j = 0; j < PyList_Size(dl); ++j) {
        ex->flat_dims.push_back(PyLong_AsLongLong(PyList_GetItem(dl, j)));
        ++rank;
      }
      ex->dim_sizes.push_back(rank);
    }
  } else {
    ok = false;
  }
  Py_XDECREF(dims_list);
  if (!ok) {
    PJRT_Error* e = PyErrToErr("read executable metadata");
    Py_DECREF(py_exec);
    ex->py_exec = nullptr;
    delete ex;
    return e;
  }
  for (int64_t t : out_types) {
    ex->out_types.push_back(static_cast<PJRT_Buffer_Type>(t));
  }
  ex->memory_kinds.assign(ex->num_outputs, kMemoryKind);
  ex->memory_kind_sizes.assign(ex->num_outputs, sizeof(kMemoryKind) - 1);
  ex->param_memory_kinds.assign(ex->num_params, kMemoryKind);
  ex->param_memory_kind_sizes.assign(ex->num_params, sizeof(kMemoryKind) - 1);

  auto* le = new PJRT_LoadedExecutable();
  le->exec = ex;
  args->executable = le;
  return nullptr;
}

static PJRT_Error* ClientBufferFromHostBuffer(
    PJRT_Client_BufferFromHostBuffer_Args* args) {
  size_t itemsize = ItemSize(args->type);
  if (itemsize == 0) {
    return Err(PJRT_Error_Code_UNIMPLEMENTED,
               "metaljax: unsupported host buffer element type " +
                   std::to_string(args->type));
  }
  long long nelems = 1;
  for (size_t i = 0; i < args->num_dims; ++i) nelems *= args->dims[i];

  size_t span = 0;
  if (nelems > 0) {
    if (args->byte_strides == nullptr || args->num_byte_strides == 0) {
      span = static_cast<size_t>(nelems) * itemsize;
    } else {
      span = itemsize;
      for (size_t i = 0; i < args->num_byte_strides; ++i) {
        int64_t s = args->byte_strides[i];
        if (s < 0) {
          return Err(PJRT_Error_Code_UNIMPLEMENTED,
                     "metaljax: negative byte strides not supported");
        }
        span += static_cast<size_t>(args->dims[i] - 1) * s;
      }
    }
  }

  Gil gil;
  PyObject* engine = EnsureEngine();
  if (engine == nullptr) return PyErrToErr("import metaljax.engine");

  PyObject* mv;
  if (span == 0) {
    mv = Py_None;
    Py_INCREF(mv);
  } else {
    mv = PyMemoryView_FromMemory(
        const_cast<char*>(static_cast<const char*>(args->data)),
        static_cast<Py_ssize_t>(span), PyBUF_READ);
    if (mv == nullptr) return PyErrToErr("wrap host memory");
  }
  PyObject* dims = PyList_New(args->num_dims);
  for (size_t i = 0; i < args->num_dims; ++i) {
    PyList_SET_ITEM(dims, i, PyLong_FromLongLong(args->dims[i]));
  }
  PyObject* strides;
  if (args->byte_strides == nullptr || args->num_byte_strides == 0) {
    strides = Py_None;
    Py_INCREF(strides);
  } else {
    strides = PyList_New(args->num_byte_strides);
    for (size_t i = 0; i < args->num_byte_strides; ++i) {
      PyList_SET_ITEM(strides, i, PyLong_FromLongLong(args->byte_strides[i]));
    }
  }
  PyObject* py_buf = PyObject_CallMethod(
      engine, "buffer_from_host", "OiOO", mv, static_cast<int>(args->type),
      dims, strides);
  Py_DECREF(mv);
  Py_DECREF(dims);
  Py_DECREF(strides);
  if (py_buf == nullptr) return PyErrToErr("buffer_from_host");

  PJRT_Buffer* buf = WrapPyBuffer(py_buf);
  if (buf == nullptr) return PyErrToErr("wrap buffer");
  args->buffer = buf;
  args->done_with_host_buffer = ReadyEvent();
  return nullptr;
}

// ------------------------------------------------------ device descriptors

static PJRT_Error* DeviceDescriptionId(PJRT_DeviceDescription_Id_Args* args) {
  args->id = 0;
  return nullptr;
}

static PJRT_Error* DeviceDescriptionProcessIndex(
    PJRT_DeviceDescription_ProcessIndex_Args* args) {
  args->process_index = 0;
  return nullptr;
}

static PJRT_Error* DeviceDescriptionAttributes(
    PJRT_DeviceDescription_Attributes_Args* args) {
  args->attributes = nullptr;
  args->num_attributes = 0;
  return nullptr;
}

static PJRT_Error* DeviceDescriptionKind(
    PJRT_DeviceDescription_Kind_Args* args) {
  args->device_kind = g_device_kind.c_str();
  args->device_kind_size = g_device_kind.size();
  return nullptr;
}

static PJRT_Error* DeviceDescriptionDebugString(
    PJRT_DeviceDescription_DebugString_Args* args) {
  static const char kDebug[] = "MetalDevice(id=0)";
  args->debug_string = kDebug;
  args->debug_string_size = sizeof(kDebug) - 1;
  return nullptr;
}

static PJRT_Error* DeviceDescriptionToString(
    PJRT_DeviceDescription_ToString_Args* args) {
  static const char kStr[] = "MetalDevice(id=0)";
  args->to_string = kStr;
  args->to_string_size = sizeof(kStr) - 1;
  return nullptr;
}

static PJRT_Error* DeviceGetDescription(PJRT_Device_GetDescription_Args* args) {
  args->device_description = &g_desc;
  return nullptr;
}

static PJRT_Error* DeviceIsAddressable(PJRT_Device_IsAddressable_Args* args) {
  args->is_addressable = true;
  return nullptr;
}

static PJRT_Error* DeviceLocalHardwareId(
    PJRT_Device_LocalHardwareId_Args* args) {
  args->local_hardware_id = 0;
  return nullptr;
}

static PJRT_Error* DeviceAddressableMemories(
    PJRT_Device_AddressableMemories_Args* args) {
  args->memories = g_memories;
  args->num_memories = 1;
  return nullptr;
}

static PJRT_Error* DeviceDefaultMemory(PJRT_Device_DefaultMemory_Args* args) {
  args->memory = &g_memory;
  return nullptr;
}

static PJRT_Error* DeviceMemoryStats(PJRT_Device_MemoryStats_Args* args) {
  args->bytes_in_use = 0;
  args->peak_bytes_in_use_is_set = false;
  args->num_allocs_is_set = false;
  args->largest_alloc_size_is_set = false;
  args->bytes_limit_is_set = false;
  args->bytes_reserved_is_set = false;
  args->peak_bytes_reserved_is_set = false;
  args->bytes_reservable_limit_is_set = false;
  args->largest_free_block_bytes_is_set = false;
  args->pool_bytes_is_set = false;
  args->peak_pool_bytes_is_set = false;
  args->peak_allocated_bytes_is_set = false;
  return nullptr;
}

// ---------------------------------------------------------------- memory

static PJRT_Error* MemoryId(PJRT_Memory_Id_Args* args) {
  args->id = 0;
  return nullptr;
}

static PJRT_Error* MemoryKind(PJRT_Memory_Kind_Args* args) {
  args->kind = kMemoryKind;
  args->kind_size = sizeof(kMemoryKind) - 1;
  return nullptr;
}

static PJRT_Error* MemoryKindId(PJRT_Memory_Kind_Id_Args* args) {
  args->kind_id = 0;
  return nullptr;
}

static PJRT_Error* MemoryDebugString(PJRT_Memory_DebugString_Args* args) {
  static const char kStr[] = "MetalMemory(id=0)";
  args->debug_string = kStr;
  args->debug_string_size = sizeof(kStr) - 1;
  return nullptr;
}

static PJRT_Error* MemoryToString(PJRT_Memory_ToString_Args* args) {
  static const char kStr[] = "MetalMemory(id=0)";
  args->to_string = kStr;
  args->to_string_size = sizeof(kStr) - 1;
  return nullptr;
}

static PJRT_Error* MemoryAddressableByDevices(
    PJRT_Memory_AddressableByDevices_Args* args) {
  args->devices = g_devices;
  args->num_devices = 1;
  return nullptr;
}

// ------------------------------------------------------------- executable

static void UnrefExecutable(PJRT_Executable* ex) {
  if (--ex->refs == 0) {
    Gil gil;
    Py_XDECREF(ex->py_exec);
    delete ex;
  }
}

static PJRT_Error* ExecutableDestroy(PJRT_Executable_Destroy_Args* args) {
  UnrefExecutable(args->executable);
  return nullptr;
}

static PJRT_Error* ExecutableName(PJRT_Executable_Name_Args* args) {
  args->executable_name = args->executable->name.c_str();
  args->executable_name_size = args->executable->name.size();
  return nullptr;
}

static PJRT_Error* ExecutableNumReplicas(
    PJRT_Executable_NumReplicas_Args* args) {
  args->num_replicas = 1;
  return nullptr;
}

static PJRT_Error* ExecutableNumPartitions(
    PJRT_Executable_NumPartitions_Args* args) {
  args->num_partitions = 1;
  return nullptr;
}

static PJRT_Error* ExecutableNumOutputs(PJRT_Executable_NumOutputs_Args* args) {
  args->num_outputs = args->executable->num_outputs;
  return nullptr;
}

static PJRT_Error* ExecutableSizeOfGeneratedCodeInBytes(
    PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args) {
  args->size_in_bytes = 0;
  return nullptr;
}

static PJRT_Error* ExecutableGetCostAnalysis(
    PJRT_Executable_GetCostAnalysis_Args* args) {
  args->num_properties = 0;
  args->properties = nullptr;
  return nullptr;
}

static PJRT_Error* ExecutableOutputMemoryKinds(
    PJRT_Executable_OutputMemoryKinds_Args* args) {
  args->num_outputs = args->executable->num_outputs;
  args->memory_kinds = args->executable->memory_kinds.data();
  args->memory_kind_sizes = args->executable->memory_kind_sizes.data();
  return nullptr;
}

static PJRT_Error* ExecutableParameterMemoryKinds(
    PJRT_Executable_ParameterMemoryKinds_Args* args) {
  args->num_parameters = args->executable->num_params;
  args->memory_kinds = args->executable->param_memory_kinds.data();
  args->memory_kind_sizes = args->executable->param_memory_kind_sizes.data();
  return nullptr;
}

static PJRT_Error* ExecutableOutputElementTypes(
    PJRT_Executable_OutputElementTypes_Args* args) {
  args->output_types = args->executable->out_types.data();
  args->num_output_types = args->executable->out_types.size();
  return nullptr;
}

static PJRT_Error* ExecutableOutputDimensions(
    PJRT_Executable_OutputDimensions_Args* args) {
  args->num_outputs = args->executable->num_outputs;
  args->dims = args->executable->flat_dims.data();
  args->dim_sizes = args->executable->dim_sizes.data();
  return nullptr;
}

static PJRT_Error* ExecutableFingerprint(
    PJRT_Executable_Fingerprint_Args* args) {
  args->executable_fingerprint = args->executable->fingerprint.c_str();
  args->executable_fingerprint_size = args->executable->fingerprint.size();
  return nullptr;
}

// ------------------------------------------------------ loaded executable

static PJRT_Error* LoadedExecutableDestroy(
    PJRT_LoadedExecutable_Destroy_Args* args) {
  UnrefExecutable(args->executable->exec);
  delete args->executable;
  return nullptr;
}

static PJRT_Error* LoadedExecutableGetExecutable(
    PJRT_LoadedExecutable_GetExecutable_Args* args) {
  PJRT_Executable* ex = args->loaded_executable->exec;
  ++ex->refs;
  args->executable = ex;
  return nullptr;
}

static PJRT_Error* LoadedExecutableAddressableDevices(
    PJRT_LoadedExecutable_AddressableDevices_Args* args) {
  args->addressable_devices = g_devices;
  args->num_addressable_devices = 1;
  return nullptr;
}

static PJRT_Error* LoadedExecutableDelete(
    PJRT_LoadedExecutable_Delete_Args* args) {
  args->executable->deleted = true;
  return nullptr;
}

static PJRT_Error* LoadedExecutableIsDeleted(
    PJRT_LoadedExecutable_IsDeleted_Args* args) {
  args->is_deleted = args->executable->deleted;
  return nullptr;
}

static void NoopDeviceAssignmentDeleter(PJRT_DeviceAssignmentSerialized*) {}

static PJRT_Error* LoadedExecutableGetDeviceAssignment(
    PJRT_LoadedExecutable_GetDeviceAssignment_Args* args) {
  // Hand-encoded xla.DeviceAssignmentProto:
  //   replica_count=1, computation_count=1,
  //   computation_devices { replica_device_ids: [0] (packed) }
  static const char kProto[] = {0x08, 0x01, 0x10, 0x01, 0x1A,
                                0x03, 0x0A, 0x01, 0x00};
  args->serialized_bytes = kProto;
  args->serialized_bytes_size = sizeof(kProto);
  args->serialized_device_assignment = nullptr;
  args->serialized_device_assignment_deleter = NoopDeviceAssignmentDeleter;
  return nullptr;
}

static PJRT_Error* LoadedExecutableAddressableDeviceLogicalIds(
    PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args* args) {
  static PJRT_LogicalDeviceIds ids[1] = {{0, 0}};
  args->addressable_device_logical_ids = ids;
  args->num_addressable_device_logical_ids = 1;
  return nullptr;
}

static PJRT_Error* LoadedExecutableFingerprint(
    PJRT_LoadedExecutable_Fingerprint_Args* args) {
  args->executable_fingerprint = args->executable->exec->fingerprint.c_str();
  args->executable_fingerprint_size =
      args->executable->exec->fingerprint.size();
  return nullptr;
}

static PJRT_Error* LoadedExecutableExecute(
    PJRT_LoadedExecutable_Execute_Args* args) {
  if (args->num_devices != 1) {
    return Err(PJRT_Error_Code_INVALID_ARGUMENT,
               "metaljax supports a single device");
  }
  if (args->executable->deleted) {
    return Err(PJRT_Error_Code_FAILED_PRECONDITION, "executable was deleted");
  }
  if (args->options != nullptr &&
      (args->options->num_send_ops > 0 || args->options->num_recv_ops > 0)) {
    return Err(PJRT_Error_Code_UNIMPLEMENTED,
               "metaljax: send/recv callbacks not supported");
  }

  Gil gil;
  PyObject* engine = EnsureEngine();
  if (engine == nullptr) return PyErrToErr("import metaljax.engine");

  PyObject* buf_list = PyList_New(args->num_args);
  for (size_t i = 0; i < args->num_args; ++i) {
    PJRT_Buffer* b = args->argument_lists[0][i];
    if (b->deleted) {
      Py_DECREF(buf_list);
      return Err(PJRT_Error_Code_FAILED_PRECONDITION,
                 "input buffer was deleted");
    }
    Py_INCREF(b->py_buf);
    PyList_SET_ITEM(buf_list, i, b->py_buf);
  }
  PyObject* outs = PyObject_CallMethod(
      engine, "execute", "OO", args->executable->exec->py_exec, buf_list);
  Py_DECREF(buf_list);
  if (outs == nullptr) return PyErrToErr("execute");
  if (!PyList_Check(outs) ||
      static_cast<size_t>(PyList_Size(outs)) !=
          args->executable->exec->num_outputs) {
    Py_DECREF(outs);
    return Err(PJRT_Error_Code_INTERNAL,
               "metaljax: execute returned wrong number of outputs");
  }
  for (size_t j = 0; j < args->executable->exec->num_outputs; ++j) {
    PyObject* item = PyList_GetItem(outs, j);
    Py_INCREF(item);
    PJRT_Buffer* buf = WrapPyBuffer(item);
    if (buf == nullptr) {
      Py_DECREF(outs);
      return PyErrToErr("wrap output buffer");
    }
    args->output_lists[0][j] = buf;
  }
  Py_DECREF(outs);
  if (args->device_complete_events != nullptr) {
    args->device_complete_events[0] = ReadyEvent();
  }
  return nullptr;
}

// ---------------------------------------------------------------- buffers

static PJRT_Error* BufferDestroy(PJRT_Buffer_Destroy_Args* args) {
  {
    Gil gil;
    Py_XDECREF(args->buffer->py_buf);
  }
  delete args->buffer;
  return nullptr;
}

static PJRT_Error* BufferElementType(PJRT_Buffer_ElementType_Args* args) {
  args->type = args->buffer->type;
  return nullptr;
}

static PJRT_Error* BufferDimensions(PJRT_Buffer_Dimensions_Args* args) {
  args->dims = args->buffer->dims.data();
  args->num_dims = args->buffer->dims.size();
  return nullptr;
}

static PJRT_Error* BufferUnpaddedDimensions(
    PJRT_Buffer_UnpaddedDimensions_Args* args) {
  args->unpadded_dims = args->buffer->dims.data();
  args->num_dims = args->buffer->dims.size();
  return nullptr;
}

static PJRT_Error* BufferDynamicDimensionIndices(
    PJRT_Buffer_DynamicDimensionIndices_Args* args) {
  args->dynamic_dim_indices = nullptr;
  args->num_dynamic_dims = 0;
  return nullptr;
}

static PJRT_Error* BufferGetMemoryLayout(
    PJRT_Buffer_GetMemoryLayout_Args* args) {
  memset(&args->layout, 0, sizeof(args->layout));
  args->layout.struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE;
  args->layout.type = PJRT_Buffer_MemoryLayout_Type_Tiled;
  args->layout.tiled.struct_size = PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE;
  args->layout.tiled.minor_to_major = args->buffer->minor_to_major.data();
  args->layout.tiled.minor_to_major_size = args->buffer->minor_to_major.size();
  args->layout.tiled.num_tiles = 0;
  return nullptr;
}

static PJRT_Error* BufferOnDeviceSizeInBytes(
    PJRT_Buffer_OnDeviceSizeInBytes_Args* args) {
  args->on_device_size_in_bytes = args->buffer->nbytes;
  return nullptr;
}

static PJRT_Error* BufferDevice(PJRT_Buffer_Device_Args* args) {
  args->device = &g_device;
  return nullptr;
}

static PJRT_Error* BufferMemory(PJRT_Buffer_Memory_Args* args) {
  args->memory = &g_memory;
  return nullptr;
}

static PJRT_Error* BufferDelete(PJRT_Buffer_Delete_Args* args) {
  if (!args->buffer->deleted) {
    Gil gil;
    Py_CLEAR(args->buffer->py_buf);
    args->buffer->deleted = true;
  }
  return nullptr;
}

static PJRT_Error* BufferIsDeleted(PJRT_Buffer_IsDeleted_Args* args) {
  args->is_deleted = args->buffer->deleted;
  return nullptr;
}

static PJRT_Buffer* DuplicateBuffer(PJRT_Buffer* src) {
  auto* buf = new PJRT_Buffer(*src);
  Gil gil;
  Py_XINCREF(buf->py_buf);
  return buf;
}

static PJRT_Error* BufferCopyToDevice(PJRT_Buffer_CopyToDevice_Args* args) {
  args->dst_buffer = DuplicateBuffer(args->buffer);
  return nullptr;
}

static PJRT_Error* BufferCopyToMemory(PJRT_Buffer_CopyToMemory_Args* args) {
  args->dst_buffer = DuplicateBuffer(args->buffer);
  return nullptr;
}

static PJRT_Error* BufferToHostBuffer(PJRT_Buffer_ToHostBuffer_Args* args) {
  PJRT_Buffer* src = args->src;
  if (src->deleted) {
    return Err(PJRT_Error_Code_FAILED_PRECONDITION, "buffer was deleted");
  }
  if (args->dst == nullptr) {
    args->dst_size = src->nbytes;
    return nullptr;
  }
  Gil gil;
  PyObject* engine = EnsureEngine();
  if (engine == nullptr) return PyErrToErr("import metaljax.engine");
  PyObject* bytes = PyObject_CallMethod(engine, "to_host", "O", src->py_buf);
  if (bytes == nullptr) return PyErrToErr("to_host");
  char* data = nullptr;
  Py_ssize_t size = 0;
  if (PyBytes_AsStringAndSize(bytes, &data, &size) != 0) {
    Py_DECREF(bytes);
    return PyErrToErr("to_host bytes");
  }
  if (static_cast<size_t>(size) > args->dst_size) {
    Py_DECREF(bytes);
    return Err(PJRT_Error_Code_INVALID_ARGUMENT,
               "metaljax: destination buffer too small");
  }
  memcpy(args->dst, data, size);
  Py_DECREF(bytes);
  args->event = ReadyEvent();
  return nullptr;
}

static PJRT_Error* BufferIsOnCpu(PJRT_Buffer_IsOnCpu_Args* args) {
  args->is_on_cpu = false;
  return nullptr;
}

static PJRT_Error* BufferReadyEvent(PJRT_Buffer_ReadyEvent_Args* args) {
  args->event = ReadyEvent();
  return nullptr;
}

// ------------------------------------------------------ unimplemented rest

#define METALJAX_UNIMPL(Name)                                       \
  static PJRT_Error* Name##_Unimpl(Name##_Args*) {                  \
    return Err(PJRT_Error_Code_UNIMPLEMENTED,                       \
               #Name " is not implemented in the metaljax plugin"); \
  }

METALJAX_UNIMPL(PJRT_Executable_OptimizedProgram)
METALJAX_UNIMPL(PJRT_Executable_Serialize)
METALJAX_UNIMPL(PJRT_Executable_DeserializeAndLoad)
METALJAX_UNIMPL(PJRT_Executable_GetCompiledMemoryStats)
METALJAX_UNIMPL(PJRT_Executable_GetCompileOptions)
METALJAX_UNIMPL(PJRT_Buffer_UnsafePointer)
METALJAX_UNIMPL(PJRT_Buffer_IncreaseExternalReferenceCount)
METALJAX_UNIMPL(PJRT_Buffer_DecreaseExternalReferenceCount)
METALJAX_UNIMPL(PJRT_Buffer_OpaqueDeviceMemoryDataPointer)
METALJAX_UNIMPL(PJRT_Buffer_CopyRawToHost)
METALJAX_UNIMPL(PJRT_Buffer_CopyRawToHostFuture)
METALJAX_UNIMPL(PJRT_Buffer_Bitcast)
METALJAX_UNIMPL(PJRT_Buffer_DonateWithControlDependency)
METALJAX_UNIMPL(PJRT_CopyToDeviceStream_Destroy)
METALJAX_UNIMPL(PJRT_CopyToDeviceStream_AddChunk)
METALJAX_UNIMPL(PJRT_CopyToDeviceStream_TotalBytes)
METALJAX_UNIMPL(PJRT_CopyToDeviceStream_GranuleSize)
METALJAX_UNIMPL(PJRT_CopyToDeviceStream_CurrentBytes)
METALJAX_UNIMPL(PJRT_TopologyDescription_Create)
METALJAX_UNIMPL(PJRT_TopologyDescription_Destroy)
METALJAX_UNIMPL(PJRT_TopologyDescription_PlatformName)
METALJAX_UNIMPL(PJRT_TopologyDescription_PlatformVersion)
METALJAX_UNIMPL(PJRT_TopologyDescription_GetDeviceDescriptions)
METALJAX_UNIMPL(PJRT_TopologyDescription_Serialize)
METALJAX_UNIMPL(PJRT_TopologyDescription_Attributes)
METALJAX_UNIMPL(PJRT_TopologyDescription_Deserialize)
METALJAX_UNIMPL(PJRT_TopologyDescription_Fingerprint)
METALJAX_UNIMPL(PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace)
METALJAX_UNIMPL(PJRT_TopologyDescription_GetMemorySpaceKindIds)
METALJAX_UNIMPL(PJRT_Compile)
METALJAX_UNIMPL(PJRT_Client_TopologyDescription)
METALJAX_UNIMPL(PJRT_Client_CreateViewOfDeviceBuffer)
METALJAX_UNIMPL(PJRT_ExecuteContext_Create)
METALJAX_UNIMPL(PJRT_ExecuteContext_Destroy)
METALJAX_UNIMPL(PJRT_AsyncHostToDeviceTransferManager_Destroy)
METALJAX_UNIMPL(PJRT_AsyncHostToDeviceTransferManager_TransferData)
METALJAX_UNIMPL(PJRT_Client_CreateBuffersForAsyncHostToDevice)
METALJAX_UNIMPL(PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer)
METALJAX_UNIMPL(PJRT_AsyncHostToDeviceTransferManager_Device)
METALJAX_UNIMPL(PJRT_AsyncHostToDeviceTransferManager_BufferCount)
METALJAX_UNIMPL(PJRT_AsyncHostToDeviceTransferManager_BufferSize)
METALJAX_UNIMPL(PJRT_AsyncHostToDeviceTransferManager_SetBufferError)
METALJAX_UNIMPL(PJRT_AsyncHostToDeviceTransferManager_AddMetadata)
METALJAX_UNIMPL(PJRT_AsyncHostToDeviceTransferManager_TransferLiteral)
METALJAX_UNIMPL(PJRT_Client_DmaMap)
METALJAX_UNIMPL(PJRT_Client_DmaUnmap)
METALJAX_UNIMPL(PJRT_Client_CreateUninitializedBuffer)
METALJAX_UNIMPL(PJRT_Client_UpdateGlobalProcessInfo)
METALJAX_UNIMPL(PJRT_Client_CreateAliasBuffer)
METALJAX_UNIMPL(PJRT_Client_FulfillAliasBuffer)
METALJAX_UNIMPL(PJRT_Client_CreateErrorBuffer)
METALJAX_UNIMPL(PJRT_Device_PoisonExecution)
METALJAX_UNIMPL(PJRT_Device_CreateAsyncTrackingEvent)
METALJAX_UNIMPL(PJRT_AsyncTrackingEvent_Destroy)
METALJAX_UNIMPL(PJRT_Event_Create)
METALJAX_UNIMPL(PJRT_Event_Set)
METALJAX_UNIMPL(PJRT_Client_Load)
METALJAX_UNIMPL(PJRT_Device_ClearMemoryStats)

static void NoopDeviceAttributesDeleter(PJRT_Device_Attributes*) {}

static PJRT_Error* DeviceGetAttributes(PJRT_Device_GetAttributes_Args* args) {
  args->attributes = nullptr;
  args->num_attributes = 0;
  args->device_attributes = nullptr;
  args->attributes_deleter = NoopDeviceAttributesDeleter;
  return nullptr;
}

// ---------------------------------------------------------------- the API

extern "C" __attribute__((visibility("default"))) const PJRT_Api* GetPjrtApi() {
  static PJRT_Api api;
  static bool init = false;
  if (init) return &api;
  init = true;
  memset(&api, 0, sizeof(api));
  api.struct_size = PJRT_Api_STRUCT_SIZE;
  api.extension_start = nullptr;
  api.pjrt_api_version.struct_size = PJRT_Api_Version_STRUCT_SIZE;
  api.pjrt_api_version.major_version = PJRT_API_MAJOR;
  api.pjrt_api_version.minor_version = PJRT_API_MINOR;

  api.PJRT_Error_Destroy = ErrorDestroy;
  api.PJRT_Error_Message = ErrorMessage;
  api.PJRT_Error_GetCode = ErrorGetCode;
  api.PJRT_Error_ForEachPayload = ErrorForEachPayload;

  api.PJRT_Plugin_Initialize = PluginInitialize;
  api.PJRT_Plugin_Attributes = PluginAttributes;

  api.PJRT_Event_Destroy = EventDestroy;
  api.PJRT_Event_IsReady = EventIsReady;
  api.PJRT_Event_Error = EventError;
  api.PJRT_Event_Await = EventAwait;
  api.PJRT_Event_OnReady = EventOnReady;
  api.PJRT_Event_Create = PJRT_Event_Create_Unimpl;
  api.PJRT_Event_Set = PJRT_Event_Set_Unimpl;

  api.PJRT_Client_Create = ClientCreate;
  api.PJRT_Client_Destroy = ClientDestroy;
  api.PJRT_Client_PlatformName = ClientPlatformName;
  api.PJRT_Client_ProcessIndex = ClientProcessIndex;
  api.PJRT_Client_PlatformVersion = ClientPlatformVersion;
  api.PJRT_Client_Devices = ClientDevices;
  api.PJRT_Client_AddressableDevices = ClientAddressableDevices;
  api.PJRT_Client_LookupDevice = ClientLookupDevice;
  api.PJRT_Client_LookupAddressableDevice = ClientLookupAddressableDevice;
  api.PJRT_Client_AddressableMemories = ClientAddressableMemories;
  api.PJRT_Client_Compile = ClientCompile;
  api.PJRT_Client_DefaultDeviceAssignment = ClientDefaultDeviceAssignment;
  api.PJRT_Client_BufferFromHostBuffer = ClientBufferFromHostBuffer;
  api.PJRT_Client_TopologyDescription = PJRT_Client_TopologyDescription_Unimpl;
  api.PJRT_Client_CreateViewOfDeviceBuffer =
      PJRT_Client_CreateViewOfDeviceBuffer_Unimpl;
  api.PJRT_Client_CreateBuffersForAsyncHostToDevice =
      PJRT_Client_CreateBuffersForAsyncHostToDevice_Unimpl;
  api.PJRT_Client_DmaMap = PJRT_Client_DmaMap_Unimpl;
  api.PJRT_Client_DmaUnmap = PJRT_Client_DmaUnmap_Unimpl;
  api.PJRT_Client_CreateUninitializedBuffer =
      PJRT_Client_CreateUninitializedBuffer_Unimpl;
  api.PJRT_Client_UpdateGlobalProcessInfo =
      PJRT_Client_UpdateGlobalProcessInfo_Unimpl;
  api.PJRT_Client_CreateAliasBuffer = PJRT_Client_CreateAliasBuffer_Unimpl;
  api.PJRT_Client_FulfillAliasBuffer = PJRT_Client_FulfillAliasBuffer_Unimpl;
  api.PJRT_Client_CreateErrorBuffer = PJRT_Client_CreateErrorBuffer_Unimpl;
  api.PJRT_Client_Load = PJRT_Client_Load_Unimpl;

  api.PJRT_DeviceDescription_Id = DeviceDescriptionId;
  api.PJRT_DeviceDescription_ProcessIndex = DeviceDescriptionProcessIndex;
  api.PJRT_DeviceDescription_Attributes = DeviceDescriptionAttributes;
  api.PJRT_DeviceDescription_Kind = DeviceDescriptionKind;
  api.PJRT_DeviceDescription_DebugString = DeviceDescriptionDebugString;
  api.PJRT_DeviceDescription_ToString = DeviceDescriptionToString;

  api.PJRT_Device_GetDescription = DeviceGetDescription;
  api.PJRT_Device_IsAddressable = DeviceIsAddressable;
  api.PJRT_Device_LocalHardwareId = DeviceLocalHardwareId;
  api.PJRT_Device_AddressableMemories = DeviceAddressableMemories;
  api.PJRT_Device_DefaultMemory = DeviceDefaultMemory;
  api.PJRT_Device_MemoryStats = DeviceMemoryStats;
  api.PJRT_Device_PoisonExecution = PJRT_Device_PoisonExecution_Unimpl;
  api.PJRT_Device_CreateAsyncTrackingEvent =
      PJRT_Device_CreateAsyncTrackingEvent_Unimpl;
  api.PJRT_Device_ClearMemoryStats = PJRT_Device_ClearMemoryStats_Unimpl;
  api.PJRT_Device_GetAttributes = DeviceGetAttributes;

  api.PJRT_Memory_Id = MemoryId;
  api.PJRT_Memory_Kind = MemoryKind;
  api.PJRT_Memory_Kind_Id = MemoryKindId;
  api.PJRT_Memory_DebugString = MemoryDebugString;
  api.PJRT_Memory_ToString = MemoryToString;
  api.PJRT_Memory_AddressableByDevices = MemoryAddressableByDevices;

  api.PJRT_Executable_Destroy = ExecutableDestroy;
  api.PJRT_Executable_Name = ExecutableName;
  api.PJRT_Executable_NumReplicas = ExecutableNumReplicas;
  api.PJRT_Executable_NumPartitions = ExecutableNumPartitions;
  api.PJRT_Executable_NumOutputs = ExecutableNumOutputs;
  api.PJRT_Executable_SizeOfGeneratedCodeInBytes =
      ExecutableSizeOfGeneratedCodeInBytes;
  api.PJRT_Executable_GetCostAnalysis = ExecutableGetCostAnalysis;
  api.PJRT_Executable_OutputMemoryKinds = ExecutableOutputMemoryKinds;
  api.PJRT_Executable_ParameterMemoryKinds = ExecutableParameterMemoryKinds;
  api.PJRT_Executable_OptimizedProgram =
      PJRT_Executable_OptimizedProgram_Unimpl;
  api.PJRT_Executable_Serialize = PJRT_Executable_Serialize_Unimpl;
  api.PJRT_Executable_OutputElementTypes = ExecutableOutputElementTypes;
  api.PJRT_Executable_OutputDimensions = ExecutableOutputDimensions;
  api.PJRT_Executable_Fingerprint = ExecutableFingerprint;
  api.PJRT_Executable_GetCompiledMemoryStats =
      PJRT_Executable_GetCompiledMemoryStats_Unimpl;
  api.PJRT_Executable_GetCompileOptions =
      PJRT_Executable_GetCompileOptions_Unimpl;
  api.PJRT_Executable_DeserializeAndLoad =
      PJRT_Executable_DeserializeAndLoad_Unimpl;

  api.PJRT_LoadedExecutable_Destroy = LoadedExecutableDestroy;
  api.PJRT_LoadedExecutable_GetExecutable = LoadedExecutableGetExecutable;
  api.PJRT_LoadedExecutable_AddressableDevices =
      LoadedExecutableAddressableDevices;
  api.PJRT_LoadedExecutable_Delete = LoadedExecutableDelete;
  api.PJRT_LoadedExecutable_IsDeleted = LoadedExecutableIsDeleted;
  api.PJRT_LoadedExecutable_Execute = LoadedExecutableExecute;
  api.PJRT_LoadedExecutable_Fingerprint = LoadedExecutableFingerprint;
  api.PJRT_LoadedExecutable_GetDeviceAssignment =
      LoadedExecutableGetDeviceAssignment;
  api.PJRT_LoadedExecutable_AddressableDeviceLogicalIds =
      LoadedExecutableAddressableDeviceLogicalIds;

  api.PJRT_Buffer_Destroy = BufferDestroy;
  api.PJRT_Buffer_ElementType = BufferElementType;
  api.PJRT_Buffer_Dimensions = BufferDimensions;
  api.PJRT_Buffer_UnpaddedDimensions = BufferUnpaddedDimensions;
  api.PJRT_Buffer_DynamicDimensionIndices = BufferDynamicDimensionIndices;
  api.PJRT_Buffer_GetMemoryLayout = BufferGetMemoryLayout;
  api.PJRT_Buffer_OnDeviceSizeInBytes = BufferOnDeviceSizeInBytes;
  api.PJRT_Buffer_Device = BufferDevice;
  api.PJRT_Buffer_Memory = BufferMemory;
  api.PJRT_Buffer_Delete = BufferDelete;
  api.PJRT_Buffer_IsDeleted = BufferIsDeleted;
  api.PJRT_Buffer_CopyToDevice = BufferCopyToDevice;
  api.PJRT_Buffer_CopyToMemory = BufferCopyToMemory;
  api.PJRT_Buffer_ToHostBuffer = BufferToHostBuffer;
  api.PJRT_Buffer_IsOnCpu = BufferIsOnCpu;
  api.PJRT_Buffer_ReadyEvent = BufferReadyEvent;
  api.PJRT_Buffer_UnsafePointer = PJRT_Buffer_UnsafePointer_Unimpl;
  api.PJRT_Buffer_IncreaseExternalReferenceCount =
      PJRT_Buffer_IncreaseExternalReferenceCount_Unimpl;
  api.PJRT_Buffer_DecreaseExternalReferenceCount =
      PJRT_Buffer_DecreaseExternalReferenceCount_Unimpl;
  api.PJRT_Buffer_OpaqueDeviceMemoryDataPointer =
      PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Unimpl;
  api.PJRT_Buffer_CopyRawToHost = PJRT_Buffer_CopyRawToHost_Unimpl;
  api.PJRT_Buffer_CopyRawToHostFuture = PJRT_Buffer_CopyRawToHostFuture_Unimpl;
  api.PJRT_Buffer_Bitcast = PJRT_Buffer_Bitcast_Unimpl;
  api.PJRT_Buffer_DonateWithControlDependency =
      PJRT_Buffer_DonateWithControlDependency_Unimpl;

  api.PJRT_CopyToDeviceStream_Destroy = PJRT_CopyToDeviceStream_Destroy_Unimpl;
  api.PJRT_CopyToDeviceStream_AddChunk =
      PJRT_CopyToDeviceStream_AddChunk_Unimpl;
  api.PJRT_CopyToDeviceStream_TotalBytes =
      PJRT_CopyToDeviceStream_TotalBytes_Unimpl;
  api.PJRT_CopyToDeviceStream_GranuleSize =
      PJRT_CopyToDeviceStream_GranuleSize_Unimpl;
  api.PJRT_CopyToDeviceStream_CurrentBytes =
      PJRT_CopyToDeviceStream_CurrentBytes_Unimpl;

  api.PJRT_TopologyDescription_Create = PJRT_TopologyDescription_Create_Unimpl;
  api.PJRT_TopologyDescription_Destroy =
      PJRT_TopologyDescription_Destroy_Unimpl;
  api.PJRT_TopologyDescription_PlatformName =
      PJRT_TopologyDescription_PlatformName_Unimpl;
  api.PJRT_TopologyDescription_PlatformVersion =
      PJRT_TopologyDescription_PlatformVersion_Unimpl;
  api.PJRT_TopologyDescription_GetDeviceDescriptions =
      PJRT_TopologyDescription_GetDeviceDescriptions_Unimpl;
  api.PJRT_TopologyDescription_Serialize =
      PJRT_TopologyDescription_Serialize_Unimpl;
  api.PJRT_TopologyDescription_Attributes =
      PJRT_TopologyDescription_Attributes_Unimpl;
  api.PJRT_TopologyDescription_Deserialize =
      PJRT_TopologyDescription_Deserialize_Unimpl;
  api.PJRT_TopologyDescription_Fingerprint =
      PJRT_TopologyDescription_Fingerprint_Unimpl;
  api.PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace =
      PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Unimpl;
  api.PJRT_TopologyDescription_GetMemorySpaceKindIds =
      PJRT_TopologyDescription_GetMemorySpaceKindIds_Unimpl;

  api.PJRT_Compile = PJRT_Compile_Unimpl;

  api.PJRT_ExecuteContext_Create = PJRT_ExecuteContext_Create_Unimpl;
  api.PJRT_ExecuteContext_Destroy = PJRT_ExecuteContext_Destroy_Unimpl;

  api.PJRT_AsyncHostToDeviceTransferManager_Destroy =
      PJRT_AsyncHostToDeviceTransferManager_Destroy_Unimpl;
  api.PJRT_AsyncHostToDeviceTransferManager_TransferData =
      PJRT_AsyncHostToDeviceTransferManager_TransferData_Unimpl;
  api.PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer =
      PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Unimpl;
  api.PJRT_AsyncHostToDeviceTransferManager_Device =
      PJRT_AsyncHostToDeviceTransferManager_Device_Unimpl;
  api.PJRT_AsyncHostToDeviceTransferManager_BufferCount =
      PJRT_AsyncHostToDeviceTransferManager_BufferCount_Unimpl;
  api.PJRT_AsyncHostToDeviceTransferManager_BufferSize =
      PJRT_AsyncHostToDeviceTransferManager_BufferSize_Unimpl;
  api.PJRT_AsyncHostToDeviceTransferManager_SetBufferError =
      PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Unimpl;
  api.PJRT_AsyncHostToDeviceTransferManager_AddMetadata =
      PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Unimpl;
  api.PJRT_AsyncHostToDeviceTransferManager_TransferLiteral =
      PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Unimpl;

  api.PJRT_AsyncTrackingEvent_Destroy = PJRT_AsyncTrackingEvent_Destroy_Unimpl;

  return &api;
}
