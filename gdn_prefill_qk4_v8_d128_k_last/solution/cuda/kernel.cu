#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>

namespace py = pybind11;

namespace {

std::string current_module_dir() {
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&current_module_dir), &info) == 0 || info.dli_fname == nullptr) {
        throw std::runtime_error("dladdr failed for kernel.cu bridge module");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
}

py::object get_bridge_run() {
    static py::object bridge_run = py::none();
    if (!bridge_run.is_none()) {
        return bridge_run;
    }

    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, py::str(current_module_dir()));
    bridge_run = py::module_::import("pybridge.main").attr("run");
    return bridge_run;
}

py::object run(
    py::object q,
    py::object k,
    py::object v,
    py::object state,
    py::object A_log,
    py::object a,
    py::object dt_bias,
    py::object b,
    py::object cu_seqlens,
    py::object scale
) {
    return get_bridge_run()(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Forward GDN prefill through packaged pybridge implementation");
}
