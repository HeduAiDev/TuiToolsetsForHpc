#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "tui_tool_sets_runable.hpp"

using namespace tui::runable;
namespace py = pybind11;

template<typename T>
void PyPrint_matrix_(py::array_t<T> xs) {
    py::buffer_info info = xs.request();
    auto ptr = static_cast<T *>(info.ptr);
    if (info.ndim != 2) {
        throw std::runtime_error("only 2d arrays are supported");
    }
    int rows = info.shape[0];
    int cols = info.shape[1];

    print_matrix(ptr, rows, cols);
}

template<typename T>
void PyDiff_(py::array_t<T> a, py::array_t<T> b, double accuracy = 1e-3) {
    py::buffer_info info_a = a.request();
    py::buffer_info info_b = b.request();
    auto ptr_a = static_cast<T *>(info_a.ptr);
    auto ptr_b = static_cast<T *>(info_b.ptr);
    
    if (info_a.ndim != info_b.ndim || info_a.ndim != 2) {
        throw std::runtime_error("only 2d arrays are supported");
    }

    if (info_a.shape[0] != info_b.shape[0] || info_a.shape[1] != info_b.shape[1]) {
        throw std::runtime_error("matrix a and b must have the same shape");
    }

    int rows = info_a.shape[0];
    int cols = info_a.shape[1];
    diff(ptr_a, ptr_b, rows, cols, accuracy);
}


PYBIND11_MODULE(_C, m) {
    m.doc() = "A set of useful TUI tools ";
    m.def("print_matrix_double", &PyPrint_matrix_<double>);
    m.def("print_matrix_float", &PyPrint_matrix_<float>);
    #ifdef __CUDA__
    m.def("print_matrix_half", &PyPrint_matrix_<half>);
    #endif
    m.def("diff_double", &PyDiff_<double>);
    m.def("diff_float", &PyDiff_<float>);
    #ifdef __CUDA__
    m.def("diff_half", &PyDiff_<half>);
    #endif
}