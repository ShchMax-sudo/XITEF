#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;


template <typename Sequence>
inline py::array_t<typename Sequence::value_type> as_pyarray(Sequence &&seq) {
  auto size = seq.size();
  auto data = seq.data();
  std::unique_ptr<Sequence> seq_ptr =
      std::make_unique<Sequence>(std::move(seq));
  auto capsule = py::capsule(seq_ptr.get(), [](void *p) {
    std::unique_ptr<Sequence>(reinterpret_cast<Sequence *>(p));
  });
  seq_ptr.release();
  return py::array(size, data, capsule);
}

// template <typename T>
// static py::array_t<T> vector_as_array(std::vector<T> vec) {
static py::array_t<int> vector_as_array(std::vector<int> &vec) {
  return py::array(vec.size(), vec.data());
}

// template <typename T>
// static py::array_t<T> vector_as_array_nocopy(std::vector<T> vec) {
static py::array_t<int> vector_as_array_nocopy(std::vector<int> &vec) {
  return as_pyarray(std::move(vec));
}

py::array_t<int> calc(std::vector<double>&, std::vector<double>&, std::vector<double>&, int, int, int, std::vector<bool>&, double);
py::array_t<int> count(std::vector<double> &, std::vector<double> &, std::vector<double> &, std::vector<double> &, std::vector<double> &, std::vector<double> &, std::vector<double> &, double, double, double, double);
py::array_t<int> clustering(std::vector<double> &x, std::vector<double> &y, std::vector<double> &time, std::vector<double> &prob, int localPsf, int timeCoeff);