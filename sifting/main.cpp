#include "Sifting.h"


PYBIND11_MODULE(sifting, m) {
    m.def("calc", &calc);
    m.def("count", &count);
    m.def("clustering", &clustering);

    m.def("vector_as_array_nocopy", &vector_as_array_nocopy,
        "Returns a vector of 16-bit ints as a NumPy array without making a "
        "copy of the data");
};