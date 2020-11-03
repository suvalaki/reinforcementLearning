#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <iostream>
#include <iomanip>
#include "bandit.hpp"

class NormalMultiArmedBanditWrapper : public NormalPriorNormalMultiArmedBandit<double>
{
public:
    PyObject_HEAD
    NormalMultiArmedBanditWrapper() = default;
};



void run_bandits(std::size_t n_bandits, std::size_t n_samples)
{
    // setup bandits
    std::minstd_rand generator;
    NormalPriorNormalMultiArmedBandit<double> bandits(n_bandits, generator);

    for (size_t i = 0; i < n_samples; i++)
    {
        if (i == 0)
        {
            std::cout << "Generatimng Ranom Numbers"
                      << " with " << n_bandits << " bandits and " << n_samples << " samples\n";
        }
        else
        {
            std::cout << "\n";
        }
        std::vector<double> result = bandits.sample(generator);
        for (auto const &v : result)
        {
            std::cout << std::setw(9) << v << " ";
        }
    }
    std::cout << "\n";
}

/**
 * Create a function which we can use to call the multi armed bandit 
 * from python and returns null on error
 */
static PyObject *
bandit_multi_armed_bandit(PyObject *self, PyObject *args)
{
    const char *command;
    Py_ssize_t n_bandits;
    Py_ssize_t n_samples;
    int sts;

    if (not PyArg_ParseTuple(args, "nn", &n_bandits, &n_samples))
        return NULL;

    run_bandits(
        static_cast<std::size_t>(n_bandits),
        static_cast<std::size_t>(n_samples));

    Py_RETURN_NONE;
}

static PyMethodDef BanditMethods[] = {
    {"run_multi_armed_bandit", bandit_multi_armed_bandit, METH_VARARGS,
     "Execute a shell command."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef banditmodule = {
    PyModuleDef_HEAD_INIT,
    "bandit", /* name of module */
    NULL,     /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    BanditMethods};

PyMODINIT_FUNC
PyInit_bandit(void)
{
    return PyModule_Create(&banditmodule);
}

int main(int argc, char *argv[])
{
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL)
    {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    /* Add a built-in module, before Py_Initialize */
    if (PyImport_AppendInittab("bandit", PyInit_bandit) == -1)
    {
        fprintf(stderr, "Error: could not extend in-built modules table\n");
        exit(1);
    }

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(program);

    /* Initialize the Python interpreter.  Required.
       If this step fails, it will be a fatal error. */
    Py_Initialize();

    /* Optionally import the module; alternatively,
       import can be deferred until the embedded script
       imports it. */

    PyMem_RawFree(program);
    return 0;
}