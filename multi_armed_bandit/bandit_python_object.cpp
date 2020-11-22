#include <random>
#include "bandit.hpp"

#include <boost/python.hpp>

template<typename T>
inline
std::vector< T > py_list_to_std_vector( const boost::python::object& iterable )
{
    return std::vector< T >( boost::python::stl_input_iterator< T >( iterable ),
                             boost::python::stl_input_iterator< T >( ) );
}


// different from wrapping the iterator
template <class T>
inline
boost::python::list std_vector_to_py_list(std::vector<T> vector) {
    typename std::vector<T>::iterator iter;
    boost::python::list list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
        list.append(*iter);
    }
    return list;
}



void run_bandits(std::size_t n_bandits, std::size_t n_samples) {
  // setup bandits
  std::minstd_rand generator;
  NormalPriorNormalMultiArmedBandit<double> bandits(n_bandits, generator);

  for (size_t i = 0; i < n_samples; i++) {
    if (i == 0) {
      std::cout << "Generatimng Ranom Numbers"
                << " with " << n_bandits << " bandits and " << n_samples
                << " samples\n";
    } else {
      std::cout << "\n";
    }
    std::vector<double> result = bandits.sample(generator);
    for (auto const &v : result) {
      std::cout << std::setw(9) << v << " ";
    }
  }
  std::cout << "\n";
}


// Exporting class 


std::string say_hi(){
    return "hi\n";
}

class GenWrap { 

  public: 
  std::minstd_rand generator = std::minstd_rand();
  GenWrap() = default;

};

class PythonNormalMultiArmedBandit : 
  public GenWrap, 
  public NormalPriorNormalMultiArmedBandit<double> {

public:
  PythonNormalMultiArmedBandit(int n_bandits):
  GenWrap(),
  NormalPriorNormalMultiArmedBandit<double>(static_cast<std::size_t>(n_bandits), generator)
  {};
  boost::python::list sample() { 
    auto samp = NormalPriorNormalMultiArmedBandit<double>::sample(generator);
    return std_vector_to_py_list(samp);
  };
};


class TestClass {
  public: 
  int data = 0;
  TestClass(int data): data(data) {};
};


BOOST_PYTHON_MODULE(bandit_python_object)
{
    boost::python::def("hi", say_hi);
    boost::python::class_<PythonNormalMultiArmedBandit>("MultiArmedBandit", boost::python::init<int>())
      .def("sample", &PythonNormalMultiArmedBandit::sample)
    ;
    boost::python::class_<TestClass>("testCls", boost::python::init<int>());
}
