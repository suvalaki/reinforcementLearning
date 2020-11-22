TARGET_EXEC ?= a.out

PYTHON_FLAGS ?= python3-config --includes --ldflags

main:./bandit.hpp ./bandit_run.cpp
	g++ ./bandit_run.cpp -o ./bandit_run.out

rmain: ./bandit_run.out 
	./bandit_run.out 10 5

PYTHON_VERSION := 3.8
PYTHON_PATH := /home/david/.cache/pypoetry/virtualenvs/reinfocementlearning-dMkvu5-V-py3.8/bin/python
#PYTHON_INC := /home/david/miniconda3/include/python$(PYTHON_VERSION)
PYTHON_INC := /usr/include/python$(PYTHON_VERSION)
BOOST_INC := /usr/include/
BOOST_LIB := /usr/lib
BOOST_PYTHON_V :=38


CFLAGS = --std=c++11

TARGET := bandit_python_object

$(TARGET).so: $(TARGET).o
	g++ -shared -Wl,--export-dynamic $(TARGET).o \
	-L$(BOOST_LIB) \
	-L/usr/lib/python$(PYTHON_VERSION)/config \
	-lpython$(PYTHON_VERSION) \
	-lboost_python$(BOOST_PYTHON_V) \
	-o $(TARGET).so $(CFLAGS)

$(TARGET).o: $(TARGET).cpp
	g++ -I$(PYTHON_INC) -I$(BOOST_INC) -fPIC -c $(TARGET).cpp $(CFLAGS)

all: $(TARGET).so