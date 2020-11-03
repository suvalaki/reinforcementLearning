TARGET_EXEC ?= a.out

PYTHON_FLAGS ?= python3-config --includes --ldflags

main:./bandit.hpp ./bandit_run.cpp
	g++ ./bandit_run.cpp -o ./bandit_run.out

rmain: ./bandit_run.out 
	./bandit_run.out 10 5

python:./bandit.hpp 
	g++ ./bandit_python.cpp -o./bandit_python.out -I/home/david/miniconda3/include/python3.8 -I/home/david/miniconda3/include/python3.8 -L/home/david/miniconda3/lib/python3.8/config-3.8-x86_64-linux-gnu -L/home/david/miniconda3/lib  -lcrypt -lpthread -ldl  -lutil -lrt -lm -lm 
