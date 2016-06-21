all: 
	g++ -ggdb `pkg-config --cflags opencv` objrec.cpp `pkg-config --libs opencv` -std=c++11 -o objrec
