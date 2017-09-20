CXX ?= g++
CC ?= gcc
CFLAGS = -Wall -Wconversion -O3 -fPIC -pthread -fopenmp
LIBS += -lgomp 
#INCLUDEPATH += -I  ../../3rdparty/eigen3.1.3  -I ../core/include

all: train infer 


train: l1_minimization.o train.o 
	$(CXX) $(CFLAGS) $(INCLUDEPATH) -o train train.o  l1_minimization.o $(LIBS)

infer: l1_minimization.o infer.o
	$(CXX) $(CFLAGS) $(INCLUDEPATH) -o infer infer.o  l1_minimization.o $(LIBS)
	
train.o: src/train.cpp src/l1_minimization.h
	$(CXX) -c $(CFLAGS) $(INCLUDEPATH) -o  train.o src/train.cpp  $(LIBS)

infer.o: src/infer.cpp src/l1_minimization.h	
	$(CXX) -c $(CFLAGS) $(INCLUDEPATH) -o  infer.o src/infer.cpp  $(LIBS)


l1_minimization.o: src/l1_minimization.cpp src/l1_minimization.h
	$(CXX) -c $(CFLAGS)  $(INCLUDEPATH) -o l1_minimization.o src/l1_minimization.cpp

clean1:
	rm -f *~  l1_minimization.o train.o infer.o 

clean:
	rm -f *~  l1_minimization.o train.o infer.o train  infer
