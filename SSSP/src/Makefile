#
#  Makefile for fltk applications
#


LIBS = 
CC = g++ -O3 -fopenmp -lpthread

.SUFFIXES: .o .cpp .cxx

.cpp.o: 
	$(CC) -c -g -o $*.o $<

ALL.O = SSSP.o

sssp: $(ALL.O)
	$(CC) -g -o $@ $(ALL.O) $(LIBS)

clean:  
	rm -f $(ALL.O)

clean_all:
	rm -f $(ALL.O) sssp 
