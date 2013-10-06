#
#  Makefile for fltk applications
#


LIBS = 

CC = g++ -std=c++0x

.SUFFIXES: .o .cpp .cxx

.cpp.o: 
	$(CC) -c -g -o $*.o $<

ALL.O = Dijkstra.o

sssp: $(ALL.O)
	$(CC) -g -o $@ $(ALL.O) $(LIBS)

clean:  
	rm -f $(ALL.O)

clean_all:
	rm -f $(ALL.O) SSSP_Week1 
