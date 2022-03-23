CC = mpicc

CFLAGS = -g -Wall #-D_CC_OVERLAP

LDFLAGS = -lm

POISSOBJS = decomp1d.o jacobi.o
#POISSOBJS = decomp1d.o jacobi.o gfunc.o

EXECS = assignment2_part1 assignment2_part2 assignment2_part4

all: $(EXECS)

assignment2_part1: assignment2_part1.o  $(POISSOBJS) 
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)


assignment2_part2: assignment2_part2.o  $(POISSOBJS) 
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)


assignment2_part4: assignment2_part4.o  $(POISSOBJS) 
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# tmpedc1d: testdc1d.o decomp1d.o

## tests

tags:
	etags *.c *.h

.PHONY: clean tags tests

clean:
	$(RM) *.o $(EXECS) $(TESTS) TAGS tags
