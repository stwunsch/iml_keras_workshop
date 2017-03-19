CC      = g++
CFLAGS  = -Ilwtnn/include
LDFLAGS = -Llwtnn/lib -llwtnn

all: apply

apply: apply.o
	$(CC) -o $@ $^ $(LDFLAGS)

apply.o: apply.cpp
	$(CC) -c $(CFLAGS) $<

clean:
	rm -f apply.o apply
