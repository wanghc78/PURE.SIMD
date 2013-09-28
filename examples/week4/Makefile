
CFLAGS = -msse4.2 -g -O2
CC = gcc

TARGET=RGB2Gray DotProd

all:${TARGET}

%:%.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f ${TARGET} *.exe
