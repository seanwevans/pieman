CC = gcc
CFLAGS = -O3 -Wall -Wextra -mavx -mfma
LDFLAGS = -lm
TARGET = nnet
SRCS = main.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)


test:
	bash tests/test.sh
