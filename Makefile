CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -fPIC
LDFLAGS = -shared
TARGET = libpolygrav.so
SRC_DIR = src
SRC = $(wildcard $(SRC_DIR)/*.c)
OBJ = $(patsubst $(SRC_DIR)/%.c,%.o,$(SRC))

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

$(OBJ): $(SRC)
	$(CC) $(CFLAGS) -c $^

clean:
	rm -f $(OBJ) $(TARGET)
