# the source files are in src/ directory
# the header files are in include/ directory
# the main program remains in the current directory

# the compiler
CC := g++

# the compiler flags
CFLAGS := -Wall -g -Iinclude

# source files in src/ directory
SRC := $(filter-out src/tests.cpp src/main.cpp, $(wildcard src/*.cpp))

# object files in obj/ directory
OBJ := $(patsubst src/%.cpp,obj/%.o,$(SRC))

# the executable file
TARGET := final

# the default target
all: $(TARGET)

# the executable file depends on the object files
$(TARGET): $(OBJ) src/main.cpp
	$(CC) $(CFLAGS) -o $@ $^

tests: $(OBJ) src/tests.cpp
	$(CC) $(CFLAGS) -o $@ $^

# the object files depend on the source files
obj/%.o: src/%.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

# the clean target
clean:
	rm -f $(OBJ) $(TARGET) tests sample_test.txt

# the run target
run: $(TARGET)
	./$(TARGET)

# the debug target
debug: $(TARGET)
	gdb $(TARGET)

# the valgrind target
valgrind: $(TARGET)
	valgrind --leak-check=full ./$(TARGET)

# the cppcheck target
cppcheck:
	cppcheck --enable=all --inconclusive --std=c++11 --suppress=missingIncludeSystem $(SRC)
