CXX = g++
CPPFLAGS = -lgtest -std=c++17
CHECKFLAGS = -Wall -Werror -Wextra

all: lab2

lab2: first second

second:

slow: clean 
	g++  lab2/slow/slow.cpp lab2/slow/matrix.cpp -o openMp
	./openMp > lab2/slow/slow.txt

fast: clean
	g++ -fopenmp lab2/fast/fast.cpp lab2/fast/matrix.cpp -o openMp
	sh ./bash.sh


secfast: clean
	g++ -fopenmp lab2/second_fast/fast.cpp lab2/second_fast/matrix.cpp -o openMp
	./openMp > lab2/second_fast/fast.txt



# matrix.o: matrix.cpp
# 	@${CXX} -fopenmp  -std=c++17 ${CHECKFLAGS} -c $^

# matrix.a: matrix.o
# 	@ar rcs $@ $^
# 	@ranlib $@

clean: 
	rm -rf matrix.a matrix.o openMp


