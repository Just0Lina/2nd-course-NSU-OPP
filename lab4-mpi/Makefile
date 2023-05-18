np=5
SRCE = main.cpp
build:
	mpicxx -o mpi_jacobi $(SRCE) -lm -Wall

run:clean build
	mpirun -np $(np) ./mpi_jacobi

clean:
	rm -rf mpi_jacobi