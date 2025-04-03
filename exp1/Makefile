openmp_example: openmp_example.c
	gcc -fopenmp -o openmp_example openmp_example.c
run: openmp_example
	./openmp_example ./random_matrix.bin
clean:
	rm openmp_example
peak:
	xxd -l 512 -p ./result_matrix.bin
cmp:
	cmp ./result_matrix.bin ./std_result_matrix.bin