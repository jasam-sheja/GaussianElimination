INC = -I. -Isrc

all: \
	bin/gaussian_elimination_pref1.app \
	bin/gaussian_elimination_test1.app \
	bin/gaussian_elimination_test2.app \
	bin/speedup.app \
	bin/flops.app

.PHONY: clean makedirs
makedirs:
	mkdir bin

bin/speedup.app: app/speedup.cpp makedirs
	mpiCC -o bin/speedup.app app/speedup.cpp $(INC)
	
bin/flops.app: app/flops.cpp makedirs
	mpiCC -o bin/flops.app app/flops.cpp $(INC)

bin/gaussian_elimination_pref1.app: pref/gaussian_elimination_pref1.cpp makedirs
	mpiCC -o bin/gaussian_elimination_pref1.app pref/gaussian_elimination_pref1.cpp $(INC)

bin/gaussian_elimination_test1.app: test/gaussian_elimination_test1.cpp makedirs
	mpiCC -o bin/gaussian_elimination_test1.app test/gaussian_elimination_test1.cpp $(INC)

bin/gaussian_elimination_test2.app: test/gaussian_elimination_test2.cpp makedirs
	mpiCC -o bin/gaussian_elimination_test2.app test/gaussian_elimination_test2.cpp $(INC)


clean:
	rm bin/*.app
	rm bin/*.out

