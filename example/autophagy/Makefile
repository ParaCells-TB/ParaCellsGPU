all: executable

executable: clean bin build
	nvcc bin/Base.cu.o bin/BuiltInKernels.cu.o bin/Cell.cu.o bin/CellEnv.cu.o bin/Environment.cu.o bin/Identifiers.cu.o bin/Main.cu.o bin/ParaCellsError.cu.o -o bin/run
	@echo "ParaCells has been compiled successfully."

bin:
	@mkdir -p bin

build: bin/Base.cu.o bin/BuiltInKernels.cu.o bin/Cell.cu.o bin/CellEnv.cu.o bin/Environment.cu.o bin/Identifiers.cu.o bin/Main.cu.o bin/ParaCellsError.cu.o

bin/Base.cu.o:
	nvcc -dc Base.cu -o bin/Base.cu.o

bin/BuiltInKernels.cu.o:
	nvcc -dc BuiltInKernels.cu -o bin/BuiltInKernels.cu.o

bin/Cell.cu.o:
	nvcc -dc Cell.cu -o bin/Cell.cu.o

bin/CellEnv.cu.o:
	nvcc -dc CellEnv.cu -o bin/CellEnv.cu.o

bin/Environment.cu.o:
	nvcc -dc Environment.cu -o bin/Environment.cu.o

bin/Identifiers.cu.o:
	nvcc -dc Identifiers.cu -o bin/Identifiers.cu.o

bin/Main.cu.o:
	nvcc -dc Main.cu -o bin/Main.cu.o

bin/ParaCellsError.cu.o:
	nvcc -dc ParaCellsError.cu -o bin/ParaCellsError.cu.o

run:
	./bin/run

clean:
	@rm -f -r bin
