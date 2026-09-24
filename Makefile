# Four variants of the same sources: x86-64 (AVX kernels, double precision) and
# x86-32 (SSE kernels, single precision), each with and without OpenMP.
#
#   make            all four binaries in bin/
#   make x64        only the 64-bit ones (the 32-bit ones need multilib, see README)
#   make test       compare the results with scikit-learn
#   make bench      run the benchmarks behind the report, results in results/
#   make report     build the report, docs/relazione.pdf

CC     ?= gcc
NASM   ?= nasm
PYTHON ?= python3
OPT    ?= -O2
BIN    ?= bin
BUILD  ?= build

CFLAGS   = -std=gnu11 $(OPT) -Wall -Wextra
CFLAGS64 = -m64 -mavx
CFLAGS32 = -m32 -msse3 -mfpmath=sse
NO_OMP   = -Wno-unknown-pragmas
LDLIBS   = -lm

SRC := $(wildcard src/*.c)
HDR := $(wildcard src/*.h)

.PHONY: all x64 x86 exam test bench report figures clean

all: x64 x86
x64: $(BIN)/dbscan64 $(BIN)/dbscan64-omp
x86: $(BIN)/dbscan32 $(BIN)/dbscan32-omp

$(BUILD) $(BIN):
	mkdir -p $@

$(BUILD)/kernels64.o: src/asm/kernels64.nasm | $(BUILD)
	$(NASM) -f elf64 $< -o $@

$(BUILD)/kernels32.o: src/asm/kernels32.nasm | $(BUILD)
	$(NASM) -f elf32 $< -o $@

$(BIN)/dbscan64: $(SRC) $(HDR) $(BUILD)/kernels64.o | $(BIN)
	$(CC) $(CFLAGS) $(CFLAGS64) $(NO_OMP) $(SRC) $(BUILD)/kernels64.o -o $@ $(LDLIBS)

$(BIN)/dbscan64-omp: $(SRC) $(HDR) $(BUILD)/kernels64.o | $(BIN)
	$(CC) $(CFLAGS) $(CFLAGS64) -fopenmp $(SRC) $(BUILD)/kernels64.o -o $@ $(LDLIBS)

$(BIN)/dbscan32: $(SRC) $(HDR) $(BUILD)/kernels32.o | $(BIN)
	$(CC) $(CFLAGS) $(CFLAGS32) $(NO_OMP) $(SRC) $(BUILD)/kernels32.o -o $@ $(LDLIBS)

$(BIN)/dbscan32-omp: $(SRC) $(HDR) $(BUILD)/kernels32.o | $(BIN)
	$(CC) $(CFLAGS) $(CFLAGS32) -fopenmp $(SRC) $(BUILD)/kernels32.o -o $@ $(LDLIBS)

# The version delivered for the exam (git tag exam-2024), built as the course
# scripts did, to compare it with the current one in the benchmarks
EXAM := $(BUILD)/exam/No_OpenMP/64_bit

exam: $(BIN)/dbscan64-exam

$(BIN)/dbscan64-exam: | $(BUILD) $(BIN)
	rm -rf $(BUILD)/exam && mkdir -p $(BUILD)/exam
	git archive exam-2024 No_OpenMP/64_bit | tar -x -C $(BUILD)/exam
	cd $(EXAM) && $(NASM) -f elf64 sseutils64.nasm && $(NASM) -f elf64 cfs64.nasm
	$(CC) -m64 -msse -mavx -O0 -no-pie $(EXAM)/sseutils64.o $(EXAM)/cfs64.o $(EXAM)/cfs64c.c -o $@ -lm

# Tests every binary that has been built
test: x64
	$(PYTHON) tests/run_tests.py $(filter-out %-exam,$(wildcard $(BIN)/dbscan*))

bench: x64 exam
	$(PYTHON) scripts/benchmark.py

figures:
	$(MAKE) -C docs/diagrams

# Tables and charts are regenerated from results/, so `make bench report` updates every number
report: figures
	$(PYTHON) scripts/report_data.py
	cd docs/relazione && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	cp docs/relazione/main.pdf docs/relazione.pdf

clean:
	rm -rf $(BUILD) $(BIN)
