# Laboratory Work No. 1

## Project Structure

```text
lab1/
├── src/          source code
├── build/        compiled programs
├── report/       report and template
├── results/      benchmark results
├── signatures/   digital signatures
├── Makefile
└── README.md
```

## Building the Programs

```bash
make all
```

This builds:

```text
build/lab1
build/benchmark
build/properties
```

## Building the Report

Install Typst if needed:

```bash
sudo snap install typst
```

Build the report:

```bash
make report
```

## Running the Benchmark

```bash
make benchmark
```

The results are saved to:

```text
report/benchmark_results.txt
```

## Cleaning

```bash
make clean
```