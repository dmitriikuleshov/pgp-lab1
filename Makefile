NVCC := /usr/local/cuda/bin/nvcc
NVCC_FLAGS := -Werror cross-execution-space-call -lm -O2

SRC_DIR := src
BUILD_DIR := build
REPORT_DIR := report

TARGETS := \
	$(BUILD_DIR)/lab1 \
	$(BUILD_DIR)/benchmark \
	$(BUILD_DIR)/properties

.PHONY: all clean report benchmark

all: $(TARGETS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/lab1: $(SRC_DIR)/lab1.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(BUILD_DIR)/benchmark: $(SRC_DIR)/benchmark.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(BUILD_DIR)/properties: $(SRC_DIR)/properties.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

benchmark: $(BUILD_DIR)/benchmark
	$(BUILD_DIR)/benchmark > $(REPORT_DIR)/benchmark_results.txt

report:
	typst compile --font-path /mnt/c/Windows/Fonts \
		$(REPORT_DIR)/report.typ \
		$(REPORT_DIR)/report.pdf

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(REPORT_DIR)/report.pdf
	rm -f $(REPORT_DIR)/benchmark_results.txt