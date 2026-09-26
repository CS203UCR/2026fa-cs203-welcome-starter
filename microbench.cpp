#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <unordered_set>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <map>
#include <stdlib.h>
#include <cmath>
#ifdef CUDA
extern "C" {
#include <cuda.h>
uint64_t *baseline_double_cuda(uint64_t * _vector, unsigned long int size);
    uint64_t* calculate_distance_matrix(uint64_t *matrix, unsigned long int size);
    uint64_t* calculate_distance_matrix_cuda(uint64_t *h_matrix, unsigned long int size);
}
#endif
#include "perfstats.h"

double start, stop;

#define CLINK extern "C"
#define OPT(a) __attribute__(a)

extern "C" uint64_t *__attribute__((noinline)) baseline_int(uint64_t * vector, unsigned long int size) {
    for(uint i= 0 ; i < size; i++) {
        vector[i] *= vector[i];
    }
    return vector;
}

extern "C" uint64_t *__attribute__ ((optimize(4))) baseline_int_O4 (uint64_t * vector, unsigned long int size) {
    for(uint i= 0 ; i < size; i++) {
        vector[i] *= vector[i];
    }
    return vector;
}

extern "C" uint64_t *baseline_double(uint64_t * _vector, unsigned long int size) {
    //double * vector = new double[size];
    double * vector = (double*)_vector;
    for(uint i= 0 ; i < size; i++) {
        vector[i] *= vector[i];
    }
    return (uint64_t*)vector;
}

extern "C" uint64_t *baseline_float(uint64_t * _vector, unsigned long int size) {
    //double * vector = new double[size];
    float * vector = (float*)_vector;
    for(uint i= 0 ; i < size; i++) {
        vector[i] *= vector[i];
    }
    return (uint64_t*)vector;
}

extern "C" uint64_t *baseline_char(uint64_t * _vector, unsigned long int size) {
    //double * vector = new double[size];
    char * vector = (char*)_vector;
    for(uint i= 0 ; i < size; i++) {
        vector[i] *= vector[i];
    }
    return (uint64_t*)vector;
}

extern "C" uint64_t *__attribute__ ((optimize(4),noinline)) baseline_double_O4(uint64_t * _vector, unsigned long int size) {
    //double * vector = new double[size];
    double * vector = (double*)_vector;
    for(uint i= 0 ; i < size; i++) {
        vector[i] *= vector[i];
    }
    return (uint64_t*)vector;
}


volatile int ROW_SIZE = 1024;
extern "C" uint64_t *__attribute__ ((optimize(4))) matrix_column_major(uint64_t * _vector, unsigned long int size) {
#define ROW_SIZE 1024
    double * vector = (double*)_vector;

    for (int k = 0; k < ROW_SIZE; k++) {
        for(uint i= 0 ; i < size/ROW_SIZE; i++) {
            vector[i*ROW_SIZE + k] = sqrt(vector[i*ROW_SIZE + k]); // This Line
        }
    }
//    std::cout << "Execution matrix_column_major complete\n";
    
    return (uint64_t*)vector;
}


//int ROW_SIZE = 1024;
extern "C" uint64_t *__attribute__ ((optimize(4))) matrix_row_major(uint64_t * _vector, unsigned long int size) {

    double * vector = (double*)_vector;

    for(uint i= 0; i < size/ROW_SIZE; i++) {
        for (int k = 0; k < ROW_SIZE; k++) {
            vector[i*ROW_SIZE + k] = sqrt(vector[i*ROW_SIZE + k]); // This Line
        }
    }
    
    return (uint64_t*)vector;
}

extern "C" uint64_t *__attribute__ ((optimize(0))) everything(uint64_t * vector, unsigned long int size) {
    matrix_column_major(vector, size);
    baseline_int(vector,size);
    return vector;
}

extern "C" uint64_t *__attribute__ ((optimize(0))) everything_opt(uint64_t * vector, unsigned long int size) {
    matrix_column_major(vector, size);
    baseline_int_O4(vector,size);
    return vector;
}

extern "C" uint64_t *__attribute__ ((optimize(0))) option_1(uint64_t * vector, unsigned long int size) {
    matrix_row_major(vector, size);
    baseline_int(vector,size);
    return vector;
}

extern "C" uint64_t *__attribute__ ((optimize(0))) option_2(uint64_t * vector, unsigned long int size) {
    matrix_column_major(vector, size);
    baseline_int_O4(vector,size);
    return vector;
}


uint vector_size;

int main(int argc, char *argv[])
{
    int i;
    int reps=1,freq;
    unsigned long int size=1024;
    char *stat_file = NULL;
    char default_filename[] = "stat.csv";
    char preamble[1024];
    char epilogue[1024];
    std::vector<std::string> functions;
    std::vector<std::string> default_functions;
    std::vector<unsigned long int> sizes;
    std::vector<unsigned long int> default_sizes;
    std::vector<int> frequencies;
    std::vector<int> default_frequencies;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] != '-') continue;

        char option = argv[i][1];

        // Helper lambda to consume subsequent non-flag arguments
        auto get_next_args = [&]() {
            std::vector<std::string> values;
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                values.push_back(argv[++i]);
            }
            return values;
        };

        switch (option) {
            case 'o':
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    stat_file = argv[++i];
                }
                break;

            case 'r':
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    reps = std::stoi(argv[++i]);
                }
                break;

            case 's': {
                for (const auto& val : get_next_args()) {
                    sizes.push_back(std::stoi(val));
                }
                break;
            }

            case 'M': {
                for (const auto& val : get_next_args()) {
                    frequencies.push_back(std::stoi(val));
                }
                break;
            }

            case 'f':
                functions = get_next_args();
                break;

            case 'h':
                // TODO: Print help menu
                break;
        }
    }

// Apply fallback defaults if containers or pointers are empty
    if (sizes.empty())       sizes = {1024 * 1024};
    if (functions.empty())   functions = {"baseline_int"};
    if (frequencies.empty()) frequencies = {4600};
    if (stat_file == nullptr) stat_file = default_filename;
    std::map<const std::string, uint64_t*(*)(uint64_t *, unsigned long int)>
    function_map =
    {
#define FUNCTION(n) {#n, n}
        FUNCTION(baseline_int),
        FUNCTION(baseline_int_O4),
        FUNCTION(baseline_double),
        FUNCTION(baseline_float),
        FUNCTION(baseline_char),
        FUNCTION(baseline_double_O4),
#ifdef CUDA
        FUNCTION(baseline_double_cuda),
        FUNCTION(calculate_distance_matrix_cuda),
        FUNCTION(calculate_distance_matrix),
#endif
        FUNCTION(matrix_row_major),
        FUNCTION(matrix_column_major),
        FUNCTION(everything),
        FUNCTION(option_1),
        FUNCTION(option_2),
        FUNCTION(everything_opt)
    };
    vector_size  = *std::max_element(sizes.begin(), sizes.end())+256;
    uint64_t * vector = new uint64_t[vector_size]; // and 256 because we matrix_* needs a little extra space
    char header[]="size,rep,function,IC,Cycles,CPI,MHz,CT,ET,cmdlineMHz";

    perfstats_print_header(stat_file, header);
    for(uint i = 0; i < vector_size; i++) {
        uint64_t bit;
        uint64_t temp = 0xdeadbeef+i;
        bit = ((temp >> 0) ^ (temp >> 1) ^ (temp >> 3) ^ (temp >> 4)) & 0x1llu;
        vector[i] = (temp >> 1) | (bit << 63);
    }
    for(auto & freq: frequencies ) {
//        change_cpufrequnecy(freq);
        for(auto & size: sizes ) {
            for(uint r = 0; r < reps; r++) {
                for(auto & function : functions) {
                        sprintf(preamble, "%lu,%d,%s,", size, r, function.c_str());
                        perfstats_init();
                        perfstats_enable();
                        function_map[function](vector, size);               
                        perfstats_disable();
                        sprintf(epilogue, ",%d\n",freq);
//                        perfstats_print(preamble, stat_file, epilogue);
                        perfstats_deinit();
                    }                                
                }
            }
//        restore_cpufrequnecy();
        }
    std::cout << "Execution complete\n" ;
    return 0;
}
