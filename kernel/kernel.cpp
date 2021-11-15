#include <cassert>
#include <iostream>
#include <limits>

#include "kernel.h"

#include <omp.h>

using std::cout;
using std::endl;

int THD_COUNT = 1;

using std::string;

void normalize(csr_t* snaph, array2d_t<float> & matrix) {
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0 ; i < matrix.row_count ; i++) {
            vid_t degree = snaph->get_degree(i);
            matrix.row_normalize(i, degree+1);
        }
    }
}

void matrix_mul(csr_t* snaph, array2d_t<float> & input, array2d_t<float> & output) {
    int64_t row_count = output.row_count;
    int64_t col_count = output.col_count;
    vid_t* offset = snaph->offset;
    vid_t* nebrs = snaph->nebrs;

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0 ; i < row_count ; i++) {
            // Add its own features
            output.row_add(input.data_ptr + i*col_count, i);

            // Aggregate the features of its neighbors.
            for (int j = offset[i] ; j < offset[i+1] ; j++) {
                output.row_add(input.data_ptr + nebrs[j]*col_count, nebrs[j]);
            }
        }
    }
}


void _gspmm(csr_t* snaph, array2d_t<float> & input, array2d_t<float> & output, 
                     op_t op, bool reverse, bool norm /*= true*/)
{
    // snaph is the original graph
    // V_count is 19717, 19717 rows in input and output
    // op = eSUM
    // here we need to do a graph sparse matrix multiplication
    // the op will be summation as indicated by 'op' variable.

    // Formula: A*H*W. A: adjacency matrix, H: feature, W: weights. input = H*W, A is the csr representation

    //If in backward, normalize it first, else normalize it after computation
    
    //The core logic goes here. 

    if (reverse) {
        // Backward, normalize first
        normalize(snaph, input);
        // Calculate A*input
        matrix_mul(snaph, input, output);
    } else {
        // Calculate A*input
        matrix_mul(snaph, input, output);  
        // Normalize
        normalize(snaph, output);
    }
}

void invoke_gspmm(graph_t& graph, array2d_t<float> & input_array, array2d_t<float> & output_array,
                 bool reverse, bool norm /*= true*/)
{
    if (reverse) {
        return _gspmm(&graph.csr, input_array, output_array, eSUM, reverse, norm);
    } else {
        return _gspmm(&graph.csc, input_array, output_array, eSUM, reverse, norm);
    }
}