#include "nn.h"

#include <math.h>
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>


Network nn_alloc(size_t *nn_layout, size_t nn_layout_size) 
{ 
    Network nn;
    nn.layers_count = nn_layout_size;
    nn.layers_sizes = calloc(nn_layout_size, sizeof(*nn.layers_sizes));
    assert(nn.layers_sizes != NULL);
    // Set layers sizes
    for(size_t l = 0; l < nn_layout_size; l++) {
        nn.layers_sizes[l] = nn_layout[l]; 
    }
    // Allocate neurons, weights and biases
    nn.ns = malloc(sizeof(*nn.ns)*nn_layout_size);     //
    nn.ws = malloc(sizeof(*nn.ws)*(nn_layout_size-1));   // -1 is because nn_layout_size includes inputs
    nn.bs = malloc(sizeof(*nn.bs)*(nn_layout_size-1));   //
    for(size_t l = 0; l < nn_layout_size-1; l++) {
        nn.ns[l] = row_alloc(nn_layout[l]);
        nn.ws[l] = mat_alloc(nn_layout[l], nn_layout[l+1]);
        nn.bs[l] = row_alloc(nn_layout[l+1]);
    }    
    nn_rand_weights_biases(nn, -1.0f, 1.0f);
    nn.ns[nn_layout_size-1] = row_alloc(nn_layout[nn_layout_size-1]);
    return nn;
}
void nn_free(Network nn) 
{
    assert(nn.layers_sizes != NULL);
    for(size_t l = 0; l < nn.layers_count; l++) {
        row_free(nn.ns[l]);

        if(l < nn.layers_count-1) {
            mat_free(nn.ws[l]);
            row_free(nn.bs[l]);
        }
    }
    free(nn.ns);
    free(nn.ws);
    free(nn.bs);
    free(nn.layers_sizes);
}
void nn_rand_weights_biases(Network nn, float min, float max)
{
    for(size_t l = 0; l < nn.layers_count-1; l++) {
        mat_rand(nn.ws[l], min, max);
        row_rand(nn.bs[l], min, max);
    }    
}
static inline float sigmoid(float x)
{
    return 1.f / (1.f + expf(-x));
}
static inline float activation_func(float x) {
    return sigmoid(x);
}
void nn_activ_func(Mat m)
{
    for(size_t row = 0; row < m.rows; row++) {
        for(size_t col = 0; col < m.cols; col++) {
            MAT_AT(m, row, col) = activation_func(MAT_AT(m, row, col));
        }
    }
}
// Indexing looks weird because 1 additional neurons layer representing input layer
// so the neurons array is longer than weights and biases arrays by 1 
// thus the weights, biases and deltas look like they are from previous layer
void nn_forward(Network nn, Row input_data, bool *should_stop)
{
    if(!nn.layers_sizes 
    || !nn.ns
    || !nn.ws
    || !nn.bs) {
        fprintf(stderr, "Invalid nn\n");
        return;
    }
    if(!input_data.data) {
        fprintf(stderr, "Invalid nn forward data\n");
        return;
    }
    if(*should_stop) return;
    for(size_t data_num = 0; data_num < input_data.cols; data_num++) {
        ROW_AT(nn.ns[0], data_num) = ROW_AT(input_data, data_num);
    }
    size_t last_layer = nn.layers_count-1;
    for(size_t l = 0; l < last_layer; l++) {
        if(*should_stop) return;
        mat_dot(row_as_mat(nn.ns[l+1]), row_as_mat(nn.ns[l]), nn.ws[l]);
        mat_sum(row_as_mat(nn.ns[l+1]), row_as_mat(nn.bs[l]));
        nn_activ_func(row_as_mat(nn.ns[l+1]));
    }
}
// Indexing looks weird because 1 additional neurons layer representing input layer
// so the neurons array is longer than deltas, weights and biases arrays by 1
// thus the weights, biases and deltas look like they are from previous layer
void nn_backpropagation(Network nn,
                     Row target_output, 
                     Mat *dW,
                     Row *dB,
                     bool *should_stop) {
    if(!nn.layers_sizes) {
        fprintf(stderr, "Invalid nn\n");
        return;
    }
    if(target_output.cols != nn.layers_sizes[nn.layers_count-1]) {
        fprintf(stderr, "Bad target data dimensions!\n");
        return;
    }
    if(*should_stop) return;
    Row *delta = malloc(sizeof(*delta)*(nn.layers_count-1));
    size_t last_layer = nn.layers_count-1;
    for(size_t l = 0; l < last_layer; l++) {
        // l+1 cuz delta size based on neurons layer size
        delta[l] = row_alloc(nn.layers_sizes[l+1]);
    }
    for(size_t col = 0; col < target_output.cols; col++) {
        float A = (ROW_AT(nn.ns[last_layer], col));
        ROW_AT(delta[last_layer-1], col)   // delta[l] = (Al - y) * Al * (1 - Al)
            = (A
            - ROW_AT(target_output, col))
            * A * (1 - A);
    } 
    for(ssize_t l = (ssize_t)last_layer-1; l >= 0; l--) { 
        size_t current_layer = nn.layers_sizes[l];
        size_t next_layer = nn.layers_sizes[l+1];
        // Weights
        for(size_t row = 0; row < current_layer; row++) {
            if(*should_stop) goto nn_backpropagation_exit;
            for(size_t col = 0; col < next_layer; col++) {
                MAT_AT(dW[l], row, col) // delta Wl = delta[l] * Al-1
                    = ROW_AT(delta[l], col) 
                    * ROW_AT(nn.ns[l], row);
            }
        }
        // Biases
        for(size_t col = 0; col < next_layer; col++) {
            ROW_AT(dB[l], col) // delta bl = delta[l]
                = ROW_AT(delta[l], col);
        }
        // Deltas for prev layer if not last layer
        if(l > 0) {
            // delta[l]-1 = delta[l] * Wl * Al-1 * (1 - Al-1)
            for(size_t row = 0; row < current_layer; row++) {
                if(*should_stop) goto nn_backpropagation_exit;
                float weightDeltaSum = 0.f;
                for(size_t col = 0; col < next_layer; col++) {
                    weightDeltaSum 
                        += ROW_AT(delta[l], col) * MAT_AT(nn.ws[l], row, col);
                }
                float Aprev = ROW_AT(nn.ns[l], row);
                ROW_AT(delta[l-1], row) 
                    = weightDeltaSum * Aprev * (1.f - Aprev);
            }
        }
    }
nn_backpropagation_exit:
    for(size_t l = 0; l < last_layer; l++)
        row_free(delta[l]);
    free(delta);
}
// Indexing looks weird because 1 additional neurons layer representing input layer
// so the neurons array is longer than deltas, weights and biases arrays by 1
// thus the weights, biases and deltas look like they are from previous layer
void train(Network nn,
           float learning_rate,
           Mat input_data,
           Mat target_output,
           size_t epoch_count,
           bool *should_stop)
{
    Mat *dWsum = malloc(sizeof(*dWsum)*(nn.layers_count-1));
    Row *dBsum = malloc(sizeof(*dBsum)*(nn.layers_count-1));

    Mat *dW = malloc(sizeof(*dW)*(nn.layers_count-1));
    Row *dB = malloc(sizeof(*dB)*(nn.layers_count-1));
    size_t last_layer = nn.layers_count-1;
    for(size_t l = 0; l < last_layer; l++) {
        dW[l] = mat_alloc(nn.layers_sizes[l], nn.layers_sizes[l+1]);
        dB[l] = row_alloc(nn.layers_sizes[l+1]);

        dWsum[l] = mat_alloc(nn.layers_sizes[l], nn.layers_sizes[l+1]);
        dBsum[l] = row_alloc(nn.layers_sizes[l+1]);
    }
#ifdef SHOWLOSS
    size_t lossFreq = (epoch_count+9) / 10;
#endif
    for(size_t epoch = 0; epoch < epoch_count; epoch++) {
#ifdef SHOWLOSS
        float lossVal = 0;
#endif
        size_t data_size = input_data.rows;
        for(size_t data_num = 0; data_num < data_size; data_num++) {
            nn_forward(nn, mat_row(input_data, data_num), should_stop);
            nn_backpropagation(nn,
                            mat_row(target_output, data_num),
                            dW,
                            dB,
                            should_stop); 
            for(size_t l = 0; l < last_layer; l++) {
                if(*should_stop) goto nn_train_exit;
                mat_sum(dWsum[l], dW[l]);
                mat_sum(row_as_mat(dBsum[l]), row_as_mat(dB[l]));
            }
#ifdef SHOWLOSS
            lossVal += loss(NN_OUTPUT(nn), mat_row(target_output, data_num));
#endif
        }
#ifdef SHOWLOSS
        lossVal /= (float)data_size;
        if(epoch % lossFreq == 0) {
            printf("%f\n", lossVal);
        }
#endif
        // Adjust weights and biases
        for(size_t l = 0; l < last_layer; l++) {
            size_t current_layer = nn.layers_sizes[l];
            size_t next_layer = nn.layers_sizes[l+1];
            for(size_t row = 0; row < current_layer; row++) {
                if(*should_stop) goto nn_train_exit;
                for(size_t col = 0; col < next_layer; col++) {
                    MAT_AT(nn.ws[l], row, col)
                        -= learning_rate * MAT_AT(dWsum[l], row, col)/(float)data_size;
                }
            }
            for(size_t col = 0; col < next_layer; col++) {
                MAT_AT(nn.bs[l], 0, col)
                    -= learning_rate * MAT_AT(dBsum[l], 0, col)/(float)data_size;
            }
            mat_zero(dWsum[l]);
            row_zero(dBsum[l]);
        }
    }
nn_train_exit:
    for(size_t l = 0; l < last_layer; l++) {
        mat_free(dW[l]);
        row_free(dB[l]);
        mat_free(dWsum[l]);
        row_free(dBsum[l]);
    }

    free(dW);
    free(dB);
    free(dWsum);
    free(dBsum);
}
