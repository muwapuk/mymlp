#ifndef NN_GPU_H
#define NN_GPU_H

#include "nn.h"

void nn_forward_gpu(Network nn, Row input_data, bool *should_stop);
void nn_backpropagation_gpu(Network nn,
                     Row target_output, 
                     Mat *dw,
                     Row *db,
                     bool *should_stop);
void train_gpu(Network nn,
           float learning_rate,
           Mat input_data,
           Mat target_output,
           size_t epoch_count,
           bool *should_stop);

#endif
