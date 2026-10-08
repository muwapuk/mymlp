#ifndef NN_GPU_H
#define NN_GPU_H

#include "nn.h"

#define NN_FORWARD_SHADER_PATH "./nn_forward.comp"
#define NN_TRAIN_SHADER_SRC "./nn_train.comp"

unsigned int nn_train_program;

void init_nn_gpu(Network nn);
void clear_nn_gpu(Network nn);
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
