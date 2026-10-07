#ifndef NN_NN_H
#define NN_NN_H

#include <sys/types.h>
#include <stdbool.h>

#include "mat.h"

typedef struct {
    size_t layers_count;
    size_t *layers_sizes;

    Row *ns;   // Neurons values
    Mat *ws;  // Weights 
    Row *bs;   // Biases
} Network;
#define NN_INPUT(nn) (assert((nn).layers_count > 0), (nn).ns[0])
#define NN_OUTPUT(nn) (assert((nn).layers_count > 0), (nn).ns[(nn).layers_count-1])

Network nn_alloc(size_t nn_layout[], size_t nn_layout_size);
void nn_free(Network nn);
float rand_float(void);
void nn_rand_weights_biases(Network nn, float min, float max);
void nn_activ_func(Mat m);
void nn_forward(Network nn, Row input_data, bool *should_stop);
void nn_backpropagation(Network nn,
                     Row target_output, 
                     Mat *dw,
                     Row *db,
                     bool *should_stop);
void train(Network nn,
           float learning_rate,
           Mat input_data,
           Mat target_output,
           size_t epoch_count,
           bool *should_stop);
float loss(Row output, Row target);

#endif
