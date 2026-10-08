#include "nn_gpu.h"
#include "raylib.h"
#include "rlgl.h"
#include "glad.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


unsigned int ssbo_inputs;
unsigned int ssbo_outputs;
unsigned int ssbo_weights;
unsigned int ssbo_biases;
unsigned int ssbo_deltas;
unsigned int ssbo_w_deltas;
unsigned int ssbo_b_deltas;

unsigned int u_input_size;
unsigned int u_output_size;
unsigned int u_batch_count;
unsigned int u_weights_offset;
unsigned int u_biases_offset;
unsigned int u_backpropogation;

void init_nn_gpu(Network nn)
{
    // Init shader
    char *nn_train_shader_src = LoadFileText(NN_TRAIN_SHADER_SRC);
    unsigned int nn_train_shader = rlLoadShader(nn_train_shader_src, RL_COMPUTE_SHADER);
    UnloadFileText(nn_train_shader_src);
    nn_train_program = rlLoadShaderProgramCompute(nn_train_shader);

    // Init buffers
    size_t w_elements = 0;
    size_t b_elements = 0;
    size_t delta_elements = 0;
    for(size_t l = 0; l < nn.layers_count-1; l++) {
        w_elements += nn.ws[l].cols*nn.ws[l].rows;
        b_elements += nn.bs[l].cols;
        delta_elements += nn.ns[l+1].cols;
    }
    float *w_data = malloc(w_elements*sizeof(*w_data));
    float *b_data = malloc(b_elements*sizeof(*b_data));
    float *delta_data = calloc(delta_elements, sizeof(*delta_data));
    float *w_delta_data = calloc(w_elements, sizeof(*w_data));
    float *b_delta_data = calloc(b_elements, sizeof(*b_data));
    
    size_t w_pos = 0;
    size_t b_pos = 0;
    for(size_t l = 0; l < nn.layers_count-1; l++) {
        memcpy(w_data+w_pos, nn.ws[l].data, nn.ws[l].cols*nn.ws[l].rows*sizeof(*w_data));
        memcpy(b_data+b_pos, nn.bs[l].data, nn.bs[l].cols*sizeof(*b_data));

        b_pos += nn.bs[l].cols;
        w_pos += nn.ws[l].cols*nn.ws[l].rows;
    }
    ssbo_weights = rlLoadShaderBuffer(w_elements*sizeof(*w_data), w_data, RL_STATIC_DRAW);
    ssbo_biases = rlLoadShaderBuffer(b_elements*sizeof(*b_data), b_data, RL_STATIC_DRAW);
    ssbo_deltas = rlLoadShaderBuffer(delta_elements*sizeof(*delta_data), w_data, RL_STATIC_DRAW);
    ssbo_w_deltas = rlLoadShaderBuffer(b_elements*sizeof(*w_delta_data), b_data, RL_STATIC_DRAW);
    ssbo_b_deltas = rlLoadShaderBuffer(b_elements*sizeof(*b_delta_data), b_data, RL_STATIC_DRAW);

    free(w_data);
    free(b_data);
    free(delta_data);
    free(w_delta_data);
    free(b_delta_data);

    rlEnableShader(nn_train_program);
        rlBindShaderBuffer(ssbo_weights, 2);
        rlBindShaderBuffer(ssbo_biases, 3);
        rlBindShaderBuffer(ssbo_deltas, 4);
        rlBindShaderBuffer(ssbo_w_deltas, 5);
        rlBindShaderBuffer(ssbo_b_deltas, 6);

    // Get uniforms locations
        u_input_size = rlGetLocationUniform(nn_train_program, "u_input_size");
        u_output_size = rlGetLocationUniform(nn_train_program, "u_output_size");
        u_batch_count = rlGetLocationUniform(nn_train_program, "u_batch_count");
        u_weights_offset = rlGetLocationUniform(nn_train_program, "u_weights_offset");
        u_biases_offset = rlGetLocationUniform(nn_train_program, "u_biases_offset");
        u_backpropogation = rlGetLocationUniform(nn_train_program, "u_backpropogation");
    rlDisableShader();
}
void clear_nn_gpu(Network nn)
{
    rlUnloadShaderBuffer(ssbo_weights);
    rlUnloadShaderBuffer(ssbo_biases);
    rlUnloadShaderBuffer(ssbo_deltas);
    rlUnloadShaderBuffer(ssbo_w_deltas);
    rlUnloadShaderBuffer(ssbo_b_deltas);

    rlUnloadShaderProgram(nn_train_program);
}
static void init_train_inputs(Mat input_data, Mat target_output) {
    
}
void train_gpu(Network nn,
           float learning_rate,
           Mat input_data,
           Mat target_output,
           size_t epoch_count,
           bool *should_stop)
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
        // TODO: if(*should_stop) return;

    size_t workgroup_size = 128;
    for(size_t l = 0; l < nn.layers_count-1; l++) {
        rlBindShaderBuffer(ssbo_input_neurons, 0);
        rlBindShaderBuffer(ssbo_output_neurons, 1);

        rlSetUniform(u_input_size, &nn.layers_sizes[l], RL_SHADER_UNIFORM_UINT, 1);
        rlSetUniform(u_output_size, &nn.layers_sizes[l+1], RL_SHADER_UNIFORM_UINT, 1);
        rlSetUniform(u_weights_offset, &weights_offset, RL_SHADER_UNIFORM_UINT, 1);
        rlSetUniform(u_biases_offset, &biases_offset, RL_SHADER_UNIFORM_UINT, 1);
        
        size_t workgroups_num = (nn.layers_sizes[l+1]+workgroup_size-1)/workgroup_size;
        rlComputeShaderDispatch(workgroups_num, 1, 1);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        rlReadShaderBuffer(ssbo_output_neurons, nn.ns[l+1].data, nn.ns[l+1].cols*sizeof(*nn.ns->data), 0);

        // Swap input and output neuron buffers for next layer
        unsigned int tmp = ssbo_input_neurons;
        ssbo_input_neurons = ssbo_output_neurons;
        ssbo_output_neurons = tmp;

        weights_offset += nn.ws[l].cols*nn.ws[l].rows;
        biases_offset += nn.bs[l].cols;
    }

    rlDisableShader();

    rlUnloadShaderBuffer(ssbo_input_neurons);
    rlUnloadShaderBuffer(ssbo_output_neurons);
    rlUnloadShaderBuffer(ssbo_weights);
    rlUnloadShaderBuffer(ssbo_biases);

}
/*
void nn_forward_gpu(Network nn, Row input_data, bool *should_stop)
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
        // TODO: if(*should_stop) return;

    size_t workgroup_size = 128;
    for(size_t l = 0; l < nn.layers_count-1; l++) {
        rlBindShaderBuffer(ssbo_input_neurons, 0);
        rlBindShaderBuffer(ssbo_output_neurons, 1);

        rlSetUniform(u_input_size, &nn.layers_sizes[l], RL_SHADER_UNIFORM_UINT, 1);
        rlSetUniform(u_output_size, &nn.layers_sizes[l+1], RL_SHADER_UNIFORM_UINT, 1);
        rlSetUniform(u_weights_offset, &weights_offset, RL_SHADER_UNIFORM_UINT, 1);
        rlSetUniform(u_biases_offset, &biases_offset, RL_SHADER_UNIFORM_UINT, 1);
        
        size_t workgroups_num = (nn.layers_sizes[l+1]+workgroup_size-1)/workgroup_size;
        rlComputeShaderDispatch(workgroups_num, 1, 1);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        rlReadShaderBuffer(ssbo_output_neurons, nn.ns[l+1].data, nn.ns[l+1].cols*sizeof(*nn.ns->data), 0);

        // Swap input and output neuron buffers for next layer
        unsigned int tmp = ssbo_input_neurons;
        ssbo_input_neurons = ssbo_output_neurons;
        ssbo_output_neurons = tmp;

        weights_offset += nn.ws[l].cols*nn.ws[l].rows;
        biases_offset += nn.bs[l].cols;
    }

    rlDisableShader();

    rlUnloadShaderBuffer(ssbo_input_neurons);
    rlUnloadShaderBuffer(ssbo_output_neurons);
    rlUnloadShaderBuffer(ssbo_weights);
    rlUnloadShaderBuffer(ssbo_biases);
} */
