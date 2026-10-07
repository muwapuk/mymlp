#include "nn_gpu.h"
#include "raylib.h"
#include "rlgl.h"
#include "glad.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


void init_nn_gpu()
{
    char *nn_train_shader_src = LoadFileText(NN_FORWARD_SHADER_PATH);
    unsigned int nn_train_shader = rlLoadShader(nn_train_shader_src, RL_COMPUTE_SHADER);
    UnloadFileText(nn_train_shader_src);
    nn_train_program = rlLoadShaderProgramCompute(nn_train_shader);
}
void clear_nn_gpu()
{
    rlUnloadShaderProgram(nn_train_program);
}
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
    //char *nn_train_shader_src = LoadFileText(NN_FORWARD_SHADER_PATH);
    //unsigned int nn_train_shader = rlLoadShader(nn_train_shader_src, RL_COMPUTE_SHADER);
    //UnloadFileText(nn_train_shader_src);
    //unsigned int nn_train_program = rlLoadShaderProgramCompute(nn_train_shader);

    // TODO: if(*should_stop) return;
    size_t n_elements_count = 0;
    size_t w_elements_count = 0;
    size_t b_elements_count = 0;
    for(size_t l = 0; l < nn.layers_count-1; l++) {
        n_elements_count = nn.layers_sizes[l] > n_elements_count ? nn.layers_sizes[l] : n_elements_count;
        w_elements_count += nn.ws[l].cols*nn.ws[l].rows;
        b_elements_count += nn.bs[l].cols;
    }
    n_elements_count = 
        nn.layers_sizes[nn.layers_count-1] > n_elements_count 
        ? nn.layers_sizes[nn.layers_count-1] 
        : n_elements_count;

    float *n_data = calloc(n_elements_count, sizeof(*n_data));
    float *w_data = malloc(w_elements_count*sizeof(*w_data));
    float *b_data = malloc(b_elements_count*sizeof(*b_data));
    
    size_t w_pos = 0;
    size_t b_pos = 0;
    for(size_t l = 0; l < nn.layers_count-1; l++) {
        memcpy(w_data+w_pos, nn.ws[l].data, nn.ws[l].cols*nn.ws[l].rows*sizeof(*w_data));
        memcpy(b_data+b_pos, nn.bs[l].data, nn.bs[l].cols*sizeof(*b_data));

        b_pos += nn.bs[l].cols;
        w_pos += nn.ws[l].cols*nn.ws[l].rows;
    }
    memcpy(n_data, input_data.data, input_data.cols*sizeof(*n_data));

    unsigned int ssbo_input_neurons = rlLoadShaderBuffer(n_elements_count*sizeof(*n_data), n_data, RL_STREAM_COPY);
    unsigned int ssbo_output_neurons = rlLoadShaderBuffer(n_elements_count*sizeof(*n_data), 0, RL_STREAM_COPY);
    unsigned int ssbo_weights = rlLoadShaderBuffer(w_elements_count*sizeof(*w_data), w_data, RL_STATIC_DRAW);
    unsigned int ssbo_biases = rlLoadShaderBuffer(b_elements_count*sizeof(*b_data), b_data, RL_STATIC_DRAW);

    free(n_data);
    free(w_data);
    free(b_data);

    int u_input_size = rlGetLocationUniform(nn_train_program, "u_input_size");
    int u_output_size = rlGetLocationUniform(nn_train_program, "u_output_size");
    int u_weights_offset = rlGetLocationUniform(nn_train_program, "u_weights_offset");
    int u_biases_offset = rlGetLocationUniform(nn_train_program, "u_biases_offset");

    size_t weights_offset = 0;
    size_t biases_offset = 0;

    rlEnableShader(nn_train_program);
    rlBindShaderBuffer(ssbo_weights, 2);
    rlBindShaderBuffer(ssbo_biases, 3);

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
float loss(Row output, Row target)
{
    float loss = 0.f;
    for(size_t col = 0; col < output.cols; col++) {
        float diff = ROW_AT(output, col) - ROW_AT(target, col);
        loss += 0.5f * diff * diff;
    }
    return loss;
}
