/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

struct ConvParamsBase {
    using index_t = uint32_t;

    int batch, dim, seqlen, width;
    bool silu_activation;

    index_t x_batch_stride;
    index_t x_c_stride;
    index_t x_l_stride;
    index_t weight_c_stride;
    index_t weight_width_stride;
    index_t out_batch_stride;
    index_t out_c_stride;
    index_t out_l_stride;

    int conv_state_len;
    index_t conv_state_batch_stride;
    index_t conv_state_c_stride;
    index_t conv_state_l_stride;

    // Common data pointers.
    void *__restrict__ x_ptr;
    void *__restrict__ weight_ptr;
    void *__restrict__ bias_ptr;
    void *__restrict__ out_ptr;

    void *__restrict__ conv_state_ptr;
    int32_t *__restrict__ cache_seqlens;

    void *__restrict__ seq_idx_ptr;

    // No __restrict__ since initial_states could be the same as final_states.
    void * initial_states_ptr;
    index_t initial_states_batch_stride;
    index_t initial_states_l_stride;
    index_t initial_states_c_stride;

    void * final_states_ptr;
    index_t final_states_batch_stride;
    index_t final_states_l_stride;
    index_t final_states_c_stride;
};

struct ConvParamsBwd: public ConvParamsBase {
    index_t dx_batch_stride;
    index_t dx_c_stride;
    index_t dx_l_stride;
    index_t dweight_c_stride;
    index_t dweight_width_stride;
    index_t dout_batch_stride;
    index_t dout_c_stride;
    index_t dout_l_stride;

    // Common data pointers.
    void *__restrict__ dx_ptr;
    void *__restrict__ dweight_ptr;
    void *__restrict__ dbias_ptr;
    void *__restrict__ dout_ptr;

    void * dinitial_states_ptr;
    index_t dinitial_states_batch_stride;
    index_t dinitial_states_l_stride;
    index_t dinitial_states_c_stride;

    void * dfinal_states_ptr;
    index_t dfinal_states_batch_stride;
    index_t dfinal_states_l_stride;
    index_t dfinal_states_c_stride;
};
