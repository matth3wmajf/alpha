#include <stdint.h>
#include <stdlib.h>

#include <math.h>
#include <string.h>

#include <alpha/numeric.h>
#include <alpha/feedforward.h>

/* ... */
int feedforward_create(feedforward_t *pt_feedforward)
{
/* ... */
	pt_feedforward->pt_input_layer_buffer = NULL;
	pt_feedforward->t_input_layer_buffer_size = 0;

/* ... */
	pt_feedforward->pt_hidden_layer_size_buffer = NULL;
	pt_feedforward->t_hidden_layer_size_buffer_size = 0;

/* ... */
	pt_feedforward->ppt_hidden_layer_weight_buffer = NULL;
	pt_feedforward->t_hidden_layer_weight_buffer_size = 0;

/* ... */
	pt_feedforward->ppt_hidden_layer_bias_buffer = NULL;
	pt_feedforward->t_hidden_layer_weight_bias_size = 0;

/* ... */
	pt_feedforward->pt_output_layer_buffer = NULL;
	pt_feedforward->t_output_layer_buffer_size = 0;

/* ... */
	pt_feedforward->pt_output_layer_weight_buffer = NULL;
	pt_feedforward->t_output_layer_weight_buffer_size = 0;

/* ... */
	pt_feedforward->pt_output_layer_bias_buffer = NULL;
	pt_feedforward->t_output_layer_bias_buffer_size = 0;

/* ... */
	return 0;
}

/* ... */
int feedforward_resize(feedforward_t *pt_feedforward, uintmax_t t_input_layer_size, uintmax_t *pt_hidden_layer_size_buffer, uintmax_t t_hidden_layer_size_buffer_size, uintmax_t t_output_layer_size)
{
/* ... */
	pt_feedforward->t_input_layer_buffer_size = t_input_layer_size;
	pt_feedforward->pt_input_layer_buffer = realloc(pt_feedforward->pt_input_layer_buffer, sizeof(double_t) * t_input_layer_size);

/* ... */
	pt_feedforward->t_hidden_layer_size_buffer_size = t_hidden_layer_size_buffer_size;
	pt_feedforward->pt_hidden_layer_size_buffer = realloc(pt_feedforward->pt_hidden_layer_size_buffer, sizeof(uintmax_t) * t_hidden_layer_size_buffer_size);
	for(uintmax_t l_i = 0; l_i < t_hidden_layer_size_buffer_size; l_i++) pt_feedforward->pt_hidden_layer_size_buffer[l_i] = pt_hidden_layer_size_buffer[l_i];

/* ... */
	pt_feedforward->t_hidden_layer_weight_buffer_size = t_hidden_layer_size_buffer_size;
	pt_feedforward->ppt_hidden_layer_weight_buffer = realloc(pt_feedforward->ppt_hidden_layer_weight_buffer, sizeof(double_t *) * pt_feedforward->t_hidden_layer_weight_buffer_size);

/* ... */
	for(uintmax_t l_i = 0; l_i < pt_feedforward->t_hidden_layer_weight_buffer_size; l_i++)
	{
		if(pt_feedforward->ppt_hidden_layer_weight_buffer[l_i] != NULL) pt_feedforward->ppt_hidden_layer_weight_buffer[l_i] = NULL;
	}

	for(uintmax_t l_i = 0; l_i < pt_feedforward->t_hidden_layer_weight_buffer_size; l_i++)
	{
/* ... */
		uintmax_t l_last_layer_size = (l_i == 0) ? t_input_layer_size : pt_feedforward->pt_hidden_layer_size_buffer[l_i - 1];
		uintmax_t l_next_layer_size = pt_feedforward->pt_hidden_layer_size_buffer[l_i];
/* ... */
		pt_feedforward->ppt_hidden_layer_weight_buffer[l_i] = realloc(pt_feedforward->ppt_hidden_layer_weight_buffer[l_i], sizeof(double_t) * l_last_layer_size * l_next_layer_size);
	}

/* ... */
	pt_feedforward->t_hidden_layer_weight_bias_size = t_hidden_layer_size_buffer_size;
	pt_feedforward->ppt_hidden_layer_bias_buffer = realloc(pt_feedforward->ppt_hidden_layer_bias_buffer, sizeof(double_t *) * pt_feedforward->t_hidden_layer_weight_bias_size);
	
/* ... */
	for(uintmax_t l_i = 0; l_i < pt_feedforward->t_hidden_layer_weight_buffer_size; l_i++)
	{
		if(pt_feedforward->ppt_hidden_layer_bias_buffer[l_i] != NULL) pt_feedforward->ppt_hidden_layer_bias_buffer[l_i] = NULL;
	}

/* ... */
	for(uintmax_t l_i = 0; l_i < pt_feedforward->t_hidden_layer_weight_bias_size; l_i++) pt_feedforward->ppt_hidden_layer_bias_buffer[l_i] = realloc(pt_feedforward->ppt_hidden_layer_bias_buffer[l_i], sizeof(double_t) * pt_feedforward->pt_hidden_layer_size_buffer[l_i]);

/* ... */
	pt_feedforward->t_output_layer_buffer_size = t_output_layer_size;
	pt_feedforward->pt_output_layer_buffer = realloc(pt_feedforward->pt_output_layer_buffer, sizeof(double_t) * t_output_layer_size);

/* ... */
	pt_feedforward->t_output_layer_weight_buffer_size = t_output_layer_size;
	pt_feedforward->pt_output_layer_weight_buffer = realloc(pt_feedforward->pt_output_layer_weight_buffer, sizeof(double_t) * t_output_layer_size * pt_feedforward->pt_hidden_layer_size_buffer[t_hidden_layer_size_buffer_size - 1]);

/* ... */
	pt_feedforward->t_output_layer_bias_buffer_size = t_output_layer_size;
	pt_feedforward->pt_output_layer_bias_buffer = realloc(pt_feedforward->pt_output_layer_bias_buffer, sizeof(double_t) * t_output_layer_size);

/* ... */
	return 0;
}

/* ... */
int feedforward_delete(feedforward_t *pt_feedforward)
{
/* ... */
	if(pt_feedforward->pt_input_layer_buffer != NULL) free(pt_feedforward->pt_input_layer_buffer);
	pt_feedforward->pt_input_layer_buffer = NULL;
	pt_feedforward->t_input_layer_buffer_size = 0;

/* ... */
if(pt_feedforward->pt_hidden_layer_size_buffer != NULL) free(pt_feedforward->pt_hidden_layer_size_buffer);
	pt_feedforward->pt_hidden_layer_size_buffer = NULL;
	pt_feedforward->t_hidden_layer_size_buffer_size = 0;

/* ... */
	for(uintmax_t l_i = 0; l_i < pt_feedforward->t_hidden_layer_weight_buffer_size; l_i++)
	{
/* ... */
		if(pt_feedforward->ppt_hidden_layer_weight_buffer[l_i] != NULL) free(pt_feedforward->ppt_hidden_layer_weight_buffer[l_i]);
		pt_feedforward->ppt_hidden_layer_weight_buffer[l_i] = NULL;
	}

	if(pt_feedforward->ppt_hidden_layer_weight_buffer != NULL) free(pt_feedforward->ppt_hidden_layer_weight_buffer);
	pt_feedforward->ppt_hidden_layer_weight_buffer = NULL;
	pt_feedforward->t_hidden_layer_weight_buffer_size = 0;

/* ... */
	for(uintmax_t l_i = 0; l_i < pt_feedforward->t_hidden_layer_weight_bias_size; l_i++)
	{
/* ... */
		if(pt_feedforward->ppt_hidden_layer_bias_buffer[l_i] != NULL) free(pt_feedforward->ppt_hidden_layer_bias_buffer[l_i]);
		pt_feedforward->ppt_hidden_layer_bias_buffer[l_i] = NULL;
	}

/* ... */
	if(pt_feedforward->ppt_hidden_layer_bias_buffer != NULL) free(pt_feedforward->ppt_hidden_layer_bias_buffer);
	pt_feedforward->ppt_hidden_layer_bias_buffer = NULL;
	pt_feedforward->t_hidden_layer_weight_bias_size = 0;

/* ... */
	if(pt_feedforward->pt_output_layer_buffer != NULL) free(pt_feedforward->pt_output_layer_buffer);
	pt_feedforward->pt_output_layer_buffer = NULL;
	pt_feedforward->t_output_layer_buffer_size = 0;

/* ... */
	if(pt_feedforward->pt_output_layer_weight_buffer != NULL) free(pt_feedforward->pt_output_layer_weight_buffer);
	pt_feedforward->pt_output_layer_weight_buffer = NULL;
	pt_feedforward->t_output_layer_weight_buffer_size = 0;

/* ... */
	return 0;
}

/* ... */
int feedforward_random(feedforward_t *pt_feedforward)
{
/* Initialize the weights of the hidden layers using the Xavier initialization. */
	for(uintmax_t l_i = 0; l_i < pt_feedforward->t_hidden_layer_weight_buffer_size; l_i++)
	{
/* ... */
		uintmax_t l_last_layer_size = (l_i == 0) ? pt_feedforward->t_input_layer_buffer_size : pt_feedforward->pt_hidden_layer_size_buffer[l_i - 1];
		uintmax_t l_next_layer_size = pt_feedforward->pt_hidden_layer_size_buffer[l_i];

/* The Xavier scaling factor. */
		double_t l_scale = sqrt(2.0 / (l_last_layer_size + l_next_layer_size));

/* Initialize the weights. */
		for(uintmax_t l_j = 0; l_j < l_last_layer_size * l_next_layer_size; l_j++)
		{
			double_t l_random = (double_t)rand() / (double_t)RAND_MAX * 2.0 - 1.0;
			pt_feedforward->ppt_hidden_layer_weight_buffer[l_i][l_j] = l_random * l_scale;
		}

/* Initialize the biases to small values. */
		for(uintmax_t l_j = 0; l_j < pt_feedforward->pt_hidden_layer_size_buffer[l_i]; l_j++) pt_feedforward->ppt_hidden_layer_bias_buffer[l_i][l_j] = 0.01;
	}

/* Initialize the output layer weights. */
	uintmax_t l_last_hidden_size = pt_feedforward->pt_hidden_layer_size_buffer[pt_feedforward->t_hidden_layer_weight_buffer_size - 1];
	double_t l_output_scale = sqrt(2.0 / (l_last_hidden_size + pt_feedforward->t_output_layer_buffer_size));
	for(uintmax_t l_i = 0; l_i < pt_feedforward->t_output_layer_buffer_size * l_last_hidden_size; l_i++)
	{
		double_t l_random = (double_t)rand() / (double_t)RAND_MAX * 2.0 - 1.0;
		pt_feedforward->pt_output_layer_weight_buffer[l_i] = l_random * l_output_scale;
	}

/* Initialize the output layer biases. */
	for(uintmax_t l_i = 0; l_i < pt_feedforward->t_output_layer_buffer_size; l_i++) pt_feedforward->pt_output_layer_bias_buffer[l_i] = 0.01;

/* ... */
	return 0;
}

/* ... */
int feedforward_forward(feedforward_t *pt_feedforward)
{
/* ... */
	double_t *pl_current_layer_activation_buffer = pt_feedforward->pt_input_layer_buffer;
	uintmax_t l_current_layer_activation_buffer_size = pt_feedforward->t_input_layer_buffer_size;
	double_t *pl_next_layer_activation_buffer = NULL;
	uintmax_t l_next_layer_activation_buffer_size = 0;

/* ... */
	for(uintmax_t l_i = 0; l_i < pt_feedforward->t_hidden_layer_size_buffer_size; l_i++)
	{
/* ... */
		l_next_layer_activation_buffer_size = pt_feedforward->pt_hidden_layer_size_buffer[l_i];
		pl_next_layer_activation_buffer = malloc(sizeof(double_t) * l_next_layer_activation_buffer_size);
		if(pl_next_layer_activation_buffer == NULL) return -1;

/* ... */
		numeric_matmul(pl_current_layer_activation_buffer, pt_feedforward->ppt_hidden_layer_weight_buffer[l_i], pl_next_layer_activation_buffer, 1, l_current_layer_activation_buffer_size, l_next_layer_activation_buffer_size);

/* ... */
		for(uintmax_t l_j = 0; l_j < l_next_layer_activation_buffer_size; l_j++)
		{
			pl_next_layer_activation_buffer[l_j] += pt_feedforward->ppt_hidden_layer_bias_buffer[l_i][l_j];
			numeric_sigmoid(pl_next_layer_activation_buffer[l_j], &pl_next_layer_activation_buffer[l_j]);
		}

/* ... */
		if(l_i > 0) free(pl_current_layer_activation_buffer);

/* ... */
		pl_current_layer_activation_buffer = pl_next_layer_activation_buffer;
		l_current_layer_activation_buffer_size = l_next_layer_activation_buffer_size;
	}

/* ... */
	numeric_matmul(pl_current_layer_activation_buffer, pt_feedforward->pt_output_layer_weight_buffer, pt_feedforward->pt_output_layer_buffer, 1, l_current_layer_activation_buffer_size, pt_feedforward->t_output_layer_buffer_size);

/* ... */
	for(uintmax_t l_j = 0; l_j < pt_feedforward->t_output_layer_buffer_size; l_j++)
	{
		pt_feedforward->pt_output_layer_buffer[l_j] += pt_feedforward->pt_output_layer_bias_buffer[l_j];
		numeric_sigmoid(pt_feedforward->pt_output_layer_buffer[l_j], &pt_feedforward->pt_output_layer_buffer[l_j]);
	}

/* ... */
	if(pt_feedforward->t_hidden_layer_size_buffer_size > 0) free(pl_current_layer_activation_buffer);

/* ... */
	return 0;
}