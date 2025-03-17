#include <stdatomic.h>
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
/* Allocate (or re-allocate) the memory for the input layer. */
	pt_feedforward->t_input_layer_buffer_size = t_input_layer_size;
	pt_feedforward->pt_input_layer_buffer = realloc(pt_feedforward->pt_input_layer_buffer, sizeof(double_t) * t_input_layer_size);

/*
 *	Allocate (or re-allocate) the memory for the buffer storing the different
 *	sizes of the different hidden layers.
 */
	pt_feedforward->t_hidden_layer_size_buffer_size = t_hidden_layer_size_buffer_size;
	pt_feedforward->pt_hidden_layer_size_buffer = realloc(pt_feedforward->pt_hidden_layer_size_buffer, sizeof(uintmax_t) * t_hidden_layer_size_buffer_size);
	for(uintmax_t l_i = 0; l_i < t_hidden_layer_size_buffer_size; l_i++) pt_feedforward->pt_hidden_layer_size_buffer[l_i] = pt_hidden_layer_size_buffer[l_i];

/*
 *	Allocate (or re-allocate) the memory that stores the pointers to the weight
 *	matrices for each of the hidden layers.
 */
	pt_feedforward->t_hidden_layer_weight_buffer_size = t_hidden_layer_size_buffer_size;
	pt_feedforward->ppt_hidden_layer_weight_buffer = realloc(pt_feedforward->ppt_hidden_layer_weight_buffer, sizeof(double_t *) * pt_feedforward->t_hidden_layer_weight_buffer_size);

/*
 *	Ensure that all of the pointers for the weight matrices are null, ready for
 *	allocation.
 */
	for(uintmax_t l_i = 0; l_i < pt_feedforward->t_hidden_layer_weight_buffer_size; l_i++)
	{
		if(pt_feedforward->ppt_hidden_layer_weight_buffer[l_i] != NULL) pt_feedforward->ppt_hidden_layer_weight_buffer[l_i] = NULL;
	}

/*
 *	Then, allocate the memory for the actual weight matrices for each of the
 *	hidden layers.
 */
	for(uintmax_t l_i = 0; l_i < pt_feedforward->t_hidden_layer_weight_buffer_size; l_i++)
	{
/* Keep track of the sizes of the previous & next layers. */
		uintmax_t l_last_layer_size = (l_i == 0) ? t_input_layer_size : pt_feedforward->pt_hidden_layer_size_buffer[l_i - 1];
		uintmax_t l_next_layer_size = pt_feedforward->pt_hidden_layer_size_buffer[l_i];
/* Allocate the memory for the weight matrix. */
		pt_feedforward->ppt_hidden_layer_weight_buffer[l_i] = realloc(pt_feedforward->ppt_hidden_layer_weight_buffer[l_i], sizeof(double_t) * l_last_layer_size * l_next_layer_size);
	}

/* Allocate the pointers for the bias vectors for each of the hidden layers. */
	pt_feedforward->t_hidden_layer_weight_bias_size = t_hidden_layer_size_buffer_size;
	pt_feedforward->ppt_hidden_layer_bias_buffer = realloc(pt_feedforward->ppt_hidden_layer_bias_buffer, sizeof(double_t *) * pt_feedforward->t_hidden_layer_weight_bias_size);
	
/*
 *	Ensure that all of the pointers for the bias vectors are null, ready for
 *	allocation.
 */
	for(uintmax_t l_i = 0; l_i < pt_feedforward->t_hidden_layer_weight_buffer_size; l_i++)
	{
		if(pt_feedforward->ppt_hidden_layer_bias_buffer[l_i] != NULL) pt_feedforward->ppt_hidden_layer_bias_buffer[l_i] = NULL;
	}

/* Allocate the memory for the bias vectors for each of the hidden layers. */
	for(uintmax_t l_i = 0; l_i < pt_feedforward->t_hidden_layer_weight_bias_size; l_i++) pt_feedforward->ppt_hidden_layer_bias_buffer[l_i] = realloc(pt_feedforward->ppt_hidden_layer_bias_buffer[l_i], sizeof(double_t) * pt_feedforward->pt_hidden_layer_size_buffer[l_i]);

/* Allocate the memory for the output layer. */
	pt_feedforward->t_output_layer_buffer_size = t_output_layer_size;
	pt_feedforward->pt_output_layer_buffer = realloc(pt_feedforward->pt_output_layer_buffer, sizeof(double_t) * t_output_layer_size);

/* Allocate the memory for the weight matrix of the output layer. */
	pt_feedforward->t_output_layer_weight_buffer_size = t_output_layer_size;
	pt_feedforward->pt_output_layer_weight_buffer = realloc(pt_feedforward->pt_output_layer_weight_buffer, sizeof(double_t) * t_output_layer_size * pt_feedforward->pt_hidden_layer_size_buffer[t_hidden_layer_size_buffer_size - 1]);

/* Allocate the memory for the bias vector of the output layer. */
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
/* Keep track of the previous & next layer sizes. */
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

/* Return with success. */
	return 0;
}

/* ... */
int feedforward_forward(feedforward_t *pt_feedforward, double_t *pt_input_buffer, uintmax_t t_input_buffer_size, double_t *pt_output_buffer, uintmax_t t_output_buffer_size)
{
/* Copy the input buffer to the input layer. */
	memcpy(pt_feedforward->pt_input_layer_buffer, pt_input_buffer, sizeof(double_t) * t_input_buffer_size);

/*
 *	Keep track of the activation buffer for our current layer & the next layer.
 */
	double_t *pl_current_layer_activation_buffer = pt_feedforward->pt_input_layer_buffer;
	uintmax_t l_current_layer_activation_buffer_size = pt_feedforward->t_input_layer_buffer_size;
	double_t *pl_next_layer_activation_buffer = NULL;
	uintmax_t l_next_layer_activation_buffer_size = 0;

/* Loop for each hidden layer. */
	for(uintmax_t l_i = 0; l_i < pt_feedforward->t_hidden_layer_size_buffer_size; l_i++)
	{
/*
 *	Set the next layer & allocate memory for temporarily tracking it's
 *	activations.
 */
		l_next_layer_activation_buffer_size = pt_feedforward->pt_hidden_layer_size_buffer[l_i];
		pl_next_layer_activation_buffer = malloc(sizeof(double_t) * l_next_layer_activation_buffer_size);
		if(pl_next_layer_activation_buffer == NULL) return -1;

/*
 *	Multiply the current activation values (the previous layer's output) by the
 *	current hidden layer's weights.
 */
		numeric_matmul(pl_current_layer_activation_buffer, pt_feedforward->ppt_hidden_layer_weight_buffer[l_i], pl_next_layer_activation_buffer, 1, l_current_layer_activation_buffer_size, l_next_layer_activation_buffer_size);

/*
 *	Add the biases to the current activation values, and apply the sigmoid
 *	function.
 */
		for(uintmax_t l_j = 0; l_j < l_next_layer_activation_buffer_size; l_j++)
		{
			pl_next_layer_activation_buffer[l_j] += pt_feedforward->ppt_hidden_layer_bias_buffer[l_i][l_j];
			numeric_sigmoid(pl_next_layer_activation_buffer[l_j], &pl_next_layer_activation_buffer[l_j]);
		}

/* Free the temporary activation buffer. */
		if(l_i > 0) free(pl_current_layer_activation_buffer);

/* Set the current layer's activation buffer to the next layer's activation buffer. */
		pl_current_layer_activation_buffer = pl_next_layer_activation_buffer;
		l_current_layer_activation_buffer_size = l_next_layer_activation_buffer_size;
	}

/* Multiply the last hidden layer's output by the output layer's weights. */
	numeric_matmul(pl_current_layer_activation_buffer, pt_feedforward->pt_output_layer_weight_buffer, pt_feedforward->pt_output_layer_buffer, 1, l_current_layer_activation_buffer_size, pt_feedforward->t_output_layer_buffer_size);

/*
 *	Apply the output layer's biases to the output layer's values, and then
 *	apply the sigmoid function.
 */
	for(uintmax_t l_j = 0; l_j < pt_feedforward->t_output_layer_buffer_size; l_j++)
	{
		pt_feedforward->pt_output_layer_buffer[l_j] += pt_feedforward->pt_output_layer_bias_buffer[l_j];
		numeric_sigmoid(pt_feedforward->pt_output_layer_buffer[l_j], &pt_feedforward->pt_output_layer_buffer[l_j]);
	}

/* Free the memory previously allocated for the last hidden layer's activation buffer. */
	if(pt_feedforward->t_hidden_layer_size_buffer_size > 0) free(pl_current_layer_activation_buffer);

/* Copy the output layer buffer to the attached output buffer. */
	memcpy(pt_output_buffer, pt_feedforward->pt_output_layer_buffer, sizeof(double_t) * t_output_buffer_size);

/* Return with success. */
	return 0;
}

int feedforward_backward(feedforward_t *pt_feedforward, double_t *pt_input_buffer, uintmax_t t_input_buffer_size, double_t *pt_target_buffer, uintmax_t t_target_buffer_size, double_t t_learning_rate)
{
/* Copy the input buffer to the internal structure's input layer. */
	memcpy(pt_feedforward->pt_input_layer_buffer, pt_input_buffer, sizeof(double_t) * t_input_buffer_size);

/* The size of the buffer storing the hidden layer buffers (number of hidden layers). */
	uintmax_t ppl_hidden_activation_buffer_buffer_size = pt_feedforward->t_hidden_layer_size_buffer_size;
	
/*
 * The actual buffer that stores the pointers to the hidden layers' activation buffers.
 * This is a temporary buffer used to store the activations of the different hidden layers.
 */
	double_t **ppl_hidden_activation_buffer_buffer = malloc(sizeof(double_t *) * ppl_hidden_activation_buffer_buffer_size);
	if(ppl_hidden_activation_buffer_buffer == NULL) return -1;

/* The buffer that keeps track of the previous layer's activation buffer. */
	double_t *pl_previous_layer_activation_buffer = pt_feedforward->pt_input_layer_buffer;
	uintmax_t pl_previous_layer_activation_buffer_size = t_input_buffer_size;

/* Forward propagation: Loop for each hidden layer. */
	for(uintmax_t l_i = 0; l_i < ppl_hidden_activation_buffer_buffer_size; l_i++)
	{
/* Allocate memory for the current hidden layer's activation buffer. */
		uintmax_t l_current_layer_activation_buffer_size = pt_feedforward->pt_hidden_layer_size_buffer[l_i];
		double_t *pl_current_layer_activation_buffer = malloc(l_current_layer_activation_buffer_size * sizeof(double_t));
		if(!pl_current_layer_activation_buffer)
		{
/* Free the memory previously allocated for the hidden layer activation buffers. */
			for(uintmax_t l_j = 0; l_j < l_i; l_j++) free(ppl_hidden_activation_buffer_buffer[l_j]);
/* Free the memory storing the pointers to the hidden layer activation buffers. */
			free(ppl_hidden_activation_buffer_buffer);
/* Return with failure. */
			return -1;
		}

/*
 *	Multiply the previous layer's activation vector by the current hidden
 *	layer's weights.
 */
		numeric_matmul(pl_previous_layer_activation_buffer, pt_feedforward->ppt_hidden_layer_weight_buffer[l_i], pl_current_layer_activation_buffer, 1, pl_previous_layer_activation_buffer_size, l_current_layer_activation_buffer_size);

/* Loop for each of the current layer's activation values. */
		for(uintmax_t l_j = 0; l_j < l_current_layer_activation_buffer_size; l_j++)
		{
/* Add the bias to the current activation value. */
			pl_current_layer_activation_buffer[l_j] += pt_feedforward->ppt_hidden_layer_bias_buffer[l_i][l_j];
/* Compute the sigmoid of the current activation value. */
			double_t l_temp;
			numeric_sigmoid(pl_current_layer_activation_buffer[l_j], &l_temp);
			pl_current_layer_activation_buffer[l_j] = l_temp;
		}

/* Prepare for the next iteration. */
		ppl_hidden_activation_buffer_buffer[l_i] = pl_current_layer_activation_buffer;
		pl_previous_layer_activation_buffer = pl_current_layer_activation_buffer;
		pl_previous_layer_activation_buffer_size = l_current_layer_activation_buffer_size;
	}
	
/*
 *	Multiply the last hidden layer's activation vector by the output layer's
 *	weights.
 */
	numeric_matmul(pl_previous_layer_activation_buffer, pt_feedforward->pt_output_layer_weight_buffer, pt_feedforward->pt_output_layer_buffer, 1, pl_previous_layer_activation_buffer_size, pt_feedforward->t_output_layer_buffer_size);
	for(uintmax_t l_j = 0; l_j < pt_feedforward->t_output_layer_buffer_size; l_j++)
	{
/* Add the bias. */
		pt_feedforward->pt_output_layer_buffer[l_j] += pt_feedforward->pt_output_layer_bias_buffer[l_j];
/* Set the current activation value to the sigmoid of itself. */
		double_t l_temp;
		numeric_sigmoid(pt_feedforward->pt_output_layer_buffer[l_j], &l_temp);
		pt_feedforward->pt_output_layer_buffer[l_j] = l_temp;
	}
	
/* Compute the MSE for debugging purposes. */
	double_t l_mse;
	numeric_mse(pt_feedforward->pt_output_layer_buffer, pt_target_buffer, pt_feedforward->t_output_layer_buffer_size, &l_mse);

/*
 *	Allocate delta buffer for the output layer.
 *	This buffer will store the gradients of the output layer with respect to
 *	it's inputs.
 */
	double_t *pl_delta_output_buffer = malloc(pt_feedforward->t_output_layer_buffer_size * sizeof(double_t));
	if(!pl_delta_output_buffer)
	{
/*
 *	If error occurs while allocating this memory, then free the memory
 *	allocated for the hidden layer activation buffers, and free the
 *	memory storing the pointers to the hidden layer activation buffers
 *	themselves.
 */
		for(uintmax_t l_i = 0; l_i < ppl_hidden_activation_buffer_buffer_size; l_i++) free(ppl_hidden_activation_buffer_buffer[l_i]);
		free(ppl_hidden_activation_buffer_buffer);

/* Return with failure. */
		return -1;
	}

/* Compute the error gradients for each output. */
	for(uintmax_t l_j = 0; l_j < pt_feedforward->t_output_layer_buffer_size; l_j++)
	{
/* Calculate the error. */
		double_t l_error = pt_feedforward->pt_output_layer_buffer[l_j] - pt_target_buffer[l_j];
/* Compute the derivative of the sigmoid of the current output. */
		double_t l_deriv;
		numeric_sigmoid_derivative(pt_feedforward->pt_output_layer_buffer[l_j], &l_deriv);
/* And set the delta output buffer to the error times the derivative. */
		pl_delta_output_buffer[l_j] = l_error * l_deriv;
	}
	
/*
 *	Update output layer weights and biases using the computed gradients.
 *	Here, we use the last hidden layer's activations (or the input layer if
 *	there's no hidden layer) for the update.
 */
	double_t *pl_last_hidden_activation_buffer = (ppl_hidden_activation_buffer_buffer_size > 0) ? ppl_hidden_activation_buffer_buffer[ppl_hidden_activation_buffer_buffer_size - 1] : pt_feedforward->pt_input_layer_buffer;
	uintmax_t l_last_hidden_activation_buffer_size = (ppl_hidden_activation_buffer_buffer_size > 0) ? pt_feedforward->pt_hidden_layer_size_buffer[ppl_hidden_activation_buffer_buffer_size - 1] : t_input_buffer_size;
	for(uintmax_t l_i = 0; l_i < l_last_hidden_activation_buffer_size; l_i++)
	{
		for(uintmax_t l_j = 0; l_j < pt_feedforward->t_output_layer_buffer_size; l_j++) pt_feedforward->pt_output_layer_weight_buffer[l_i * pt_feedforward->t_output_layer_buffer_size + l_j] -= t_learning_rate * pl_last_hidden_activation_buffer[l_i] * pl_delta_output_buffer[l_j];
	}

/* Update output biases using the gradient descent rule. */
	for(uintmax_t l_j = 0; l_j < pt_feedforward->t_output_layer_buffer_size; l_j++) pt_feedforward->pt_output_layer_bias_buffer[l_j] -= t_learning_rate * pl_delta_output_buffer[l_j];

/*
 *	Backpropagate the error in order to compute gradients for the hidden layers.
 *	Here, we start with the output layer's delta value(s) & work backwards
 *	through the hidden layer(s).
 */
	double_t *pl_delta_next_layer_buffer = pl_delta_output_buffer;
	uintmax_t l_delta_next_layer_buffer_size = pt_feedforward->t_output_layer_buffer_size;
	
/* Iterate over the hidden layers in reverse. */
	for(intmax_t l_i = ppl_hidden_activation_buffer_buffer_size - 1; l_i >= 0; l_i--)
	{
/*
 *	Prepare the current hidden layer's node(s).
 *	Here, we allocate memory for storing the current hidden layer's delta
 *	value(s).
 */
		uintmax_t l_delta_current_layer_buffer_size = pt_feedforward->pt_hidden_layer_size_buffer[l_i];
		double_t *pl_delta_current_layer_buffer = malloc(l_delta_current_layer_buffer_size * sizeof(double_t));
		if(!pl_delta_current_layer_buffer)
		{
/* ... */
			if(l_i != (intmax_t)(ppl_hidden_activation_buffer_buffer_size - 1)) free(pl_delta_next_layer_buffer);
			for(uintmax_t l_j = 0; l_j < ppl_hidden_activation_buffer_buffer_size; l_j++) free(ppl_hidden_activation_buffer_buffer[l_j]);
			free(ppl_hidden_activation_buffer_buffer);
			free(pl_delta_output_buffer);
/* Return with failure. */
			return -1;
		}
/* For each node in the current hidden layer, compute its delta value. */
		for(uintmax_t l_j = 0; l_j < l_delta_current_layer_buffer_size; l_j++)
		{
			double_t l_sum = 0.0;
/*
 *	For the last hidden layer, use the output layer's weights and deltas.
 *	If this isn't the last hidden layer, use the weights and deltas from the
 *	next hidden layer.
 */
			if(l_i == (intmax_t)(ppl_hidden_activation_buffer_buffer_size - 1))
			{
				for(uintmax_t l_k = 0; l_k < l_delta_next_layer_buffer_size; l_k++)
				{
					double_t l_weight = pt_feedforward->pt_output_layer_weight_buffer[l_j * l_delta_next_layer_buffer_size + l_k];
					l_sum += l_weight * pl_delta_next_layer_buffer[l_k];
				}
			}
			else
			{
				uintmax_t l_next_hidden_layer_size = pt_feedforward->pt_hidden_layer_size_buffer[l_i + 1];
				for(uintmax_t l_k = 0; l_k < l_next_hidden_layer_size; l_k++)
				{
					double_t l_weight = pt_feedforward->ppt_hidden_layer_weight_buffer[l_i + 1][l_j * l_next_hidden_layer_size + l_k];
					l_sum += l_weight * pl_delta_next_layer_buffer[l_k];
				}

/* ... */
				l_delta_next_layer_buffer_size = pt_feedforward->pt_hidden_layer_size_buffer[l_i + 1];
			}

/* Retrieve the activation value for the current node. */
			double_t l_activation = ppl_hidden_activation_buffer_buffer[l_i][l_j];
/*
 *	Compute the derivative of the sigmoid function with the activation value
 *	being the input, and the set output.
 */
			double_t l_deriv;
			numeric_sigmoid_derivative(l_activation, &l_deriv);
/* ... */
			pl_delta_current_layer_buffer[l_j] = l_sum * l_deriv;
		}

/* ... */
		double_t *pl_previous_activation_layer_buffer = (l_i == 0) ? pt_feedforward->pt_input_layer_buffer : ppl_hidden_activation_buffer_buffer[l_i - 1];
		uintmax_t l_previous_activation_layer_buffer_size = (l_i == 0) ? t_input_buffer_size : pt_feedforward->pt_hidden_layer_size_buffer[l_i - 1];
		
/* Update the weights connecting the previous layer to the current hidden layer. */
		for(uintmax_t l_j = 0; l_j < l_previous_activation_layer_buffer_size; l_j++)
		{
			for(uintmax_t l_k = 0; l_k < l_delta_current_layer_buffer_size; l_k++) pt_feedforward->ppt_hidden_layer_weight_buffer[l_i][l_j * l_delta_current_layer_buffer_size + l_k] -= t_learning_rate * pl_previous_activation_layer_buffer[l_j] * pl_delta_current_layer_buffer[l_k];
		}

/* Update the biases for the current hidden layer. */
		for(uintmax_t l_k = 0; l_k < l_delta_current_layer_buffer_size; l_k++) pt_feedforward->ppt_hidden_layer_bias_buffer[l_i][l_k] -= t_learning_rate * pl_delta_current_layer_buffer[l_k];
		
/* ... */
		if(l_i != (intmax_t)(ppl_hidden_activation_buffer_buffer_size - 1)) free(pl_delta_next_layer_buffer);

/* ... */
		pl_delta_next_layer_buffer = pl_delta_current_layer_buffer;
	}

/* ... */
	free(pl_delta_next_layer_buffer);
	
/* ... */
	for(uintmax_t l_i = 0; l_i < ppl_hidden_activation_buffer_buffer_size; l_i++) free(ppl_hidden_activation_buffer_buffer[l_i]);
	free(ppl_hidden_activation_buffer_buffer);
	
/* ... */
	free(pl_delta_output_buffer);

/* Return with success. */
	return 0;
}
