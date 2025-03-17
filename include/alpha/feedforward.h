#pragma once

#include <stdint.h>

#include <math.h>

/* The structure representing the feedforward neural network. */
typedef struct
{
/* The input layer's nodes & the amount of nodes. */
	double_t *pt_input_layer_buffer;
	uintmax_t t_input_layer_buffer_size;
/* The sizes of the hidden layers & the amount of hidden layer sizes. */
	uintmax_t *pt_hidden_layer_size_buffer;
	uintmax_t t_hidden_layer_size_buffer_size;
/* The weight matrices for the hidden layers. */
	double_t **ppt_hidden_layer_weight_buffer;
	uintmax_t t_hidden_layer_weight_buffer_size;
/* The bias vectors for the hidden layers. */
	double_t **ppt_hidden_layer_bias_buffer;
	uintmax_t t_hidden_layer_weight_bias_size;
/* The output layer's nodes & the amount of nodes. */
	double_t *pt_output_layer_buffer;
	uintmax_t t_output_layer_buffer_size;
/* The output layer's weight matrix. */
	double_t *pt_output_layer_weight_buffer;
	uintmax_t t_output_layer_weight_buffer_size;
/* The output layer's bias vector. */
	double_t *pt_output_layer_bias_buffer;
	uintmax_t t_output_layer_bias_buffer_size;
} feedforward_t;

/**
 *	@brief Initialize an instance of a feedforward neural network.
 *	@param pt_feedforward The handle of the feedforward neural network.
 *	@return The result status code. In this case, it'll always return 0.
 */
int feedforward_create(feedforward_t *pt_feedforward);

/**
 *	@brief Set the sizes of the feedforward neural network & (re-)allocate it's
 *	memory to match these sizes.
 *	@param pt_feedforward The handle of the feedforward neural network.
 *	@param t_input_size The number of nodes in the input layer.
 *	@param pt_hidden_size_buffer The buffer of the number of nodes for each
 *	hidden layer.
 *	@param t_hidden_size_buffer_size The number of hidden layers.
 *	@param t_output_size The number of nodes in the output layer.
 *	@return The result status code. In this case, it'll always return 0.
 */
int feedforward_resize(feedforward_t *pt_feedforward, uintmax_t t_input_size, uintmax_t *pt_hidden_size_buffer, uintmax_t t_hidden_size_buffer_size, uintmax_t t_output_size);

/**
 *	@brief De-initialize an instance of a feedforward neural network.
 *	@param pt_feedforward The handle of the feedforward neural network.
 *	@return The result status code. In this case, it'll always return 0.
 */
int feedforward_delete(feedforward_t *pt_feedforward);

/**
 *	@brief Initialize the weights & biases of the feedforward neural network to
 *	random values.
 *	@param pt_feedforward The handle of the feedforward neural network.
 *	@return The result status code. In this case, it'll always return 0.
 */
int feedforward_random(feedforward_t *pt_feedforward);

/**
 *	@brief Perform the forward pass of the feedforward neural network.
 *	@param pt_feedforward The handle of the feedforward neural network.
 *	@return The result status code.
 */
 int feedforward_forward(feedforward_t *pt_feedforward, double_t *pt_input_buffer, uintmax_t t_input_buffer_size, double_t *pt_output_buffer, uintmax_t t_output_buffer_size);

/**
 *	@brief Perform the backward pass of the feedforward neural network.
 *	@param pt_feedforward The handle of the feedforward neural network.
 *	@param pt_input_buffer The input buffer.
 *	@param t_input_buffer_size The size of the input buffer.
 *	@param pt_target_buffer The target buffer.
 *	@param t_target_buffer_size The size of the target buffer.
 *	@param t_learning_rate The learning rate.
 *	@return The result status code.
 */
int feedforward_backward(feedforward_t *pt_feedforward, double_t *pt_input_buffer, uintmax_t t_input_buffer_size, double_t *pt_target_buffer, uintmax_t t_target_buffer_size, double_t t_learning_rate);