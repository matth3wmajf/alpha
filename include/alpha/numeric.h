#pragma once

#include <stdint.h>

#include <math.h>

/**
 *	@brief Perform the matrix multiplication of two matrices.
 *	@param pt_a The first matrix, stored as a one-dimensional buffer.
 *	@param pt_b The second matrix, stored as a one-dimensional buffer.
 *	@param pt_c The result matrix, also stored as a one-dimensional buffer.
 *	@param t_m The number of rows in the first matrix.
 *	@param t_k The number of columns in the first matrix, and the number of rows in the second matrix.
 *	@param t_n The number of columns in the second matrix.
 *	@return The result status code. In this case, it'll always return 0.
 */
int numeric_matmul(const float_t *pt_a, const float_t *pt_b, float_t *pt_c, uintmax_t t_m, uintmax_t t_k, uintmax_t t_n);

/**
 *	@brief Perform the sigmoid function on a value.
 *	@param t_x The input value.
 *	@param pt_y The output value.
 *	@return The result status code. In this case, it'll always return 0.
 */
int numeric_sigmoid(float_t t_x, float_t *pt_y);

/**
 *	@brief Perform the sigmoid function's derivative on a value.
 *	@param t_x The input value.
 *	@param pt_y The output value.
 *	@return The result status code. In this case, it'll always return 0.
 */
int numeric_sigmoid_derivative(float_t t_x, float_t *pt_y);

/**
 *	@brief Perform the ReLU function on a value.
 *	@param t_x The input value.
 *	@param pt_y The output value.
 *	@return The result status code. In this case, it'll always return 0.
 */
int numeric_relu(float_t t_x, float_t *pt_y);

 /**
  *	@brief Perform the ReLU function's derivative on a value.
  *	@param t_x The input value.
  *	@param pt_y The output value.
  *	@return The result status code. In this case, it'll always return 0.
  */
int numeric_relu_derivative(float_t t_x, float_t *pt_y);

/**
 *	@brief Perform the hyperbolic tangent function on a value.
 *	@param t_x The input value.
 *	@param pt_y The output value.
 *	@return The result status code. In this case, it'll always return 0.
 */
 int numeric_tanh(float_t t_x, float_t *pt_y);

 /**
  *	@brief Perform the hyperbolic tangent function's derivative on a value.
  *	@param t_x The input value.
  *	@param pt_y The output value.
  *	@return The result status code. In this case, it'll always return 0.
  */
int numeric_tanh_derivative(float_t t_x, float_t *pt_y);

/**
 *	@brief Perform the mean squared error function on two vectors.
 *	@param pt_output The output vector (values outputted by a neural network).
 *	@param pt_target The target vector (values desired to be outputted by a
 *	neural network).
 *	@param t_n The number of elements in the vectors (both vectors should have
 *	the same number of elements).
 *	@param pt_result The result of the mean squared error function.
 *	@return The result status code. In this case, it'll always return 0.
 */
int numeric_mse(float_t *pt_output, float_t *pt_target, uintmax_t t_n, float_t *pt_result);

/**
 *	@brief Perform the softmax function on a vector.
 *	@param pt_input The input vector.
 *	@param pt_output The output vector.
 *	@param t_n The number of elements in the vector.
 *	@return The result status code. In this case, it'll always return 0.
 */
int numeric_softmax(float_t *pt_input, float_t *pt_output, uintmax_t t_n);

/**
 *  @brief Apply layer normalization to a vector.
 *  @param pt_input The input vector.
 *  @param pt_output The output vector.
 *  @param pt_gamma The scale parameter vector.
 *  @param pt_beta The shift parameter vector.
 *  @param t_size The size of the vectors.
 *  @return The result status code.
 */
int numeric_layer_norm(float_t *pt_input, float_t *pt_output, float_t *pt_gamma, float_t *pt_beta, uintmax_t t_size);

/**
 *	@brief Transpose a matrix.
 *	@param pt_input The input matrix.
 *	@param pt_output The output (transposed) matrix.
 *	@param t_rows The number of rows in the input matrix.
 *	@param t_cols The number of columns in the input matrix.
 *	@return The result status code. In this case, it'll always return 0.
 */
int numeric_transpose(const float_t *pt_input_buffer, float_t *pt_output_buffer, uintmax_t t_r, uintmax_t t_c);