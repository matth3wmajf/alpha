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
int numeric_matmul(const double_t *pt_a, const double_t *pt_b, double_t *pt_c, uintmax_t t_m, uintmax_t t_k, uintmax_t t_n);

/**
 *	@brief Perform the sigmoid function on a value.
 *	@param t_x The input value.
 *	@param pt_y The output value.
 *	@return The result status code. In this case, it'll always return 0.
 */
int numeric_sigmoid(double_t t_x, double_t *pt_y);

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
int numeric_mse(double_t *pt_output, double_t *pt_target, uintmax_t t_n, double_t *pt_result);

/**
 *	@brief Perform the sigmoid function's derivative on a value.
 *	@param t_x The input value.
 *	@param pt_y The output value.
 *	@return The result status code. In this case, it'll always return 0.
 */
int numeric_sigmoid_derivative(double_t t_x, double_t *pt_y);