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