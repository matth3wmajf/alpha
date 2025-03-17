#include <stdint.h>

#include <math.h>

#include <alpha/numeric.h>

/* ... */
int numeric_matmul(const double_t *pt_a, const double_t *pt_b, double_t *pt_c, uintmax_t t_m, uintmax_t t_k, uintmax_t t_n)
{
	for(uintmax_t l_i = 0; l_i < t_m; l_i++)
	{
		for(uintmax_t l_j = 0; l_j < t_n; l_j++)
		{
/* Compute the sum. */
			double_t l_sum = 0.0;
			for(uintmax_t l_p = 0; l_p < t_k; l_p++) l_sum += pt_a[l_i * t_k + l_p] * pt_b[l_p * t_n + l_j];

/* Store the result. */
			pt_c[l_i * t_n + l_j] = l_sum;
		}
	}

/* Return with success. */
	return 0;
}

/* ... */
int numeric_sigmoid(double_t t_x, double_t *pt_y)
{
/* Set the output value to the sigmoid of the input value. */
	*pt_y = 1.0 / (1.0 + exp(-t_x));

/* Return with success. */
	return 0;
}

/* ... */
int numeric_mse(double_t *pt_output, double_t *pt_target, uintmax_t t_n, double_t *pt_result)
{
/*
 *	The sum.
 *	We initialize this to zero, although it'll eventually store the sum of the
 *	squared differences.
 */
	double_t l_sum = 0.0;

/* Loop for each output & target value. */
	for(uintmax_t l_i = 0; l_i < t_n; l_i++)
	{
/* Compute the difference. */
		double_t l_diff = pt_output[l_i] - pt_target[l_i];

/* Square it, and add it to the sum. */
		l_sum += l_diff * l_diff;
	}

/* Set the result, which is the sum divided by the number of elements. */
	*pt_result = l_sum / (double_t)t_n;

/* Return with success. */
	return 0;
}

/* ... */
int numeric_sigmoid_derivative(double_t t_x, double_t *pt_y)
{
/*
 *	Set the output value to the value computed by the sigmoid function's
 *	derivative.
 */
	*pt_y = t_x * (1.0 - t_x);

/* Return with success. */
	return 0;
}