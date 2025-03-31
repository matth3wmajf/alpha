#include <stdint.h>

#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <alpha/numeric.h>

/* ... */
int numeric_matmul(const float_t *pt_a, const float_t *pt_b, float_t *pt_c, uintmax_t t_m, uintmax_t t_k, uintmax_t t_n)
{
#ifdef _OPENMP
#pragma omp target teams distribute parallel for collapse(2) schedule(dynamic) map(to: pt_a[0:t_m*t_k], pt_b[0:t_k*t_n]) map(from: pt_c[0:t_m*t_n])
#endif
	for(uintmax_t l_i = 0; l_i < t_m; l_i++)
	{
		for(uintmax_t l_j = 0; l_j < t_n; l_j++)
		{
/* Compute the sum. */
			float_t l_sum = 0.0f;
			for(uintmax_t l_p = 0; l_p < t_k; l_p++) l_sum += pt_a[l_i * t_k + l_p] * pt_b[l_p * t_n + l_j];

/* Store the result. */
			pt_c[l_i * t_n + l_j] = l_sum;
		}
	}

/* Return with success. */
	return 0;
}

/* ... */
int numeric_sigmoid(float_t t_x, float_t *pt_y)
{
/* Set the output value to the sigmoid of the input value. */
	*pt_y = 1.0 / (1.0f + expf(-t_x));

/* Return with success. */
	return 0;
}

/* ... */
int numeric_sigmoid_derivative(float_t t_x, float_t *pt_y)
{
/*
 *	Set the output value to the value computed by the sigmoid function's
 *	derivative.
 */
	*pt_y = t_x * (1.0f - t_x);

/* Return with success. */
	return 0;
}

/* ... */
int numeric_relu(float_t t_x, float_t *pt_y)
{
/* Set the output value to the ReLU output of the input value. */
	*pt_y = (t_x > 0) ? t_x : 0;

/* Return with success. */
	return 0;
}

/* ... */
int numeric_relu_derivative(float_t t_x, float_t *pt_y)
{
/*
 *	Set the output value to the value computed by the ReLU function's
 *	derivative.
 */
	*pt_y = (t_x > 0) ? 1.0f : 0.0f;

/* Return with success. */
	return 0;
}

/* ... */
int numeric_mse(float_t *pt_output, float_t *pt_target, uintmax_t t_n, float_t *pt_result)
{
/*
 *	The sum.
 *	We initialize this to zero, although it'll eventually store the sum of the
 *	squared differences.
 */
	float_t l_sum = 0.0f;

/* Loop for each output & target value. */
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(uintmax_t l_i = 0; l_i < t_n; l_i++)
	{
/* Compute the difference. */
		float_t l_diff = pt_output[l_i] - pt_target[l_i];

/* Square it, and add it to the sum. */
		l_sum += l_diff * l_diff;
	}

/* Set the result, which is the sum divided by the number of elements. */
	*pt_result = l_sum / (float_t)t_n;

/* Return with success. */
	return 0;
}

/* ... */
int numeric_softmax(float_t *pt_input, float_t *pt_output, uintmax_t t_n)
{
/* Determine the maximum value. */
	float_t l_max_value = pt_input[0];
	for(uintmax_t l_i = 1; l_i < t_n; l_i++) if(pt_input[l_i] > l_max_value) l_max_value = pt_input[l_i];

/* ... */
	float_t l_sum_exp = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(uintmax_t l_i = 0; l_i < t_n; l_i++)
	{
		pt_output[l_i] = expf(pt_input[l_i] - l_max_value);
		l_sum_exp += pt_output[l_i];
	}

/* ... */
	for(uintmax_t l_i = 0; l_i < t_n; l_i++) pt_output[l_i] /= l_sum_exp;

/* Return with success. */
	return 0;
}