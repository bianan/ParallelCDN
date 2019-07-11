//Copyright [2019] [Yatao (An) Bian / ETH Zurich]
/**
    This file is part of the implementation of PCDN, SCDN and CDN as described in the paper:

Parallelized Coordinate Descent Newton Method for Efficient L1-Regularized Minimization.
https://ieeexplore.ieee.org/abstract/document/8661743

    Please cite the paper if you are using this code in your work.

    The code may be used free of charge for non-commercial and
    educational purposes, the only requirement is that this text is
    preserved within the derivative work. For any other purpose you
    must contact the authors for permission. This code may not be
    redistributed without permission from the authors.

*/
#ifndef _L1_MINIMIZATION_H
#define _L1_MINIMIZATION_H
//#include <string.h>

#include <omp.h>
//typedef unsigned int size_t
#define PCDN_CAS_LR
#ifdef __cplusplus
extern "C" {
#endif

struct FeatureNode
{
	int index;
	double value;
};

struct Problem
{
	int nnz;  //   no bias
	int l;  //   #sample
	int n;   // #feature
	int *y;   //  label
	struct FeatureNode **x;   //  design matrix X
	double bias;            /* == 0 if no bias term */
};

enum SolverType
{
	L1R_LR_B, //0   ------
	L1R_L2LOSS_SVC, //1  ------
	//L1R_LR//2
}; /* solver_type */

enum Algorithm
{
	kCDN, //0
	kSCDN, // 1
	kPCDN,//2
	kNum
};

struct Parameter
{
	int solver_type;   //  default L1R_LR_B
	Algorithm algorithm_type;  //   default kPCDN
	/* these are for training only */
	double eps;	        /* stopping criteria */
	double C;
	int nr_weight;
	int *weight_label;
	double* weight;
	//experiment
	double eps_g;
	int eps_n;
};

struct Model
{
	struct Parameter param;
	int num_class;		/* number of classes */
	int num_feature;
	double *w;
	int *label;		/* label of each class */
	double bias; // model bias,  used for L1R_LR_B   ==0   no bias
};

//struct SmCsr
//{//  sparse matrix compressed row   used in GPU(OpenCL implementation)
//	double *Av; // nnz
//	int *Aj;//  nnz  corresponding column number of the elements in Av
//	int *Ap; //   #rows +1
//};



struct Model* train(const struct Problem *prob, const struct Problem *probtest, const struct Parameter *param);

void cross_validation(const struct Problem *prob, const struct Parameter *param, int nr_fold, int *target);
int predict_values(const struct Model *model_, const struct FeatureNode *x, double* dec_values);
int infer(const struct Model *model_, const struct FeatureNode *x);
//int predict_probability(const struct Model *model_, const struct FeatureNode *x, double* prob_estimates);

int save_model(const char *model_file_name, const struct Model *model_);
struct Model *load_model(const char *model_file_name);

int get_nr_feature(const struct Model *model_);
int get_nr_class(const struct Model *model_);
void get_labels(const struct Model *model_, int* label);

void destroy_model(struct Model *model_);
void destroy_param(struct Parameter *param);
const char *check_parameter(const struct Problem *prob, const struct Parameter *param);
extern void (*liblinear_print_string) (const char *);
void info(const char *fmt,...);
void info_save(const char *fmt,...);
void save_exp(const char *fmt,...);
double evaluate_testing_bias(double *w, int w_size, double bias, const Problem *probtest);
#ifdef __cplusplus
}
#endif
#endif
