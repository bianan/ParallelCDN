//Copyright [2017] [An Bian / ETH Zurich]
/** 
    This file is part of the implementation of PCDN, SCDN and CDN as described in the paper:

Parallelized Coordinate Descent Newton Method for Efficient L1-Regularized Minimization.
https://arxiv.org/abs/1306.4080

    Please cite the paper if you are using this code in your work.

    The code may be used free of charge for non-commercial and
    educational purposes, the only requirement is that this text is
    preserved within the derivative work. For any other purpose you
    must contact the authors for permission. This code may not be
    redistributed without permission from the authors.  
*/
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <stdarg.h>
#include <fstream>
#include <vector>
#include "l1_minimization.h"
#include "cas_array.h"

extern std::string gOutfile_name;
//extern std::string gOutpath;
extern std::string gOutfile_name_verbosity;
extern Parameter g_param;
extern struct Problem gProb;
extern int gNum_procs;
extern int g_pcdn_thread_num;
extern int g_scdn_thread_num;
extern int g_bundle_size;


typedef signed char schar;

template <class T> static inline void Swap(T& x, T& y) { T t=x; x=y; y=t; }

#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{   
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}

void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
void info_save(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);

	std::ofstream outfile(gOutfile_name.c_str(),std::ios::app);
	if (! outfile)
	{
		printf("error in outfile!!");
	}
	outfile<<buf;
	outfile.close();
}
void save_exp(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	//(*liblinear_print_string)(buf);

	std::ofstream outfile(gOutfile_name_verbosity.c_str(),std::ios::app);
	if (! outfile)
	{
		printf("error in outfile!!");
	}
	outfile<<buf;
	outfile.close();
}
#else
void info(const char *fmt,...) {}
#endif

// w^T x + bias
double evaluate_testing_bias(double *w, int w_size, double bias, const Problem *probtest)
{
	int i;
	int l = probtest->l;
	int *y = probtest->y;
	int correct = 0;

	for(i=0; i<l; i++)
	{
		double dec_value = bias;
		FeatureNode *x = probtest->x[i];
		while(x->index != -1)
		{
			int idx = x->index;
			if(idx <= w_size)
				dec_value += w[idx-1]*x->value;
			x++;
		}

		if(dec_value>0 && y[i]==1)
			correct++;
		else if(dec_value<=0 && y[i]==-1)
			correct++;
	}

	return (double)correct/l;
}

// w^T x
double evaluate_testing(double *w, int w_size, const Problem *probtest)
{
	return evaluate_testing_bias(w, w_size, 0.0, probtest);
}


// A coordinate descent algorithm for 
// L1-regularized L2-loss support vector classification
//
//  min_w \sum |wj| + C \sum max(0, 1-yi w^T xi)^2,
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Yuan et al. (2010) and appendix of LIBLINEAR paper, Fan et al. (2008)

#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l1r_l2_svc(
	Problem *prob_col, const Problem *probtest,
	double *w, double eps,
	double Cp, double Cn)
{
	int l = prob_col->l;
	int w_size = prob_col->n;
	int j, s, iter = 0;
	int max_iter = 100000;
	int active_size = w_size;
	int max_num_linesearch = 20;

	double sigma = 0.01;
	double d, G_loss, G, H;
	double Gmax_old = INF;
	double Gmax_new;
	double Gmax_init;
	double Gnorm1;
	double Gnorm1_init;
	double d_old, d_diff;
	double loss_old, loss_new;
	double appxcond, cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *b = new double[l]; // b = 1-ywTx
	double *xj_sq = new double[w_size];
	FeatureNode *x;

	double C[3] = {Cn,0,Cp};

	for(j=0; j<l; j++)
	{
		b[j] = 1;
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = -1;
	}
	for(j=0; j<w_size; j++)
	{
		w[j] = 0;
		index[j] = j;
		xj_sq[j] = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index;
			double val = x->value;
			x->value *= y[ind]; // x->value stores yi*xij
			xj_sq[j] += C[GETI(ind)]*val*val;
			x++;
		}
	}

	//XXX
	double total = 0;
	double start = clock();

	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1 = 0;

		for(j=0; j<active_size; j++)
		{
			int i = j+rand()%(active_size-j);
			Swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			j = index[s];
			G_loss = 0;
			H = 0;

			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index;
				if(b[ind] > 0)
				{
					double val = x->value;
					double tmp = C[GETI(ind)]*val;
					G_loss -= tmp*b[ind];
					H += tmp*val;
				}
				x++;
			}
			G_loss *= 2;

			G = G_loss;
			H *= 2;
			H = max(H, 1e-12);

			double Gp = G+1;
			double Gn = G-1;
			double violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					Swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1 += violation;

			// obtain Newton direction d
			if(Gp <= H*w[j])
				d = -Gp/H;
			else if(Gn >= H*w[j])
				d = -Gn/H;
			else
				d = -w[j];

			if(fabs(d) < 1.0e-12)
				continue;

			double delta = fabs(w[j]+d)-fabs(w[j]) + G*d;
			d_old = 0;
			int num_linesearch;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				d_diff = d_old - d;
				cond = fabs(w[j]+d)-fabs(w[j]) - sigma*delta;

				appxcond = xj_sq[j]*d*d + G_loss*d + cond;
				if(appxcond <= 0)
				{
					x = prob_col->x[j];
					while(x->index != -1)
					{
						b[x->index] += d_diff*x->value;
						x++;
					}
					break;
				}

				if(num_linesearch == 0)
				{
					loss_old = 0;
					loss_new = 0;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int ind = x->index;
						if(b[ind] > 0)
							loss_old += C[GETI(ind)]*b[ind]*b[ind];
						double b_new = b[ind] + d_diff*x->value;
						b[ind] = b_new;
						if(b_new > 0)
							loss_new += C[GETI(ind)]*b_new*b_new;
						x++;
					}
				}
				else
				{
					loss_new = 0;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int ind = x->index;
						double b_new = b[ind] + d_diff*x->value;
						b[ind] = b_new;
						if(b_new > 0)
							loss_new += C[GETI(ind)]*b_new*b_new;
						x++;
					}
				}

				cond = cond + loss_new - loss_old;
				if(cond <= 0)
					break;
				else
				{
					d_old = d;
					d *= 0.5;
					delta *= 0.5;
				}
			}

			w[j] += d;

			// recompute b[] if line search takes too many steps
			if(num_linesearch >= max_num_linesearch)
			{
				info("#");
				for(int i=0; i<l; i++)
					b[i] = 1;

				for(int i=0; i<w_size; i++)
				{
					if(w[i]==0) continue;
					x = prob_col->x[i];
					while(x->index != -1)
					{
						b[x->index] -= w[i]*x->value;
						x++;
					}
				}
			}
		}

		if(iter == 0)
		{
			Gmax_init = Gmax_new;
			Gnorm1_init = Gnorm1;
		}
		iter++;


		//if(iter % 10 == 0)
		//	info(".");

		//if(Gmax_new <= eps*Gmax_init)
		if(Gnorm1 <= eps*Gnorm1_init)
		{
			if(active_size == w_size)
				break;
			else
			{
				active_size = w_size;
				//info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	//XXX testing
	total += clock()-start;

	double v = 0;
	int nnz = 0;
	double acc = 0;
	double gc_norm = 0;

	if(probtest != NULL)
	{
		double tmpbias = probtest->bias;
		if(tmpbias == 0)
			acc = evaluate_testing(w, w_size, probtest);
		else
			acc = evaluate_testing_bias(w, w_size-1, tmpbias*w[w_size-1], probtest);
	}

	for(int j=0; j<w_size; j++)
	{
		G_loss = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index;
			if(b[ind] > 0)
			{
				double val = x->value;
				double tmp = C[GETI(ind)]*val;
				G_loss -= tmp*b[ind];
			}
			x++;
		}
		G = G_loss*2;

		if(w[j] > 0)
			gc_norm += (G+1)*(G+1);
		else if(w[j] < 0)
			gc_norm += (G-1)*(G-1);
		else if(fabs(G) > 1)
			gc_norm += (fabs(G)-1)*(fabs(G)-1);
	}
	gc_norm = sqrt(gc_norm);

	for(int j=0; j<w_size; j++)
	{
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	}
	for(int j=0; j<l; j++)
		if(b[j] > 0)
			v += C[y[j]+1]*b[j]*b[j];

	info("iter %d time %lf f %lf accuracy %lf nonzero %d n %d Gnorm %lf eps %lf\n",
		iter, total/CLOCKS_PER_SEC, v, acc, nnz, w_size, gc_norm, eps);

	for(j=0; j<w_size; j++)
	{
		x = prob_col->x[j];
		while(x->index != -1)
		{
			x->value *= y[x->index]; // restore x->value
			x++;
		}
	}

	/*
	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
	info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value

	double v = 0;
	int nnz = 0;
	for(j=0; j<w_size; j++)
	{
	x = prob_col->x[j];
	while(x->index != -1)
	{
	x->value *= prob_col->y[x->index]; // restore x->value
	x++;
	}
	if(w[j] != 0)
	{
	v += fabs(w[j]);
	nnz++;
	}
	}
	for(j=0; j<l; j++)
	if(b[j] > 0)
	v += C[y[j]]*b[j]*b[j];

	info("Objective value = %lf\n", v);
	info("#nonzeros/#features = %d/%d\n", nnz, w_size);
	*/

	delete [] index;
	delete [] y;
	delete [] b;
	delete [] xj_sq;
}

// A modified  coordinate descent newtion implementation for 
// L1-regularized L2-loss support vector classification based on LIBLINEAR 1.7
// The shrinking procedure is modified to be consistent with other parallel algorithms.
//
//  min_w \sum |wj| + C \sum max(0, 1-yi w^T xi)^2,
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//


#if 1
static void solve_l1r_l2_svc_cdn(
	Problem *prob_col, const Problem *probtest,
	double *w, double eps,
	double Cp, double Cn)
{//        cdn    
	//  add testing in_iteration  report  total_num_line_search, acc, model_nnz
	//  remove GETI(i)
	//  shrink after loop
	//  atomic for b.add b.set
	//  not using appro-line 
	// use 	 omp wall time
	const int l = prob_col->l;
	const int n = prob_col->n;
	const int max_iter = 1000000;
	const int max_num_linesearch = 20;
	const double sigma = 0.01;
	const double beta = 0.5;
	const double d_lower_bound = 1.0e-12;
	const double  h_lower_bound = 1.0e-12;
	info_save("cdn for L1-regularized L2-loss support vector classification.\nmax_num_linesearch %d nnz in train data %d\n\n",
		max_num_linesearch, gProb.nnz);

	double Gmax_old = INF;
	double Gmax_new;
	double Gmax_init;
	double Gnorm1;
	double Gnorm1_init;
	double d_old, d_diff;
	double loss_old, loss_new;
	double appxcond, cond;

	int *index = new int[n];
	schar *y = new schar[l];
	//double *b = new double[l];    // b = 1-ywTx
	cas_array<double> b(l);

	double *xj_sq = new double[n];
	FeatureNode *x;

	double C[3] = {Cn,0,Cp};

	for(int j=0; j<l; j++)
	{
		b[j] = 1;
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = -1;
	}

	for(int j=0; j<n; j++)
	{
		w[j] = 0;     
		index[j] = j;
		xj_sq[j] = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index;
			double val = x->value;
			x->value *= y[ind];              // x->value stores yi*xij
			xj_sq[j] += C[y[ind]+1]*val*val;
			x++;
		}
	}

	//XXX
	/*double total_time = 0;
	double start = clock();*/
	double total_time = 0;
	double start = omp_get_wtime();

	//bian test
	double obj = 0;
	int model_nnz = 0;
	double acc = 0;

	// for  shrink after loop
	bool *feature_status=new bool[n];
	double *violation_array=new double[n];
	memset(feature_status,1,sizeof(bool)*n);

	// reporting line search
	int total_num_line_search =0;

	int iter = 0;
	int active_size = n;
	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1 = 0;

		for(int s=0; s<active_size; s++)
		{
			int i = s+rand()%(active_size-s);
			Swap(index[i], index[s]);
		}

		for(int s=0; s<active_size; s++)
		{
			int j = index[s];
			double G = 0;
			double H = 0;

			FeatureNode *x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index;
				if(b[ind] > 0)
				{
					double val = x->value;
					double tmp = C[y[ind]+1]*val;
					G -= tmp*b[ind];
					H += tmp*val;
				}
				x++;
			}
			G *= 2;
			H *= 2;
			H = max(H, h_lower_bound);

			double Gp = G+1;
			double Gn = G-1;
			//double violation = 0;
			violation_array[s]=0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation_array[s] = -Gp;
				else if(Gn > 0)
					violation_array[s] = Gn;
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					//active_size--;
					//Swap(index[s], index[active_size]);
					//s--;
					feature_status[j]=0;
					continue;
				}
			}
			else if(w[j] > 0)
				violation_array[s] = fabs(Gp);
			else
				violation_array[s] = fabs(Gn);

			//Gmax_new = max(Gmax_new, violation_array[s]);
			//Gnorm1 += violation_array[s];

			// obtain Newton direction d
			double d=0;
			if(Gp <= H*w[j])
				d = -Gp/H;
			else if(Gn >= H*w[j])
				d = -Gn/H;
			else
				d = -w[j];

			if(fabs(d) < d_lower_bound)
				continue;


			//---------------begin one-dimensional Armijo--------------------
#pragma region Armijo

			double delta = fabs(w[j]+d)-fabs(w[j]) + G*d;
			double d_old = 0;
			double d_diff =0;
			double cond=0;
			double appxcond=0;
			double  loss_old = 0;
			double loss_new = 0;
			int num_linesearch;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				d_diff = d_old - d;
				cond = fabs(w[j]+d)-fabs(w[j]) - sigma*delta;

#if 0
				appxcond = xj_sq[j]*d*d + G*d + cond;
				if(appxcond <= 0)
				{
					x = prob_col->x[j];
					while(x->index != -1)
					{
						//b[x->index] += d_diff*x->value;
						b.add(x->index,d_diff*x->value);
						x++;
					}

					break;//    terminate 
				}
#endif

				if(num_linesearch == 0)
				{
					loss_old = 0;
					loss_new = 0;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int i = x->index;
						if(b[i] > 0)
							loss_old += C[y[i]+1]*b[i]*b[i];
						double b_new = b[i] + d_diff*x->value;
						//b[i] = b_new;
						b.set(i,b_new);
						if(b_new > 0)
							loss_new += C[y[i]+1]*b_new*b_new;
						x++;
					}
				}
				else
				{
					loss_new = 0;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int i = x->index;
						double b_new = b[i] + d_diff*x->value;
						//b[i] = b_new;
						b.set(i,b_new);
						if(b_new > 0)
							loss_new += C[y[i]+1]*b_new*b_new;
						x++;
					}
				}

				cond = cond + loss_new - loss_old;
				if(cond <= 0)
					break;
				else
				{
					d_old = d;
					d *= beta;
					delta *= beta;
				}
			}//   for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)

			total_num_line_search+=num_linesearch+1;//   treat recompute as one line search    atomic   needed  

			w[j] += d;

			// recompute b[] if line search takes too many steps
			if(num_linesearch >= max_num_linesearch)
			{
				info_save("#");
				for(int i=0; i<l; i++)
					//b[i] = 1;
					b.set(i,1);

				for(int j=0; j<n; j++)
				{
					if(w[j]==0) 
						continue;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						//b[x->index] -= w[j]*x->value;
						b.add(x->index,-w[j]*x->value);
						x++;
					}
				}
			}
#pragma endregion Armijo

			//--------------------end one-dimensional Armijo------------------
		}//  for(int s=0; s<active_size; s++)

		for (int s=0;s<active_size;s++)
		{
			Gmax_new = max(Gmax_new, violation_array[s]);
			Gnorm1 += violation_array[s];
		}
		for (int s=0;s<active_size;s++)
		{
			if (feature_status[index[s]]==0)
			{
				active_size--;
				Swap(index[s],index[active_size]);
				s--;
			}
		}

		if(iter == 0)
		{
			Gmax_init = Gmax_new;
			Gnorm1_init = Gnorm1;
			info_save("eps_end %lf\n\n",eps*Gnorm1_init);
		}

		//total_time+= clock()-start;
		total_time+=omp_get_wtime()-start;


		//--------------- start of in_iteration testing
#if 1
		obj=0; 
		model_nnz = 0;
		for(int j=0; j<n; j++)
		{
			if(w[j] != 0)
			{
				obj += fabs(w[j]);
				model_nnz++;
			}
		}
		for(int i=0; i<l; i++)
			if(b[i] > 0)
				obj += C[y[i]+1]*b[i]*b[i];

		if(probtest != NULL)
		{
			double tmpbias = probtest->bias;
			if(tmpbias < 0)
				acc = evaluate_testing(w, n, probtest);
			else
				acc = evaluate_testing_bias(w, n-1, tmpbias*w[n-1], probtest);
		}
		if(iter%10 == 0)
			info_save("iter %d time %lf  accuracy %lf f %lf model_nnz %d Gnorm1 %lf total_num_line_search %d \n",
			iter,total_time,  acc,obj,model_nnz, Gnorm1,total_num_line_search);

		save_exp("iter %d time %lf accuracy %lf f %lf model_nnz %d Gnorm1 %lf total_num_line_search %d \n",
			iter,total_time,acc,obj,model_nnz, Gnorm1,total_num_line_search);

		//info_save("iter %d active_size %d\n",iter,active_size);
#endif
		//-------------------  bian end of in_iteration testing

		//start=clock();
		start = omp_get_wtime();
		iter++;

		//if(iter % 10 == 0)
		//	info(".");

		//if(Gmax_new <= eps*Gmax_init)
		if(Gnorm1 <= eps*Gnorm1_init)
		{
			if(active_size == n)
				break;
			else
			{
				active_size = n;
				memset(feature_status,1,sizeof(bool)*n);
				info("*");
				Gmax_old = INF;
				continue;
			}
		}
		Gmax_old = Gmax_new;
	}//  while(iter < max_iter)


	//XXX testing
	//total_time += clock()-start;
	total_time += omp_get_wtime()-start;

	double v = 0;
	int nnz = 0;
	//	double acc = 0;
	double gc_norm = 0;

	if(probtest != NULL)
	{
		double tmpbias = probtest->bias;
		if(tmpbias == 0)
			acc = evaluate_testing(w, n, probtest);
		else
			acc = evaluate_testing_bias(w, n-1, tmpbias*w[n-1], probtest);
	}

	double G_loss;

	for(int j=0; j<n; j++)
	{
		G_loss = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index;
			if(b[ind] > 0)
			{
				double val = x->value;
				double tmp = C[y[ind]+1]*val;
				G_loss -= tmp*b[ind];
			}
			x++;
		}
		G_loss*=2;
		//G = G_loss;

		if(w[j] > 0)
			gc_norm += (G_loss+1)*(G_loss+1);
		else if(w[j] < 0)
			gc_norm += (G_loss-1)*(G_loss-1);
		else if(fabs(G_loss) > 1)
			gc_norm += (fabs(G_loss)-1)*(fabs(G_loss)-1);
	}
	gc_norm = sqrt(gc_norm);

	for(int j=0; j<n; j++)
	{
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	}
	for(int j=0; j<l; j++)
		if(b[j] > 0)
			v += C[y[j]+1]*b[j]*b[j];

	//info_save("#iter %d time %lf f %lf accuracy %lf model_nnz %d l %d n %d Gnorm  %lf Gnorm1 %lf eps_end %lf total_num_line_search %d\n",
	//	iter, total_time/CLOCKS_PER_SEC, v, acc, nnz, l, n, gc_norm, Gnorm1, eps*Gnorm1_init,total_num_line_search);
	info_save("#iter %d time %lf f %lf accuracy %lf model_nnz %d l %d n %d Gnorm  %lf Gnorm1 %lf eps_end %lf total_num_line_search %d\n",
		iter, total_time, v, acc, nnz, l, n, gc_norm, Gnorm1, eps*Gnorm1_init,total_num_line_search);

	for(int j=0; j<n; j++)
	{
		x = prob_col->x[j];
		while(x->index != -1)
		{
			x->value *= y[x->index]; // restore x->value
			x++;
		}
	}

	delete [] index;
	delete [] y;
	//delete [] b;
	delete [] xj_sq;
}
#else
static void solve_l1r_l2_svc_cdn(
	Problem *prob_col, const Problem *probtest,
	double *w, double eps,
	double Cp, double Cn)
{//   cdn    
	//  add testing in_iteration  report  total_num_line_search, acc, model_nnz
	//  remove GETI(i)
	//  shrink after loop
	// no atomic
	//  not using appro-line 
	// use 	cas_array<double> dTx(l);   
	//@TODO: experiment with dTx
	const bool appr_lineserch = false;
	const int l = prob_col->l;
	const int n = prob_col->n;
	const int max_iter = 100000;
	const int max_num_linesearch = 20;
	const double sigma = 0.01;
	const double beta = 0.5;
	const double d_lower_bound = 1.0e-12;
	const double  h_lower_bound = 1.0e-12;
	info_save("cdn for svc max_num_linesearch %d nnz in train data %d appr_lineserch %d\n",
		max_num_linesearch, gProb.nnz, appr_lineserch);

	double Gmax_old = INF;
	double Gmax_new;
	double Gmax_init;
	double Gnorm1;
	double Gnorm1_init;
	double d_old, d_diff;
	double loss_old, loss_new;
	double appxcond, cond;

	int *index = new int[n];
	schar *y = new schar[l];
	double *b = new double[l];    // b = 1-ywTx
	//cas_array<double> b(l);

	double *xj_sq = new double[n];
	FeatureNode *x;

	double C[3] = {Cn,0,Cp};

	for(int j=0; j<l; j++)
	{
		b[j] = 1;
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = -1;
	}

	for(int j=0; j<n; j++)
	{
		w[j] = 0;     
		index[j] = j;
		xj_sq[j] = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index;
			double val = x->value;
			x->value *= y[ind];              // x->value stores yi*xij
			xj_sq[j] += C[y[ind]+1]*val*val;
			x++;
		}
	}

	//XXX
	double total_time = 0;
	double start = clock();

	//bian test
	double obj = 0;
	int model_nnz = 0;
	double acc = 0;

	// for  shrink after loop
	bool *feature_status=new bool[n];
	double *violation_array=new double[n];
	memset(feature_status,1,sizeof(bool)*n);

	// reporting line search
	int total_num_line_search =0;

	//  
	cas_array<double> dTx(l);
	double *b_new=new double[l];

	int iter = 0;
	int active_size = n;
	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1 = 0;

		for(int s=0; s<active_size; s++)
		{
			int i = s+rand()%(active_size-s);
			Swap(index[i], index[s]);
		}

		for(int s=0; s<active_size; s++)
		{
			int j = index[s];
			double G = 0;
			double H = 0;
			memset(dTx.arr,0,sizeof(double)*l);//

			FeatureNode *x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index;
				if(b[ind] > 0)
				{
					double val = x->value;
					double tmp = C[y[ind]+1]*val;
					G -= tmp*b[ind];
					H += tmp*val;
				}
				x++;
			}
			G *= 2;
			H *= 2;
			H = max(H, h_lower_bound);

			double Gp = G+1;
			double Gn = G-1;
			//double violation = 0;
			violation_array[s]=0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation_array[s] = -Gp;
				else if(Gn > 0)
					violation_array[s] = Gn;
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					//active_size--;
					//Swap(index[s], index[active_size]);
					//s--;
					feature_status[j]=0;
					continue;
				}
			}
			else if(w[j] > 0)
				violation_array[s] = fabs(Gp);
			else
				violation_array[s] = fabs(Gn);

			//Gmax_new = max(Gmax_new, violation_array[s]);
			//Gnorm1 += violation_array[s];

			// obtain Newton direction d
			double d=0;
			if(Gp <= H*w[j])
				d = -Gp/H;
			else if(Gn >= H*w[j])
				d = -Gn/H;
			else
				d = -w[j];

			if(fabs(d) < d_lower_bound)
				continue;

			x = prob_col->x[j];
			while(x->index != -1)
			{
				int idx=x->index;
				dTx.add(idx,d*x->value);//dTx(i)+= sigma_j[d_(j)*X_(i,j)]    here has race!!
				x++;
			}

			//---------------begin one-dimensional Armijo--------------------
#pragma region Armijo

			double delta = fabs(w[j]+d)-fabs(w[j]) + G*d;
			//double d_old = 0;
			//double d_diff =0;
			double cond=0;
			//double appxcond=0;
			double  loss_old = 0;
			x = prob_col->x[j];
			while(x->index != -1)
			{
				int i = x->index;
				if(b[i] > 0)
					loss_old += C[y[i]+1]*b[i]*b[i];
				x++;
			}
			double loss_new = 0;

			int num_linesearch;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				//d_diff = d_old - d;
				cond = fabs(w[j]+d)-fabs(w[j]) - sigma*delta;

#if 0
				appxcond = xj_sq[j]*d*d + G*d + cond;
				if(appxcond <= 0)
				{
					x = prob_col->x[j];
					while(x->index != -1)
					{
						//b[x->index] += d_diff*x->value;
						b[x->index] += (1-beta)*dTx[x->index];
						//b.add(x->index,d_diff*x->value);
						x++;
					}

					break;//    terminate 
				}
#endif

				//if(num_linesearch == 0)
				//{
				//loss_old = 0;
				loss_new = 0;
				x = prob_col->x[j];
				while(x->index != -1)
				{
					int i = x->index;
					//if(b[i] > 0)
					//	loss_old += C[y[i]+1]*b[i]*b[i];
					//double b_new = b[i] + d_diff*x->value;
					b_new[i] = b[i] - dTx[i];
					//b[i] = b_new;
					//b.set(i,b_new);
					if(b_new[i] > 0)
						loss_new += C[y[i]+1]*b_new[i]*b_new[i];
					x++;
				}
				//}
				//else
				//{
				//loss_new = 0;
				//x = prob_col->x[j];
				//while(x->index != -1)
				//{
				//	int i = x->index;
				//	//double b_new = b[i] + d_diff*x->value;
				//	//double b_new = b[i] + (1-beta)*dTx[i];
				//	double b_new = b[i] + 0.5*dTx[i];
				//	b[i] = b_new;
				//	//b.set(i,b_new);
				//	if(b_new > 0)
				//		loss_new += C[y[i]+1]*b_new*b_new;
				//	x++;
				//}
				//}

				cond = cond + loss_new - loss_old;
				if(cond <= 0)
				{
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int i = x->index;
						b[i] = b_new[i];
						x++;
					}
					break;
				}

				else
				{
					//d_old = d;
					d *= beta;
					delta *= beta;	
					//if (num_linesearch > 0)
					//{
					x = prob_col->x[j];
					while(x->index != -1)
					{
						dTx[x->index]*=beta;
						x++;
					}
					//}

				}
			}//   for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)

			total_num_line_search+=num_linesearch+1;//   treat recompute as one line search    atomic   needed  

			w[j] += d;

			// recompute b[] if line search takes too many steps
			if(num_linesearch >= max_num_linesearch)
			{
				info_save("#");
				for(int i=0; i<l; i++)
					b[i] = 1;
				//b.set(i,1);

				for(int j=0; j<n; j++)
				{
					if(w[j]==0) 
						continue;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						b[x->index] -= w[j]*x->value;
						//b.add(x->index,-w[j]*x->value);
						x++;
					}
				}
			}
#pragma endregion Armijo

			//--------------------end one-dimensional Armijo------------------
		}//  for(int s=0; s<active_size; s++)

		for (int s=0;s<active_size;s++)
		{
			Gmax_new = max(Gmax_new, violation_array[s]);
			Gnorm1 += violation_array[s];
		}
		for (int s=0;s<active_size;s++)
		{
			if (feature_status[index[s]]==0)
			{
				active_size--;
				Swap(index[s],index[active_size]);
				s--;
			}
		}

		if(iter == 0)
		{
			Gmax_init = Gmax_new;
			Gnorm1_init = Gnorm1;
			info_save("eps_end %lf\n\n",eps*Gnorm1_init);
		}

		total_time+= clock()-start;

		//---------------  bian start of in_iteration testing
#if 1
		obj=0; 
		model_nnz = 0;
		for(int j=0; j<n; j++)
		{
			if(w[j] != 0)
			{
				obj += fabs(w[j]);
				model_nnz++;
			}
		}
		for(int i=0; i<l; i++)
			if(b[i] > 0)
				obj += C[y[i]+1]*b[i]*b[i];

		if(probtest != NULL)
		{
			double tmpbias = probtest->bias;
			if(tmpbias < 0)
				acc = evaluate_testing(w, n, probtest);
			else
				acc = evaluate_testing_bias(w, n-1, tmpbias*w[n-1], probtest);
		}
		if(iter%10 == 0)
			info_save("iter %d time %lf accuracy %lf f %lf model_nnz %d Gnorm1 %lf total_num_line_search %d \n",
			iter,total_time/CLOCKS_PER_SEC,acc,obj,model_nnz, Gnorm1,total_num_line_search);
		save_exp("iter %d time %lf accuracy %lf f %lf model_nnz %d Gnorm1 %lf total_num_line_search %d \n",
			iter,total_time/CLOCKS_PER_SEC,acc,obj,model_nnz, Gnorm1,total_num_line_search);

		info_save("iter %d active_size %d\n",iter,active_size);
#endif
		//-------------------  bian end of in_iteration testing

		start=clock();
		iter++;

		//if(iter % 10 == 0)
		//	info(".");

		//if(Gmax_new <= eps*Gmax_init)
		if(Gnorm1 <= eps*Gnorm1_init)
		{
			if(active_size == n)
				break;
			else
			{
				active_size = n;
				memset(feature_status,1,sizeof(bool)*n);
				info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}//  while(iter < max_iter)


	//XXX testing
	total_time += clock()-start;

	double v = 0;
	int nnz = 0;
	//	double acc = 0;
	double gc_norm = 0;

	if(probtest != NULL)
	{
		double tmpbias = probtest->bias;
		if(tmpbias < 0)
			acc = evaluate_testing(w, n, probtest);
		else
			acc = evaluate_testing_bias(w, n-1, tmpbias*w[n-1], probtest);
	}

	double G_loss;

	for(int j=0; j<n; j++)
	{
		G_loss = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index;
			if(b[ind] > 0)
			{
				double val = x->value;
				double tmp = C[y[ind]+1]*val;
				G_loss -= tmp*b[ind];
			}
			x++;
		}
		G_loss*=2;
		//G = G_loss;

		if(w[j] > 0)
			gc_norm += (G_loss+1)*(G_loss+1);
		else if(w[j] < 0)
			gc_norm += (G_loss-1)*(G_loss-1);
		else if(fabs(G_loss) > 1)
			gc_norm += (fabs(G_loss)-1)*(fabs(G_loss)-1);
	}
	gc_norm = sqrt(gc_norm);

	for(int j=0; j<n; j++)
	{
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	}
	for(int j=0; j<l; j++)
		if(b[j] > 0)
			v += C[y[j]+1]*b[j]*b[j];

	info_save("#iter %d time %lf f %lf accuracy %lf model_nnz %d l %d n %d Gnorm  %lf Gnorm1 %lf eps_end %lf total_num_line_search %d\n",
		iter, total_time/CLOCKS_PER_SEC, v, acc, nnz, l, n, gc_norm, Gnorm1, eps*Gnorm1_init,total_num_line_search);

	for(int j=0; j<n; j++)
	{
		x = prob_col->x[j];
		while(x->index != -1)
		{
			x->value *= y[x->index]; // restore x->value
			x++;
		}
	}

	delete [] index;
	delete [] y;
	delete [] b;
	delete [] xj_sq;
	delete [] b_new;
}

#endif

// A parallel  coordinate descent newton (PCDN) implementation for 
// L1-regularized L2-loss support vector classification 
//
//  min_w \sum |wj| + C \sum max(0, 1-yi w^T xi)^2,
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//

static void solve_l1r_l2_svc_pcdn_dtx_iiter(
	Problem *prob_col, const Problem *probtest, const int _bundle_size,
	double *w, double eps,
	double Cp, double Cn)
{//   pcdn    
	//  add testing in_iteration  report  total_num_line_search, acc, model_nnz
	//  shrink after loop    
	//  p-D line search   
	//not using dTx_nz_idx
	//not using cas_array 
	// using omp wall time
	// reporting total_num_line_search
	//  no adaptive of bundle size
	const int l = prob_col->l;
	const int n = prob_col->n;
	const int max_iter = 1000000;
	const int max_num_linesearch = 20;
	const double sigma = 0.01;
	const double beta = 0.5;
	const double d_lower_bound = 1.0e-12;
	const double  h_lower_bound = 1.0e-12;
	omp_set_num_threads(g_pcdn_thread_num);
	info_save("pcdn for L1-regularized L2-loss support vector classification.\n#threads %d bundle_size %d  max_num_linesearch %d nnz in train data %d\n\n",
			  g_pcdn_thread_num,_bundle_size, max_num_linesearch, prob_col->nnz);

	double Gmax_old = INF;
	double Gmax_new;
	double Gmax_init;
	double Gnorm1;
	double Gnorm1_init;
	double d_old, d_diff;
	double loss_old, loss_new;
	double appxcond, cond;

	int *index = new int[n];
	schar *y = new schar[l];
	double *b = new double[l];    // b = 1-ywTx
	double *xj_sq = new double[n];
	FeatureNode *x;

	double C[3] = {Cn,0,Cp};

	for(int i=0; i<l; i++)
	{
		b[i] = 1;
		if(prob_col->y[i] > 0)
			y[i] = 1;
		else
			y[i] = -1;
	}

	for(int j=0; j<n; j++)
	{
		w[j] = 0;     
		index[j] = j;
		xj_sq[j] = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index;
			double val = x->value;
			x->value *= y[ind];              // x->value stores yi*xij
			xj_sq[j] += C[y[ind]+1]*val*val;
			x++;
		}
	}

	//XXX
	double total_time = 0;
	double start = omp_get_wtime();

	//bian test
	double obj = 0;
	int model_nnz = 0;
	double acc = 0;

	// for  shrink after loop
	bool *feature_status=new bool[n];
	double *violation_array=new double[n];
	memset(feature_status,1,sizeof(bool)*n);

	// reporting line search
	int total_num_line_search =0;

	//  pcdn
	int bundle_size = _bundle_size;
	double *G_ptr=new double[n];
	double *d_ptr=new double[n];
	double bundle_d_norm=0;
	//double *d_diff_ptr=new double[n];
	//double *d_old_ptr=new double[n];
	//cas_array<double> dTx(l);
	double *dTx = new double[l];
	double *b_new=new double[l];


	int iter = 0;
	int iiter = 0;
	int active_size = n;
	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1 = 0;

		for(int s=0; s<active_size; s++)
		{
			int i = s+rand()%(active_size-s);
			Swap(index[i], index[s]);
		}

		int iiter_tmp = active_size/bundle_size;
		int feature_upper_id=(active_size/bundle_size)*bundle_size;
		if (active_size%bundle_size != 0)
		{
			feature_upper_id+=bundle_size;
			iiter_tmp++;
		}

		iiter+=iiter_tmp;

		for (int feature_id=0;feature_id<feature_upper_id;feature_id+=bundle_size)
		{
			//memset(dTx.arr,0,sizeof(double)*l);
			memset(dTx,0,sizeof(double)*l);
			bundle_d_norm=0;
#pragma omp parallel for reduction(+:bundle_d_norm)
			for(int s=feature_id; s<feature_id+bundle_size; s++)
			{
				if (s<active_size)
				{
					int j = index[s];
					//double G = 0;
					G_ptr[j]=0;
					double H = 0;

					FeatureNode *x = prob_col->x[j];
					while(x->index != -1)
					{
						int i = x->index;
						if(b[i] > 0)
						{
							double val = x->value;
							double tmp = C[y[i]+1]*val;
							G_ptr[j] -= tmp*b[i];
							H += tmp*val;
						}
						x++;
					}
					G_ptr[j] *= 2;
					H *= 2;
					H = max(H, h_lower_bound);

					double Gp = G_ptr[j]+1;
					double Gn = G_ptr[j]-1;
					//double violation = 0;
					violation_array[s]=0;
					d_ptr[j]=0;
					if(w[j] == 0)
					{
						if(Gp < 0)
							violation_array[s] = -Gp;
						else if(Gn > 0)
							violation_array[s] = Gn;
						else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
						{
							//active_size--;
							//Swap(index[s], index[active_size]);
							//s--;
							feature_status[j]=0;
							continue;
						}
					}
					else if(w[j] > 0)
						violation_array[s] = fabs(Gp);
					else
						violation_array[s] = fabs(Gn);

					//Gmax_new = max(Gmax_new, violation_array[s]);
					//Gnorm1 += violation_array[s];

					// obtain Newton direction d
					//double d=0;
					if(Gp <= H*w[j])
						d_ptr[j] = -Gp/H;
					else if(Gn >= H*w[j])
						d_ptr[j] = -Gn/H;
					else
						d_ptr[j] = -w[j];

					if(fabs(d_ptr[j]) < d_lower_bound)
					{
						d_ptr[j]=0;
						continue;
					}
					bundle_d_norm+=fabs(d_ptr[j]);

					x = prob_col->x[j];
					while(x->index != -1)
					{
						int i= x->index;
						//dTx.add(i,d_ptr[j]*x->value);//dTx(i)+= sigma_j[d_(j)*X_(i,j)]
#pragma omp atomic
						dTx[i]+= d_ptr[j]*x->value;
						x++;
					}

				}// if (s<active_size)
			}//  for(int s=feature_id; s<feature_id+bundle_size; s++)

			if (0 == bundle_d_norm)
			{
				continue;
			}

#if 0
			//#pragma omp parallel for
			for(int s=feature_id; s<feature_id+bundle_size; s++)
			{
				if (s<active_size)
				{
					int j=index[s];
					FeatureNode *x = prob_col->x[j];
					while(x->index != -1)
					{
						int i= x->index;
						//dTx.add(i,d_ptr[j]*x->value);//dTx(i)+= sigma_j[d_(j)*X_(i,j)]
						//#pragma omp atomic
						dTx[i]+= d_ptr[j]*x->value;
						x++;
					}
				}
			}
#endif

			//double d_old = 0;
			//double d_diff =0;
			double cond=0;
			//double appxcond=0;
			double  loss_old = 0;
			double loss_new = 0;
			int num_linesearch;

			double delta=0;
			double new_bundle_w_norm=0;
			double bundle_w_norm=0;

			for(int s=feature_id; s<feature_id+bundle_size; s++)
			{
				if (s<active_size)
				{
					int j=index[s];
					new_bundle_w_norm +=  fabs(w[j]+d_ptr[j]);
					bundle_w_norm += fabs(w[j]);
					delta+=	G_ptr[j]*d_ptr[j];
				}
			}

			delta += new_bundle_w_norm - bundle_w_norm;

			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				cond = new_bundle_w_norm - bundle_w_norm- sigma*delta;
				if(num_linesearch == 0)
				{
					loss_old = 0;
					loss_new = 0;
#pragma omp parallel for reduction(+:loss_old), reduction(+:loss_new)
					for (int i=0;i<l;i++)
					{  
						if(0 == dTx[i])
						{
							continue;
						}

						if(b[i] > 0)
						{
							loss_old += C[y[i]+1]*b[i]*b[i];
						}
						b_new[i] = b[i] - dTx[i];
						dTx[i]*=beta;
						if(b_new[i] > 0)
						{
							loss_new += C[y[i]+1]*b_new[i]*b_new[i];
						}
					}
				}
				else
				{
					loss_new = 0;
#pragma omp parallel for  reduction(+:loss_new)
					for (int i=0;i<l;i++)
					{  
						if(0 == dTx[i])
						{
							continue;
						}
						b_new[i] = b[i] - dTx[i];
						dTx[i]*=beta;
						if(b_new[i] > 0)
						{
							loss_new += C[y[i]+1]*b_new[i]*b_new[i];
						}
					}
				}

				cond = cond + loss_new - loss_old;
				if(cond <= 0)
				{
#pragma omp parallel for  
					for (int i=0;i<l;i++)
					{  
						if(0 == dTx[i])
							continue;	
						b[i]=b_new[i];
					}
					break;
				}
				else
				{
					new_bundle_w_norm=0;
					for(int s=feature_id; s<feature_id+bundle_size; s++)
					{
						if (s<active_size)
						{
							int j=index[s];
							d_ptr[j] *= beta;
							new_bundle_w_norm += fabs(w[j]+d_ptr[j]);
						}
					}
					delta *= beta;
					//for (int i=0;i<l;i++)
					//{  
					//	if(0==dTx[i])
					//		continue;	
					//	dTx[i]*=beta;
					//}
				}
			}//   for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			//#pragma  omp atomic
			total_num_line_search+=num_linesearch+1;//   treat recompute as one line search    atomic   needed  
			for(int s=feature_id; s<feature_id+bundle_size; s++)
			{
				if (s<active_size)
				{
					int j=index[s];
					w[j] += d_ptr[j];
				}
			}
			// recompute b[] if line search takes too many steps
#if 0
			if(num_linesearch >= max_num_linesearch)
			{
				info_save("#");
				for(int i=0; i<l; i++)
					b[i] = 1;

				for(int j=0; j<n; j++)
				{
					if(w[j]==0) 
						continue;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						b[x->index] -= w[j]*x->value;
						x++;
					}
				}
			}
#else
			if(num_linesearch >= max_num_linesearch)
			{
				info_save("#");
				for (int i=0;i<l;i++)
				{
					if(0 == dTx[i])
					{
						continue;
					}	
					b[i] = b[i] - dTx[i];
				}

			}
#endif

			//--------------------end one-dimensional Armijo------------------
			//}

			//}//  for(int s=0; s<active_size; s++)
		}


		for (int s=0;s<active_size;s++)
		{
			Gmax_new = max(Gmax_new, violation_array[s]);
			Gnorm1 += violation_array[s];
		}
		for (int s=0;s<active_size;s++)
		{
			if (feature_status[index[s]]==0)
			{
				active_size--;
				Swap(index[s],index[active_size]);
				s--;
			}
		}

		if(iter == 0)
		{
			Gmax_init = Gmax_new;
			Gnorm1_init = Gnorm1;
			info_save("eps_end %lf\n\n",eps*Gnorm1_init);
		}

		total_time+= omp_get_wtime()-start;

		//---------------   start of in_iteration testing
#if 1
		obj=0; 
		model_nnz = 0;
		for(int j=0; j<n; j++)
		{
			if(w[j] != 0)
			{
				obj += fabs(w[j]);
				model_nnz++;
			}
		}
		for(int i=0; i<l; i++)
			if(b[i] > 0)
				obj += C[y[i]+1]*b[i]*b[i];

		if(probtest != NULL)
		{
			double tmpbias = probtest->bias;
			if(tmpbias == 0)
				acc = evaluate_testing(w, n, probtest);
			else
				acc = evaluate_testing_bias(w, n-1, tmpbias*w[n-1], probtest);
		}
		if(iter%10 == 0)
			info_save("iter %d iiter %d time %lf accuracy %lf f %lf model_nnz %d Gnorm1 %lf total_num_line_search %d \n",
			iter,iiter,total_time,acc,obj,model_nnz, Gnorm1,total_num_line_search);

		save_exp("iter %d iiter %d time %lf accuracy %lf f %lf model_nnz %d Gnorm1 %lf total_num_line_search %d \n",
			iter,iiter,total_time,acc,obj,model_nnz, Gnorm1,total_num_line_search);

		//info_save("iter %d active_size %d\n",iter,active_size);
#endif
		//-------------------  bian end of in_iteration testing

		start=omp_get_wtime();
		iter++;

		//if(iter % 10 == 0)
		//	info(".");
		//if(Gmax_new <= eps*Gmax_init)
		if(Gnorm1 <= eps*Gnorm1_init)
		{
			if(active_size == n)
				break;
			else
			{
				active_size = n;
				memset(feature_status,1,sizeof(bool)*n);
				info("*");
				Gmax_old = INF;
				continue;
			}
		}
		Gmax_old = Gmax_new;
	}//  while(iter < max_iter)

	//XXX testing
	total_time += omp_get_wtime()-start;

	double v = 0;
	int nnz = 0;
	//	double acc = 0;
	double gc_norm = 0;

	if(probtest != NULL)
	{
		double tmpbias = probtest->bias;
		if(tmpbias == 0)
			acc = evaluate_testing(w, n, probtest);
		else
			acc = evaluate_testing_bias(w, n-1, tmpbias*w[n-1], probtest);
	}

	double G_loss;

	for(int j=0; j<n; j++)
	{
		G_loss = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index;
			if(b[ind] > 0)
			{
				double val = x->value;
				double tmp = C[y[ind]+1]*val;
				G_loss -= tmp*b[ind];
			}
			x++;
		}
		G_loss*=2;
		//G = G_loss;

		if(w[j] > 0)
			gc_norm += (G_loss+1)*(G_loss+1);
		else if(w[j] < 0)
			gc_norm += (G_loss-1)*(G_loss-1);
		else if(fabs(G_loss) > 1)
			gc_norm += (fabs(G_loss)-1)*(fabs(G_loss)-1);
	}
	gc_norm = sqrt(gc_norm);

	for(int j=0; j<n; j++)
	{
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	}
	for(int j=0; j<l; j++)
		if(b[j] > 0)
			v += C[y[j]+1]*b[j]*b[j];

	info_save("bundle_size %d #iter %d #iiter %d time %lf f %lf accuracy %lf model_nnz %d l %d n %d Gnorm  %lf Gnorm1 %lf eps_end %lf total_num_line_search %d\n",
		bundle_size,iter,iiter, total_time, v, acc, nnz, l, n, gc_norm, Gnorm1, eps*Gnorm1_init,total_num_line_search);

	for(int j=0; j<n; j++)
	{
		x = prob_col->x[j];
		while(x->index != -1)
		{
			x->value *= y[x->index]; // restore x->value
			x++;
		}
	}

	delete [] index;
	delete [] y;
	delete [] b;
	delete [] xj_sq;
	delete [] G_ptr;
	delete [] d_ptr;
	delete [] b_new;
	delete [] dTx;
}



// A coordinate descent algorithm for 
// L1-regularized logistic regression problems with bias term
//
//  min_w \sum |wj| + C \sum log(1+exp(-yi (w^T xi + bias))),
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w and bias
//
// See Yuan et al. (2010) and appendix of LIBLINEAR paper, Fan et al. (2008)

static void solve_l1r_lr_b(
	const Problem *prob_col, const Problem *probtest,
	double *w, double eps,
	double Cp, double Cn)
{//   cdn  不用 approximation line search
	// 
	int l = prob_col->l;
	int w_size = prob_col->n;
	int j, s, iter = 0;
	int max_iter = 100000;
	int active_size = w_size;
	int max_num_linesearch = 20;

	double x_min = 0;
	double sigma = 0.01;
	double d, G, H;
	double Gmax_old = INF;
	double Gmax_new;
	double Gmax_init;
	double Gnorm1;
	double Gnorm1_init;
	double sum1, appxcond1;
	double sum2, appxcond2;
	double cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *exp_wTx = new double[l];
	double *exp_wTx_new = new double[l];
	double *xj_max = new double[w_size];
	double *C_sum = new double[w_size];
	double *xjneg_sum = new double[w_size];
	double *xjpos_sum = new double[w_size];
	FeatureNode *x;

	//XXX
	double bias = 0;
	double neg_sum = 0;

	// To support weights for instances,
	// replace C[y[i]] with C[i].
	double C[2] = {Cn,Cp};

	for(j=0; j<l; j++)
	{
		exp_wTx[j] = 1;
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
		{
			y[j] = 0;
			neg_sum += C[y[j]];
		}
	}
	for(j=0; j<w_size; j++)
	{
		w[j] = 0;
		index[j] = j;
		xj_max[j] = 0;
		C_sum[j] = 0;
		xjneg_sum[j] = 0;
		xjpos_sum[j] = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index;
			double val = x->value;
			x_min = min(x_min, val);
			xj_max[j] = max(xj_max[j], val);
			C_sum[j] += C[y[ind]];
			if(y[ind] == 0)
				xjneg_sum[j] += C[y[ind]]*val;
			else
				xjpos_sum[j] += C[y[ind]]*val;
			x++;
		}
	}

	//XXX
	double total = 0;
	double start = clock();

	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1 = 0;

		for(j=0; j<active_size; j++)
		{
			int i = j+rand()%(active_size-j);
			Swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			j = index[s];
			sum1 = 0;
			sum2 = 0;
			H = 0;

			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index;
				double exp_wTxind = exp_wTx[ind];
				double tmp1 = x->value/(1+exp_wTxind);
				double tmp2 = C[y[ind]]*tmp1;
				double tmp3 = tmp2*exp_wTxind;
				sum2 += tmp2;
				sum1 += tmp3;
				H += tmp1*tmp3;
				x++;
			}

			G = -sum2 + xjneg_sum[j];

			double Gp = G+1;
			double Gn = G-1;
			double violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					Swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1 += violation;

			// obtain Newton direction d
			if(Gp <= H*w[j])
				d = -Gp/H;
			else if(Gn >= H*w[j])
				d = -Gn/H;
			else
				d = -w[j];

			if(fabs(d) < 1.0e-12)
				continue;

			d = min(max(d,-10.0),10.0);

			double delta = fabs(w[j]+d)-fabs(w[j]) + G*d;
			int num_linesearch;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				cond = fabs(w[j]+d)-fabs(w[j]) - sigma*delta;

				if(x_min >= 0)
				{
					double tmp = exp(d*xj_max[j]);
					appxcond1 = log(1+sum1*(tmp-1)/xj_max[j]/C_sum[j])*C_sum[j] + cond - d*xjpos_sum[j];
					appxcond2 = log(1+sum2*(1/tmp-1)/xj_max[j]/C_sum[j])*C_sum[j] + cond + d*xjneg_sum[j];
					if(min(appxcond1,appxcond2) <= 0)
					{
						x = prob_col->x[j];
						while(x->index != -1)
						{
							exp_wTx[x->index] *= exp(d*x->value);
							x++;
						}
						break;
					}
				}

				cond += d*xjneg_sum[j];

				int i = 0;
				x = prob_col->x[j];
				while(x->index != -1)
				{
					int ind = x->index;
					double exp_dx = exp(d*x->value);
					exp_wTx_new[i] = exp_wTx[ind]*exp_dx;
					cond += C[y[ind]]*log((1+exp_wTx_new[i])/(exp_dx+exp_wTx_new[i]));
					x++; i++;
				}

				if(cond <= 0)
				{
					int i = 0;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int ind = x->index;
						exp_wTx[ind] = exp_wTx_new[i];
						x++; i++;
					}
					break;
				}
				else
				{
					d *= 0.5;//  beta =0.5
					delta *= 0.5;
				}
			}

			w[j] += d;

			// recompute exp_wTx[] if line search takes too many steps
			if(num_linesearch >= max_num_linesearch)
			{
				info("#");
				for(int i=0; i<l; i++)
					exp_wTx[i] = bias;

				for(int i=0; i<w_size; i++)
				{
					if(w[i]==0) continue;
					x = prob_col->x[i];
					while(x->index != -1)
					{
						exp_wTx[x->index] += w[i]*x->value;
						x++;
					}
				}

				for(int i=0; i<l; i++)
					exp_wTx[i] = exp(exp_wTx[i]);
			}
		}//for(s=0; s<active_size; s++)
		printf("iter %d active_size %d\n",iter,active_size);

		/////////////////////////////////////////////////////////////////////////
		//XXX bias term
		do
		{
			G = 0;
			H = 0;

			for(int i=0; i<l; i++)
			{
				double exp_wTxind = exp_wTx[i];
				double tmp1 = 1/(1+exp_wTxind);
				double tmp2 = C[y[i]]*tmp1;
				double tmp3 = tmp2*exp_wTxind;
				G += tmp2;
				H += tmp1*tmp3;
			}

			G = -G + neg_sum;

			Gmax_new = max(Gmax_new, fabs(G));
			Gnorm1 += fabs(G);

			// obtain Newton direction d
			d = -G/H;

			if(fabs(d) < 1.0e-12)
				break;

			d = min(max(d,-10.0),10.0);

			double delta = G*d;
			int num_linesearch;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				cond = -sigma*delta + d*neg_sum;

				for(int i=0; i<l; i++)
				{
					double exp_d = exp(d);
					exp_wTx_new[i] = exp_wTx[i]*exp_d;
					cond += C[y[i]]*log((1+exp_wTx_new[i])/(exp_d+exp_wTx_new[i]));
				}

				if(cond <= 0)
				{
					for(int i=0; i<l; i++)
						exp_wTx[i] = exp_wTx_new[i];
					break;
				}
				else
				{
					d *= 0.5;
					delta *= 0.5;
				}
			}

			bias += d;

			// recompute exp_wTx[] if line search takes too many steps
			if(num_linesearch >= max_num_linesearch)
			{
				info("#");
				for(int i=0; i<l; i++)
					exp_wTx[i] = bias;

				for(int i=0; i<w_size; i++)
				{
					if(w[i]==0) continue;
					x = prob_col->x[i];
					while(x->index != -1)
					{
						exp_wTx[x->index] += w[i]*x->value;
						x++;
					}
				}

				for(int i=0; i<l; i++)
					exp_wTx[i] = exp(exp_wTx[i]);
			}

		} while(0);
		//XXX end of bias term

		/////////////////////////////////////////////////////////////////////////

		if(iter == 0)
		{
			Gmax_init = Gmax_new;
			Gnorm1_init = Gnorm1;
		}
		iter++;

		//if(iter % 10 == 0)
		//	info(".");

		//if(Gmax_new <= eps*Gmax_init)
		if(Gnorm1 <= eps*Gnorm1_init)
		{
			if(active_size == w_size)
				break;
			else
			{
				active_size = w_size;
				//info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	//XXX testing
	total += clock()-start;

	double v = 0;
	int nnz = 0;
	double acc = 0;
	double gc_norm = 0;

	if(probtest != NULL)
		acc = evaluate_testing_bias(w, w_size, bias, probtest);

	for(int j=0; j<w_size; j++)
	{
		sum2 = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index;
			double tmp2 = C[y[ind]]*x->value/(1+exp_wTx[ind]);
			sum2 += tmp2;
			x++;
		}
		G = -sum2 + xjneg_sum[j];

		if(w[j] > 0)
			gc_norm += (G+1)*(G+1);
		else if(w[j] < 0)
			gc_norm += (G-1)*(G-1);
		else if(fabs(G) > 1)
			gc_norm += (fabs(G)-1)*(fabs(G)-1);
	}

	G = 0;
	for(int i=0; i<l; i++)
	{
		double exp_wTxind = exp_wTx[i];
		double tmp1 = 1/(1+exp_wTxind);
		double tmp2 = C[y[i]]*tmp1;
		G += tmp2;
	}
	G = -G + neg_sum;
	gc_norm += G*G;
	gc_norm = sqrt(gc_norm);

	for(j=0; j<w_size; j++)
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
		for(j=0; j<l; j++)
			if(y[j] == 1)
				v += C[y[j]]*log(1+1/exp_wTx[j]);
			else
				v += C[y[j]]*log(1+exp_wTx[j]);

		info("iter %d time %lf f %lf accuracy %lf nonzero %d n %d Gnorm %lf eps %lf bias %g\n",
			iter, total/CLOCKS_PER_SEC, v, acc, nnz, w_size, gc_norm, eps, bias);

		/*
		info("\noptimization finished, #iter = %d\n", iter);
		if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

		// calculate objective value

		double v = 0;
		int nnz = 0;
		for(j=0; j<n; j++)
		if(w[j] != 0)
		{
		v += fabs(w[j]);
		nnz++;
		}
		for(j=0; j<l; j++)
		if(y[j] == 1)
		v += C[y[j]]*log(1+1/exp_wTx[j]);
		else
		v += C[y[j]]*log(1+exp_wTx[j]);

		info("Objective value = %lf\n", v);
		info("#nonzeros/#features = %d/%d\n", nnz, n);
		*/

		delete [] index;
		delete [] y;
		delete [] exp_wTx;
		delete [] exp_wTx_new;
		delete [] xj_max;
		delete [] C_sum;
		delete [] xjneg_sum;
		delete [] xjpos_sum;
}
// A modified coordinate descent newton implementation for 
// L1-regularized logistic regression problems with bias term
// The shrinking procedure is modified so that it is consistent with the other parallel algorithms.
//
//  min_w \sum |wj| + C \sum log(1+exp(-yi (w^T xi + bias))),
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w and bias

static void solve_l1r_lr_b_cdn(
	const Problem *prob_col, const Problem *probtest,
	double *w, double eps,
	double Cp, double Cn, double *model_bias)
{// bias  cdn       not using  approximation line search
	//  shrink after feature loop
	// use omp wall time
	//use new more efficient recomputing procedure  
	const int l = prob_col->l;
	const int n = prob_col->n;

	const int max_iter = 1000000;
	const int max_num_linesearch = 20;
	const double sigma = 0.01;
	const double beta = 0.5;
	const double d_lower_bound = 1.0e-12;
	info_save("cdn for l1 regularized logistic regression with bias.\nmax_num_linesearch %d nnz in train data %d \n\n",
		max_num_linesearch,prob_col->nnz);

	double x_min = 0;
	double Gmax_old = INF;
	double Gmax_new;
	double Gmax_init;
	double Gnorm1;
	double Gnorm1_init;
	//double sum1, appxcond1;
	//double sum2, appxcond2;
	//double cond;

	int *index = new int[n];
	schar *y = new schar[l];//   0, 1
	//double *exp_wTx = new double[l];//  atomic 
	cas_array<double> exp_wTx(l);
	//double *exp_wTx_new = new double[l];//  //  atomic 
	cas_array<double> exp_wTx_new(l);//  atomic
	//cas_array<double> wTx(l);//  not used

	double *xj_max = new double[n];//  no data race
	double *C_sum = new double[n];//  no data race   for approximate line search
	double *xjneg_sum = new double[n];//  no data race
	double *xjpos_sum = new double[n];//  no data race
	//feature_node *x;

	//  use tau and D 
	//double *tau = new double[l];//  derivative of the logistic loss function
	//double *D = new double[l];  

	//---bias
	double bias = 0;
	double neg_sum = 0;


	//bian test
	double obj = 0;
	int model_nnz = 0;
	double acc = 0;

	// for  shrink after loop
	bool *feature_status=new bool[n];
	double *violation_array=new double[n];
	memset(feature_status,1,sizeof(bool)*n);


	// To support weights for instances,
	const double C[2] = {Cn,Cp};

	for(int i=0; i<l; i++)
	{
		exp_wTx[i] = 1;
		if(prob_col->y[i] > 0)
			y[i] = 1;
		else
		{
			y[i] = 0;
			neg_sum += C[y[i]];
		}
	}
	for(int j=0; j<n; j++)
	{
		w[j] = 0;
		index[j] = j;
		xj_max[j] = 0;
		C_sum[j] = 0;
		xjneg_sum[j] = 0;
		xjpos_sum[j] = 0;
		FeatureNode *x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index;
			double val = x->value;
			x_min = min(x_min, val);
			xj_max[j] = max(xj_max[j], val);
			C_sum[j] += C[y[ind]];
			if(y[ind] == 0)
				xjneg_sum[j] += C[y[ind]]*val;
			else
				xjpos_sum[j] += C[y[ind]]*val;
			x++;
		}
	}

	//XXX
	//double total_time = 0;
	//double start = clock();
	double total_time = 0;
	double start = omp_get_wtime();

	//   reporting line search
	int total_num_line_search =0;

	int iter = 0;
	int active_size = n;
	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1 = 0;

		for(int s=0; s<active_size; s++)
		{
			int i = s+rand()%(active_size-s);
			Swap(index[i], index[s]);
			//swap(feature_status[i],feature_status[j]);//   
		}

		//int feature_upper_id=(active_size/NUM_THREADS)*NUM_THREADS;
		//if (active_size%NUM_THREADS != 0)
		//{
		//feature_upper_id+=NUM_THREADS;
		//}

		//for (int feature_id=0;feature_id<feature_upper_id;feature_id+=NUM_THREADS)
		//	{
		for(int s=0; s<active_size; s++)
		{
			int j = index[s];
			double sum1 = 0;// *
			double sum2 = 0;//*
			double H = 0;

			FeatureNode *x = prob_col->x[j];//  feature j的所有sample
			while(x->index != -1)
			{
				int ind = x->index;
				double exp_wTxind = exp_wTx[ind];//  exp_wTx!!
				double tmp1 = x->value/(1+exp_wTxind);
				double tmp2 = C[y[ind]]*tmp1;
				double tmp3 = tmp2*exp_wTxind;
				sum2 += tmp2;
				sum1 += tmp3;
				H += tmp1*tmp3;
				x++;
			}

			double G = -sum2 + xjneg_sum[j];

			double Gp = G+1;
			double Gn = G-1;
			//double violation = 0;
			violation_array[s]=0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation_array[s] = -Gp;
				else if(Gn > 0)
					violation_array[s] = Gn;
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					/*						active_size--;
					swap(index[s], index[active_size]);
					s--;*/
					feature_status[j]=0;
					continue;
				}
			}
			else if(w[j] > 0)
				violation_array[s] = fabs(Gp);
			else
				violation_array[s] = fabs(Gn);

			//Gmax_new = max(Gmax_new, violation_array[s]);//   atomic 
			//Gnorm1 += violation_array[s];  //   reduction

			// obtain Newton direction d
			double d=0;
			if(Gp <= H*w[j])
				d = -Gp/H;
			else if(Gn >= H*w[j])
				d = -Gn/H;
			else
				d = -w[j];

			if(fabs(d) < d_lower_bound)
				continue;

			d = min(max(d,-10.0),10.0);
			//----------------------------------line search------------------------------------------
			double delta = fabs(w[j]+d)-fabs(w[j]) + G*d;
			int num_linesearch=0;
			double cond=0;
			double appxcond1=0;
			double appxcond2=0;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				cond = fabs(w[j]+d)-fabs(w[j]) - sigma*delta;

#if 0
				if (appr_lineserch)
				{
					if(x_min >= 0)//  
					{
						//info_save("$");
						double tmp = exp(d*xj_max[j]);
						appxcond1 = log(1+sum1*(tmp-1)/xj_max[j]/C_sum[j])*C_sum[j] + cond - d*xjpos_sum[j];
						appxcond2 = log(1+sum2*(1/tmp-1)/xj_max[j]/C_sum[j])*C_sum[j] + cond + d*xjneg_sum[j];
						if(min(appxcond1,appxcond2) <= 0)
						{
							x = prob_col->x[j];
							while(x->index != -1)
							{//exp_wTx[x->index] *= exp(d*x->value);
								exp_wTx.mul(x->index,exp(d*x->value));
								x++;
							}
							num_linesearch++;
							break;
						}
					}
				}

#endif
				cond += d*xjneg_sum[j];
				//info_save("iter %d feature %d cond %lf\n",iter,j,cond);
				//int i = 0;
				x = prob_col->x[j];
				while(x->index != -1)
				{
					int ind = x->index;
					double exp_dx = exp(d*x->value);
					//info_save("idx %d %g ",ind,exp_dx);
					//exp_wTx_new[i] = exp_wTx[ind]*exp_dx;//  race
					exp_wTx_new.set(ind,exp_wTx[ind]*exp_dx);//  race   

					cond += C[y[ind]]*log((1+exp_wTx_new[ind])/(exp_dx+exp_wTx_new[ind]));
					x++; 
					//i++;
				}
				//info_save("cond2: %d\n",cond);
				if(cond <= 0)
				{
					//int i = 0;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int ind = x->index;
						exp_wTx.set(ind, exp_wTx_new[ind]);
						x++; 
						//i++;
					}
					//num_linesearch++;
					break;
				}
				else
				{
					d *= beta;//  beta =0.5
					delta *= beta;
				}
			}//  for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			//   reporting line search
			total_num_line_search += num_linesearch+1;		//  treat recompute as one search
			w[j] += d;//   no data race,

			// recompute exp_wTx[] if line search takes too many steps
#if 1
			if (num_linesearch >= max_num_linesearch)
			{//  haha  
				info_save("#");// 
				//int i = 0;
				x = prob_col->x[j];
				while(x->index != -1)
				{
					int ind = x->index;
					double exp_dx = exp(d*x->value);
					//exp_wTx_new[i] = exp_wTx[ind]*exp_dx;//  race
					exp_wTx.set(ind,exp_wTx[ind]*exp_dx);//  race  
					x++; 
					//i++;
				}
			}
#else				
			if(num_linesearch >= max_num_linesearch)
			{//   
				info_save("#");//   
				for(int i=0; i<l; i++)
					//exp_wTx[i] = bias;//  race
					exp_wTx.set(i, bias);

				for(int j=0; j<n; j++)
				{
					if(w[j]==0) 
						continue;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						//exp_wTx[x->index] += w[i]*x->value;
						exp_wTx.add(x->index,w[j]*x->value);
						x++;
					}
				}

				for(int i=0; i<l; i++)
					//exp_wTx[i] = exp(exp_wTx[i]);
					exp_wTx.set(i, exp(exp_wTx[i]));
			}// if(num_linesearch >= max_num_linesearch)
#endif

			//------------------------------------------end of line search
			//}

		}//   for (  active_size)
		//}//

		for (int s=0;s<active_size;s++)
		{
			Gmax_new = max(Gmax_new, violation_array[s]);
			Gnorm1 += violation_array[s];
		}
		for (int s=0;s<active_size;s++)
		{
			if (feature_status[index[s]]==0)
			{
				active_size--;
				Swap(index[s],index[active_size]);
				s--;
			}
		}

		/////////////////////////////////////////////////////////////////////////
		//XXX bias term
		do
		{
			double G = 0;
			double H = 0;

			for(int i=0; i<l; i++)
			{
				double exp_wTxind = exp_wTx[i];
				double tmp1 = 1/(1+exp_wTxind);
				double tmp2 = C[y[i]]*tmp1;
				double tmp3 = tmp2*exp_wTxind;
				G += tmp2;
				H += tmp1*tmp3;
			}

			G = -G + neg_sum;

			Gmax_new = max(Gmax_new, fabs(G));
			Gnorm1 += fabs(G);

			// obtain Newton direction d
			double d=0;
			d = -G/H;

			if(fabs(d) < d_lower_bound)
				break;//   为了 break才用 do while

			d = min(max(d,-10.0),10.0);

			double delta = G*d;
			int num_linesearch = 0;
			double cond=0;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				cond = -sigma*delta + d*neg_sum;

				for(int i=0; i<l; i++)
				{
					double exp_d = exp(d);
					exp_wTx_new[i] = exp_wTx[i]*exp_d;
					cond += C[y[i]]*log((1+exp_wTx_new[i])/(exp_d+exp_wTx_new[i]));
				}

				if(cond <= 0)
				{
					for(int i=0; i<l; i++)
						exp_wTx[i] = exp_wTx_new[i];
					break;
				}
				else
				{
					d *= beta;
					delta *= beta;
				}
			}

			bias += d;

			// recompute exp_wTx[] if line search takes too many steps

			if(num_linesearch >= max_num_linesearch)
			{
				info_save("#-b");
				for(int i=0; i<l; i++)
					exp_wTx[i] = bias;
				for(int i=0; i<n; i++)
				{
					if(w[i]==0) 
						continue;
					FeatureNode * x = prob_col->x[i];
					while(x->index != -1)
					{
						exp_wTx[x->index] += w[i]*x->value;
						x++;
					}
				}
				for(int i=0; i<l; i++)
					exp_wTx[i] = exp(exp_wTx[i]);
			}

		} while(0);
		//XXX end of bias term
		if(iter == 0)
		{
			Gmax_init = Gmax_new;
			Gnorm1_init = Gnorm1;
			info_save("eps_end %lf\n\n",eps*Gnorm1_init);
		}
		/////////////////////////////////////////////////////////////////////////
		total_time+= omp_get_wtime()-start;
		//---------------  bian start of in_iteration testing
		obj=0; 
		model_nnz = 0;
		for (int j=0;j<n;j++)
		{
			if (w[j] != 0)
			{
				obj += fabs(w[j]);
				model_nnz ++;
			}
		}
		for (int i = 0;i<l;i++)
		{
			if (y[i] == 1)
				obj+= C[y[i]]*log(1+1/exp_wTx[i]);
			else
				obj+=C[y[i]]* log(1+exp_wTx[i]);

		}
		if(probtest != NULL)
			acc = evaluate_testing_bias(w, n, bias, probtest);
		if(iter%10 == 0)
			info_save("iter %d time %lf accuracy %lf f %lf model_nnz %d Gnorm1 %lf total_num_line_search %d\n",
			iter,total_time,acc,obj,model_nnz, Gnorm1,total_num_line_search);

		save_exp("iter %d time %lf accuracy %lf f %lf model_nnz %d Gnorm1 %lf total_num_line_search %d\n",
			iter,total_time,acc,obj,model_nnz, Gnorm1,total_num_line_search);

		//info_save("iter %d active_size %d\n",iter,active_size);
		//-------------------  bian end of in_iteration testing
		start=omp_get_wtime();
		iter++;
		//if(iter % 10 == 0)
		//	info(".");
		//if(Gmax_new <= eps*Gmax_init)
		if(Gnorm1 <= eps*Gnorm1_init)//  结束条件
		{
			if(active_size == n)
				break;//   
			else
			{
				active_size = n;
				info("*");
				memset(feature_status,1,sizeof(bool)*n);
				Gmax_old = INF;
				continue;
			}
		}
		Gmax_old = Gmax_new;
	}//   while(iter<max_iter)

	total_time += omp_get_wtime()-start;

	//XXX testing
	//double v = 0;
	//nnz = 0;
	//double acc = 0;
	double gc_norm = 0;

	//if(probtest != NULL)
	//	acc = evaluate_testing_bias(w, n, bias, probtest);
	double sum2=0;
	double G=0;
	for(int j=0; j<n; j++)
	{
		sum2 = 0;
		FeatureNode *x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index;
			double tmp2 = C[y[ind]]*x->value/(1+exp_wTx[ind]);
			sum2 += tmp2;
			x++;
		}
		G = -sum2 + xjneg_sum[j];

		if(w[j] > 0)
			gc_norm += (G+1)*(G+1);
		else if(w[j] < 0)
			gc_norm += (G-1)*(G-1);
		else if(fabs(G) > 1)
			gc_norm += (fabs(G)-1)*(fabs(G)-1);
	}

	G = 0;
	for(int i=0; i<l; i++)
	{
		double exp_wTxind = exp_wTx[i];
		double tmp1 = 1/(1+exp_wTxind);
		double tmp2 = C[y[i]]*tmp1;
		G += tmp2;
	}
	G = -G + neg_sum;
	gc_norm += G*G;//   
	gc_norm = sqrt(gc_norm);

	//for(j=0; j<n; j++)
	//	if(w[j] != 0)
	//	{
	//		v += fabs(w[j]);
	//		nnz++;
	//	}
	//for(j=0; j<l; j++)
	//	if(y[j] == 1)
	//		v += C[y[j]]*log(1+1/exp_wTx[j]);
	//	else
	//		v += C[y[j]]*log(1+exp_wTx[j]);
	*model_bias = bias;
	info_save("#iter %d time %lf f %lf accuracy %lf model_nnz %d l %d n %d Gcnorm %g Gnorm1 %g eps_end %g bias %g total_num_line_search %d\n",
		iter, total_time, obj, acc, model_nnz, l, n, gc_norm, Gnorm1, eps*Gnorm1_init, bias,total_num_line_search);

	delete [] index;
	delete [] y;
	//	delete [] exp_wTx;
	//delete [] exp_wTx_new;
	delete [] xj_max;
	delete [] C_sum;
	delete [] xjneg_sum;
	delete [] xjpos_sum;
	delete [] feature_status;
	delete [] violation_array;
}
// A new Shotgun Coordinate Descent Newton implementation for 
// L1-regularized logistic regression problems with bias term
//
//  min_w \sum |wj| + C \sum log(1+exp(-yi (w^T xi + bias))),
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w and bias

static void solve_l1r_lr_b_scdn(
	const Problem *prob_col, const Problem *probtest,const Problem *prob,
	double *w, double eps,
	double Cp, double Cn, double *model_bias)
{//   scdn      
	//no using approximation line search
	//  if using omp， it is shotgun-cdn
	//  if not using omp，it is cdn，shrink outside loop
	//   use recompute consistent with implementation of shotgun-cdn
	// use omp time
	const int l = prob_col->l;
	const int n = prob_col->n;

	const int max_iter = 1000000;
	const int max_num_linesearch = 20;
	const double sigma = 0.01;
	const double beta = 0.5;
	const double d_lower_bound = 1.0e-12;
	omp_set_num_threads(g_scdn_thread_num);
	info_save("scdn for l1 regularized logistic regression with bias.\n");
	info_save("num_threads %d  max_num_linesearch: %d  nnz in train data %d\n\n",
			  g_scdn_thread_num,max_num_linesearch,prob_col->nnz);

	double x_min = 0;

	//double d, G, H;
	double Gmax_old = INF;
	double Gmax_new;
	double Gmax_init;
	double Gnorm1;
	double Gnorm1_init;

	int *index = new int[n];
	schar *y = new schar[l];//   0, 1
	//double *exp_wTx = new double[l];//  atomic 
	cas_array<double> exp_wTx(l);
	//double *exp_wTx_new = new double[l];//  //  atomic 
	cas_array<double> exp_wTx_new(l);//  atomic
	cas_array<double> wTx(l);//   used for recompute

	double *xj_max = new double[n];//  no data race
	double *C_sum = new double[n];//  no data race
	double *xjneg_sum = new double[n];//  no data race
	double *xjpos_sum = new double[n];//  no data race
	//feature_node *x;


	//XXX
	double bias = 0;
	double neg_sum = 0;


	//bian test
	double obj = 0;
	int model_nnz = 0;
	double acc = 0;

	// for  omp parallel
	bool *feature_status=new bool[n];
	double *violation_array=new double[n];
	memset(feature_status,1,sizeof(bool)*n);


	// To support weights for instances,
	// replace C[y[i]] with C[i].
	const double C[2] = {Cn,Cp};

	for(int i=0; i<l; i++)
	{
		exp_wTx[i] = 1;
		if(prob_col->y[i] > 0)
			y[i] = 1;
		else
		{
			y[i] = 0;
			neg_sum += C[y[i]];
		}
	}
	for(int j=0; j<n; j++)
	{
		w[j] = 0;
		index[j] = j;
		xj_max[j] = 0;
		C_sum[j] = 0;
		xjneg_sum[j] = 0;
		xjpos_sum[j] = 0;
		FeatureNode *x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index;
			double val = x->value;
			x_min = min(x_min, val);
			xj_max[j] = max(xj_max[j], val);
			C_sum[j] += C[y[ind]];
			if(y[ind] == 0)
				xjneg_sum[j] += C[y[ind]]*val;
			else
				xjpos_sum[j] += C[y[ind]]*val;
			x++;
		}
	}

	//XXX
	//double total_time = 0;
	//double start = clock();
	double total_time = 0;
	double start = omp_get_wtime();


	//   reporting line search
	int total_num_line_search =0;

	int iter = 0;
	int active_size = n;
	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1 = 0;

		for(int s=0; s<active_size; s++)
		{
			int i = s+rand()%(active_size-s);
			Swap(index[i], index[s]);
		}

#pragma omp parallel for  
		for(int s=0; s<active_size; s++)
		{

			int j = index[s];
			double sum1 = 0;// *
			double sum2 = 0;//*
			double H = 0;

			FeatureNode *x = prob_col->x[j];//  feature j的所有sample
			while(x->index != -1)
			{
				int ind = x->index;
				double exp_wTxind = exp_wTx[ind];//  exp_wTx!!
				double tmp1 = x->value/(1+exp_wTxind);
				double tmp2 = C[y[ind]]*tmp1;
				double tmp3 = tmp2*exp_wTxind;
				sum2 += tmp2;
				sum1 += tmp3;
				H += tmp1*tmp3;
				x++;
			}

			double G = -sum2 + xjneg_sum[j];

			double Gp = G+1;
			double Gn = G-1;
			//double violation = 0;
			violation_array[s]=0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation_array[s] = -Gp;
				else if(Gn > 0)
					violation_array[s] = Gn;
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					/*						active_size--;
					swap(index[s], index[active_size]);
					s--;*/
					feature_status[j]=0;
					continue;
				}
			}
			else if(w[j] > 0)
				violation_array[s] = fabs(Gp);
			else
				violation_array[s] = fabs(Gn);

			//Gmax_new = max(Gmax_new, violation_array[s]);//   atomic 
			//Gnorm1 += violation_array[s];  //   reduction

			// obtain Newton direction d
			double d=0;
			if(Gp <= H*w[j])
				d = -Gp/H;
			else if(Gn >= H*w[j])
				d = -Gn/H;
			else
				d = -w[j];

			if(fabs(d) < d_lower_bound)
				continue;

			d = min(max(d,-10.0),10.0);
			//----------------------------------line search------------------------------------------
			double delta = fabs(w[j]+d)-fabs(w[j]) + G*d;
			int num_linesearch=0;
			double cond=0;
			double appxcond1=0;
			double appxcond2=0;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				cond = fabs(w[j]+d)-fabs(w[j]) - sigma*delta;

#if 0// 
				if(x_min >= 0)//  
				{
					//info_save("*");
					double tmp = exp(d*xj_max[j]);
					appxcond1 = log(1+sum1*(tmp-1)/xj_max[j]/C_sum[j])*C_sum[j] + cond - d*xjpos_sum[j];
					appxcond2 = log(1+sum2*(1/tmp-1)/xj_max[j]/C_sum[j])*C_sum[j] + cond + d*xjneg_sum[j];
					if(min(appxcond1,appxcond2) <= 0)
					{
						x = prob_col->x[j];
						while(x->index != -1)
						{//  feature j 
							exp_wTx.mul(x->index,exp(d*x->value));
							x++;
						}
						break;
					}
				}
#endif

				cond += d*xjneg_sum[j];
				//info_save("iter %d feature %d cond %lf\n",iter,j,cond);
				//int i = 0;
				x = prob_col->x[j];
				while(x->index != -1)
				{
					int ind = x->index;
					double exp_dx = exp(d*x->value);
					//info_save("idx %d %g ",ind,exp_dx);
					//exp_wTx_new[i] = exp_wTx[ind]*exp_dx;//  race
					exp_wTx_new.set(ind,exp_wTx[ind]*exp_dx);//  race  

					cond += C[y[ind]]*log((1+exp_wTx_new[ind])/(exp_dx+exp_wTx_new[ind]));
					x++; 
				}
				//info_save("cond2: %d\n",cond);
				if(cond <= 0)
				{
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int i = x->index;
						//exp_wTx[ind] = exp_wTx_new[i];// race
						exp_wTx.set(i, exp_wTx_new[i]);
						x++; 
					}
					//num_linesearch++;
					break;
				}
				else
				{
					d *= beta;//  beta =0.5
					delta *= beta;
				}
			}//  for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
#pragma omp atomic
			total_num_line_search+=num_linesearch+1;
			w[j] += d;//   no data race, 如果超出限制，无论如何接受

			// recompute exp_wTx[] if line search takes too many steps
#if 0
			if (num_linesearch >= max_num_linesearch)
			{//  haha  
				info_save("#");// 
				x = prob_col->x[j];
				while(x->index != -1)
				{
					int ind = x->index;
					double exp_dx = exp(d*x->value);
					//exp_wTx_new[i] = exp_wTx[ind]*exp_dx;//  race
					exp_wTx.set(ind,exp_wTx[ind]*exp_dx);//  race  
					x++; 
				}

			}

#else				
#if 0
			if(num_linesearch >= max_num_linesearch)
			{//   似乎可以再优化
				//  has Problem for scdn, because exp_wTx cannot be used as wTx
				info_save("#");//   
				for(int i=0; i<l; i++)
					//exp_wTx[i] = bias;//  race
					exp_wTx.set(i, bias);

				for(int j=0; j<n; j++)
				{
					if(w[j]==0) 
						continue;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						//exp_wTx[x->index] += w[i]*x->value;
						exp_wTx.add(x->index,w[j]*x->value);
						x++;
					}
				}

				for(int i=0; i<l; i++)
					//exp_wTx[i] = exp(exp_wTx[i]);
					exp_wTx.set(i, exp(exp_wTx[i]));
			}// if(num_linesearch >= max_num_linesearch)
#else
			if(num_linesearch >= max_num_linesearch)
			{//       
				//  has problem for scdn, because exp_wTx cannot be used as wTx
				info_save("#");// 
				for (int i=0;i<l;i++)
				{
					double wTxi= bias;
					FeatureNode *x = prob->x[i];
					//info_save("*\n");
					while(x->index != -1 )//  traverse x_ij
					{
						int j = x->index-1;// prob_row starts from 1
						//info_save("%d ",j);
						if(w[j]==0)
						{
							x++;
							continue;///    dead loop exists!!!!!!
						}
						wTxi += w[j]*x->value;
						x++;
					}
					exp_wTx.set(i,exp(wTxi));
				}

				//for(int i=0; i<l; i++)
				//	//exp_wTx[i] = bias;//  race
				//	wTx.set(i, bias);

				//for(int j=0; j<n; j++)
				//{
				//	if(w[j]==0) 
				//		continue;
				//	x = prob_col->x[j];
				//	while(x->index != -1)
				//	{
				//		//exp_wTx[x->index] += w[i]*x->value;
				//		wTx.add(x->index,w[j]*x->value);
				//		x++;
				//	}
				//}

				//for(int i=0; i<l; i++)
				//	//exp_wTx[i] = exp(exp_wTx[i]);
				//	exp_wTx.set(i, exp(wTx[i]));
			}// if(num_linesearch >= max_num_linesearch)
#endif
#endif
			//------------------------------------------end of line search
			//}

		}//   for (  active_size)

		for (int s=0;s<active_size;s++)
		{
			Gmax_new = max(Gmax_new, violation_array[s]);
			Gnorm1 += violation_array[s];
		}
		for (int s=0;s<active_size;s++)
		{
			if (feature_status[index[s]]==0)
			{
				active_size--;
				//swap(feature_status[s],feature_status[active_size]);
				Swap(index[s],index[active_size]);
				s--;
			}
		}

		/////////////////////////////////////////////////////////////////////////
		//XXX bias term
		do
		{
			double G = 0;
			double H = 0;

			for(int i=0; i<l; i++)
			{
				double exp_wTxind = exp_wTx[i];
				double tmp1 = 1/(1+exp_wTxind);
				double tmp2 = C[y[i]]*tmp1;
				double tmp3 = tmp2*exp_wTxind;
				G += tmp2;
				H += tmp1*tmp3;
			}

			G = -G + neg_sum;

			Gmax_new = max(Gmax_new, fabs(G));
			Gnorm1 += fabs(G);

			// obtain Newton direction d
			double d=0;
			d = -G/H;

			if(fabs(d) < d_lower_bound)
				break;

			d = min(max(d,-10.0),10.0);

			double delta = G*d;
			int num_linesearch = 0;
			double cond=0;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				cond = -sigma*delta + d*neg_sum;

				for(int i=0; i<l; i++)
				{
					double exp_d = exp(d);
					exp_wTx_new[i] = exp_wTx[i]*exp_d;
					cond += C[y[i]]*log((1+exp_wTx_new[i])/(exp_d+exp_wTx_new[i]));
				}

				if(cond <= 0)
				{
					for(int i=0; i<l; i++)
						exp_wTx[i] = exp_wTx_new[i];
					break;
				}
				else
				{
					d *= beta;
					delta *= beta;
				}
			}

			bias += d;

			// recompute exp_wTx[] if line search takes too many steps

			if(num_linesearch >= max_num_linesearch)
			{
				info_save("#-b");
				for(int i=0; i<l; i++)
					exp_wTx[i] = bias;
				for(int i=0; i<n; i++)
				{
					if(w[i]==0) 
						continue;
					FeatureNode * x = prob_col->x[i];
					while(x->index != -1)
					{
						exp_wTx[x->index] += w[i]*x->value;
						x++;
					}
				}

				for(int i=0; i<l; i++)
					exp_wTx[i] = exp(exp_wTx[i]);
			}

		} while(0);
		//XXX end of bias term
		if(iter == 0)
		{
			Gmax_init = Gmax_new;
			Gnorm1_init = Gnorm1;
			info_save("eps_end %lf \n\n",eps*Gnorm1_init);
		}
		/////////////////////////////////////////////////////////////////////////
		total_time+= omp_get_wtime()-start;
		//---------------  bian start of in_iteration testing
		obj=0; 
		model_nnz = 0;
		for (int j=0;j<n;j++)
		{
			if (w[j] != 0)
			{
				obj += fabs(w[j]);
				model_nnz ++;
			}

		}
		for (int i = 0;i<l;i++)
		{
			if (y[i] == 1)
				obj+= C[y[i]]*log(1+1/exp_wTx[i]);
			else
				obj+=C[y[i]]* log(1+exp_wTx[i]);

		}
		if(probtest != NULL)
			acc = evaluate_testing_bias(w, n, bias, probtest);
		if(iter%10 ==0)
			info_save("iter %d time %lf accuracy %lf f %lf nnz %d Gnorm1 %lf total_num_line_search %d\n",
			iter,total_time,acc,obj,model_nnz,Gnorm1,total_num_line_search);

		save_exp("iter %d time %lf accuracy %lf f %lf nnz %d Gnorm1 %lf total_num_line_search %d\n",
			iter,total_time,acc,obj,model_nnz,Gnorm1,total_num_line_search);

		//info_save("iter %d active_size %d\n",iter,active_size);

		//-------------------  bian end of in_iteration testing
		start=omp_get_wtime();


		iter++;

		//if(iter % 10 == 0)
		//	info(".");

		//if(Gmax_new <= eps*Gmax_init)
		if(Gnorm1 <= eps*Gnorm1_init)//  结束条件
		{
			if(active_size == n)
				break;//   
			else
			{
				active_size = n;
				//info("*");
				memset(feature_status,1,sizeof(bool)*n);
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}//   while(iter<max_iter)

	total_time += omp_get_wtime()-start;


	//XXX testing
	//double v = 0;
	//nnz = 0;
	//double acc = 0;
	double gc_norm = 0;

	//if(probtest != NULL)
	//	acc = evaluate_testing_bias(w, n, bias, probtest);
	double sum2=0;
	double G=0;
	for(int j=0; j<n; j++)
	{
		sum2 = 0;
		FeatureNode *x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index;
			double tmp2 = C[y[ind]]*x->value/(1+exp_wTx[ind]);
			sum2 += tmp2;
			x++;
		}
		G = -sum2 + xjneg_sum[j];

		if(w[j] > 0)
			gc_norm += (G+1)*(G+1);
		else if(w[j] < 0)
			gc_norm += (G-1)*(G-1);
		else if(fabs(G) > 1)
			gc_norm += (fabs(G)-1)*(fabs(G)-1);
	}

	G = 0;
	for(int i=0; i<l; i++)
	{
		double exp_wTxind = exp_wTx[i];
		double tmp1 = 1/(1+exp_wTxind);
		double tmp2 = C[y[i]]*tmp1;
		G += tmp2;
	}
	G = -G + neg_sum;
	gc_norm += G*G;//   
	gc_norm = sqrt(gc_norm);

	//for(j=0; j<n; j++)
	//	if(w[j] != 0)
	//	{
	//		v += fabs(w[j]);
	//		nnz++;
	//	}
	//for(j=0; j<l; j++)
	//	if(y[j] == 1)
	//		v += C[y[j]]*log(1+1/exp_wTx[j]);
	//	else
	//		v += C[y[j]]*log(1+exp_wTx[j]);
	 *model_bias=bias;
	info_save("#iter %d time %lf f %lf accuracy %lf model_nnz %d l %d n %d Gcnorm %g Gnorm1 %lf eps_end %lf bias %g total_num_line_search %d\n",
		iter, total_time, obj, acc, model_nnz, l, n, gc_norm,Gnorm1, eps*Gnorm1_init, bias,total_num_line_search);


	delete [] index;
	delete [] y;
	//	delete [] exp_wTx;
	//delete [] exp_wTx_new;
	delete [] xj_max;
	delete [] C_sum;
	delete [] xjneg_sum;
	delete [] xjpos_sum;
	delete [] feature_status;
	delete [] violation_array;
}
// A Parallel Coordinate Descent Newton (PCDN) implementation for 
// L1-regularized logistic regression problems with bias term
//
//  min_w \sum |wj| + C \sum log(1+exp(-yi (w^T xi + bias))),
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w and bias

#if 1
static void solve_l1r_lr_b_pcdn_no_adaptive_iiter(
	const Problem *prob_col, const Problem *probtest,const int bundle_size,
	double *w, double eps,
	double Cp, double Cn, double *model_bias)
{//  l1r_lr_b_pcdn_no_adaptive_iiter
	//  new recompute,  after shrink , bundle line search
	//   not using approximation line search
	//  use new recompute for bias line search (not strictly tested), but no matter 
	//   not using bundle_feature_idx, using omp in line search.
	//  use omp time       new line search # 
	// report number of outer iteration and number of inner iteration
	const int l = prob_col->l;
	const int n = prob_col->n;
	const int max_iter = 1000000;
	const int max_num_linesearch = 20;
	const double sigma = 0.01;
	const double beta = 0.5;
	const double min_d =1.0e-12;
	omp_set_num_threads(g_pcdn_thread_num);
	info_save("pcdn for l1 regularized logistic regression with bias.\n#threads %d max_num_linesearch: %d bundle size %d nnz in train data %d\n\n",
			  g_pcdn_thread_num, max_num_linesearch, bundle_size, prob_col->nnz);

	double x_min = 0;
	double Gmax_old = INF;
	double Gmax_new;
	double Gmax_init;
	double Gnorm1;
	double Gnorm1_init;

	int *index = new int[n];
	schar *y = new schar[l];//   0, 1
	double *exp_wTx = new double[l];//  
	//cas_array<double> exp_wTx(l);
	double *exp_wTx_new = new double[l];//  //  
	//cas_array<double> exp_wTx_new(l);//  
#ifdef PCDN_CAS_LR
	cas_array<double> dTx(l);//   atomic 
#else
	double *dTx = new double[l];
#endif

	//const int linesearch_threads_num = 7; 
	//omp_set_num_threads(linesearch_threads_num);
	//info_save("bias bcdn OMP threads = %d\n", linesearch_threads_num);
	//info_save("bias bcdn  max #threads: %d\n",omp_get_max_threads());

	double *xj_max = new double[n];//  no data race
	//double *C_sum = new double[n];//  no data race  not used
	double *xjneg_sum = new double[n];//  no data race
	double *xjpos_sum = new double[n];//  no data race
	//feature_node *x;


	//  bias
	double bias = 0;
	double neg_sum_bias = 0;

	//bian test
	double obj = 0;
	int model_nnz = 0;
	double acc = 0;

	// for  omp parallel---> shrink
	bool *feature_status=new bool[n];
	double *violation_array=new double[n];
	memset(feature_status,1,sizeof(bool)*n);


	// To support weights for instances,
	// replace C[y[i]] with C[i].
	const double C[2] = {Cn,Cp};

	for(int i=0; i<l; i++)
	{
		exp_wTx[i] = 1;
		if(prob_col->y[i] > 0)
			y[i] = 1;
		else
		{
			y[i] = 0;
			neg_sum_bias += C[y[i]];
		}
	}

	for(int j=0; j<n; j++)
	{
		w[j] = 0;
		index[j] = j;
		xj_max[j] = 0;
		//C_sum[j] = 0;
		xjneg_sum[j] = 0;
		xjpos_sum[j] = 0;
		FeatureNode *x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index;
			double val = x->value;
			x_min = min(x_min, val);
			xj_max[j] = max(xj_max[j], val);
			//C_sum[j] += C[y[ind]];
			if(y[ind] == 0)
				xjneg_sum[j] += C[y[ind]]*val;
			else
				xjpos_sum[j] += C[y[ind]]*val;
			x++;
		}
	}


	int active_size = n;

	//  for bundle parallel
	double *sum1_ptr=new double[n];//   not used for not approximate line search
	double *sum2_ptr=new double[n];//   not used for not approximate line search
	double *G_ptr=new double[n];
	double *d_ptr=new double[n];


	//  space for time
#if 0
	bool *bundle_feature_flag =new bool[l];//  false: not used yet
	memset(bundle_feature_flag,0,sizeof(bool)*l);//  
	int bundle_feature_idx_size = max((l+1),l*BUNDLE_SIZE/10);//  empirical value
	int *bundle_feature_idx =new int[bundle_feature_idx_size];// 注意最后的-1结束位
	memset(bundle_feature_idx,0,sizeof(int)*(bundle_feature_idx_size));
#endif

	int feature_upper_id=0;
	double bundle_d_norm =0;
	double delta =0;
	double new_group_w_norm=0;
	double group_w_norm=0;
	double neg_sum =0;
	int num_linesearch=0;
	double cond=0;
	double appxcond1=0;
	double appxcond2=0;
	int g_idx_tmp=0;

	//  omp wall  time
	double total_time = 0;
	double start = omp_get_wtime();

	int total_num_line_search=0;
	int iter = 0;
	int iiter=0;
	//int iiter2=0;
	while(iter < max_iter)
	{		
		Gmax_new = 0;//  infinite-norm
		Gnorm1 = 0;//  1-norm

		for(int s=0; s<active_size; s++)
		{
			int i = s+rand()%(active_size-s);
			Swap(index[i], index[s]);
		}

		int iiter_tmp =  active_size/bundle_size;
		feature_upper_id=(active_size/bundle_size)*bundle_size;
		if (active_size%bundle_size != 0)
		{
			feature_upper_id+=bundle_size;
			iiter_tmp++;
		}

		iiter+=iiter_tmp;

		for (int feature_id=0;feature_id<feature_upper_id;feature_id+=bundle_size)
		{
			int s_upper_tmp = min(feature_id+bundle_size, active_size);
#ifdef PCDN_CAS_LR
			memset(dTx.arr,0,sizeof(double)*l);//   
#else
			memset(dTx,0,sizeof(double)*l);//   每次line search之前都要clear
#endif
			bundle_d_norm =0;
#pragma omp parallel for schedule(static, 10) reduction(+:bundle_d_norm)// bundle size 
			for(int s=feature_id; s<s_upper_tmp; s++)
			{
				//if (s<active_size)
				//{
					int j = index[s];
					sum1_ptr[j] = 0;               // * approximation line search
					sum2_ptr[j] = 0;                // *approximation line search
					double H = 0;

					FeatureNode *x = prob_col->x[j];//  feature j 的所有sample
					while(x->index != -1)
					{
						int ind = x->index;
						double exp_wTxind = exp_wTx[ind];//  exp_wTx!!  read shared variable to private variable
						double tmp1 = x->value/(1+exp_wTxind);
						double tmp2 = C[y[ind]]*tmp1;
						double tmp3 = tmp2*exp_wTxind;
						sum2_ptr[j] += tmp2;
						sum1_ptr[j] += tmp3;
						H += tmp1*tmp3;
						x++;
					}

					//double G = -sum2_ptr[s-feature_id] + xjneg_sum[j];// * line search
					G_ptr[j] = -sum2_ptr[j] + xjneg_sum[j];// * line search

					double Gp = G_ptr[j]+1;
					double Gn = G_ptr[j]-1;
					//double violation = 0;
					violation_array[s]=0;//  non-negative
					d_ptr[j]=0;
					if(w[j] == 0)
					{
						if(Gp < 0)
							violation_array[s] = -Gp;
						else if(Gn > 0)
							violation_array[s] = Gn;
						else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
						{
							/*						active_size--;
							swap(index[s], index[active_size]);
							s--;*/
							feature_status[j]=0;
							continue;
						}
					}
					else if(w[j] > 0)
						violation_array[s] = fabs(Gp);
					else
						violation_array[s] = fabs(Gn);

					//Gmax_new = max(Gmax_new, violation_array[s]);//   atomic 
					//Gnorm1 += violation_array[s];  //   reduction

					// obtain Newton direction d
					//double d=0;                                 // * line search

					if(Gp <= H*w[j])
						d_ptr[j] = -Gp/H;
					else if(Gn >= H*w[j])
						d_ptr[j] = -Gn/H;
					else
						d_ptr[j] = -w[j];

					if(fabs(d_ptr[j]) < min_d)
					{
						d_ptr[j]=0;
						continue;
					}
					d_ptr[j] = min(max(d_ptr[j],-10.0),10.0);
//#pragma omp atomic					
					bundle_d_norm+=fabs(d_ptr[j]);//  race!!

					//feature_node *x;
					//int j = index[s];
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int idx=x->index;
#ifdef PCDN_CAS_LR
						dTx.add(idx,d_ptr[j]*x->value);//dTx(i)+= sigma_j[d_(j)*X_(i,j)]    here has race!!
#else
#pragma omp atomic
						dTx[idx]+=d_ptr[j]*x->value;
#endif
						x++;
					}

				//}//  if(s<active_size)

			}//   for(int s=feature_id; s<feature_id+NUM_THREADS; s++)
			//#pragma omp parallel for implicit barrier
#if 0
			for (int s=feature_id; s<feature_id+NUM_THREADS; s++)
			{
				if (s<active_size)
				{

					int j = index[s];
					if(feature_status[j]==0)
						continue;

					if(fabs(d_ptr[s-feature_id]) < 1.0e-12)
						continue;

					d_ptr[s-feature_id] = min(max(d_ptr[s-feature_id],-10.0),10.0);
					feature_node *x;//  private
					//----------------------------------line search------------------------------------------
					double delta = fabs(w[j]+d_ptr[s-feature_id])-fabs(w[j]) + G_ptr[s-feature_id]*d_ptr[s-feature_id];
					int num_linesearch=0;
					double cond=0;
					double appxcond1=0;
					double appxcond2=0;
					for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
					{
						cond = fabs(w[j]+d_ptr[s-feature_id])-fabs(w[j]) - sigma*delta;

						if(x_min >= 0)//  为什么会有这个？   
						{
							double tmp = exp(d_ptr[s-feature_id]*xj_max[j]);
							appxcond1 = log(1+sum1_ptr[s-feature_id]*(tmp-1)/xj_max[j]/C_sum[j])*C_sum[j] + cond - d_ptr[s-feature_id]*xjpos_sum[j];
							appxcond2 = log(1+sum2_ptr[s-feature_id]*(1/tmp-1)/xj_max[j]/C_sum[j])*C_sum[j] + cond + d_ptr[s-feature_id]*xjneg_sum[j];
							if(min(appxcond1,appxcond2) <= 0)
							{
								x = prob_col->x[j];
								while(x->index != -1)
								{//  feature j 关联的所有sample
									//exp_wTx[x->index] *= exp(d*x->value);
									exp_wTx.mul(x->index,exp(d_ptr[s-feature_id]*x->value));
									x++;
								}
								break;
							}
						}

						cond += d_ptr[s-feature_id]*xjneg_sum[j];

						int i = 0;
						x = prob_col->x[j];
						while(x->index != -1)
						{
							int ind = x->index;
							double exp_dx = exp(d_ptr[s-feature_id]*x->value);
							//exp_wTx_new[i] = exp_wTx[ind]*exp_dx;//  race
							exp_wTx_new.set(i,exp_wTx[ind]*exp_dx);//  race   为什么用i？

							cond += C[y[ind]]*log((1+exp_wTx_new[i])/(exp_dx+exp_wTx_new[i]));
							x++; 
							i++;
						}

						if(cond <= 0)
						{
							int i = 0;
							x = prob_col->x[j];
							while(x->index != -1)
							{
								int ind = x->index;
								//exp_wTx[ind] = exp_wTx_new[i];// race
								exp_wTx.set(ind, exp_wTx_new[i]);
								x++; 
								i++;
							}
							break;
						}
						else
						{
							d_ptr[s-feature_id] *= beta;//  beta =0.5
							delta *= beta;
						}
					}//  for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)

					w[j] += d_ptr[s-feature_id];//   no data race, 如果超出限制，无论如何接受

					// recompute exp_wTx[] if line search takes too many steps
#if 1
					if (num_linesearch >= max_num_linesearch)
					{//  haha  
						info_save("#");// 
						//int i = 0;
						x = prob_col->x[j];
						while(x->index != -1)
						{
							int ind = x->index;
							double exp_dx = exp(d_ptr[s-feature_id]*x->value);
							//exp_wTx_new[i] = exp_wTx[ind]*exp_dx;//  race
							exp_wTx.set(ind,exp_wTx[ind]*exp_dx);//  race  
							x++; 
							//i++;
						}

					}

#else				
					if(num_linesearch >= max_num_linesearch)
					{//   似乎可以再优化
						info_save("#");//   
						for(int i=0; i<l; i++)
							//exp_wTx[i] = bias;//  race
							exp_wTx.set(i, bias);

						for(int j=0; j<n; j++)
						{
							if(w[j]==0) 
								continue;
							x = prob_col->x[j];
							while(x->index != -1)
							{
								//exp_wTx[x->index] += w[i]*x->value;
								exp_wTx.add(x->index,w[j]*x->value);
								x++;
							}
						}

						for(int i=0; i<l; i++)
							//exp_wTx[i] = exp(exp_wTx[i]);
							exp_wTx.set(i, exp(exp_wTx[i]));
					}// if(num_linesearch >= max_num_linesearch)
#endif

					//------------------------------------------end of line search
				}
			}


#else

#if 0

			for (int s=feature_id; s<feature_id+NUM_THREADS; s++)
			{
				if (s<active_size)
				{

					int j = index[s];
					if(0==feature_status[j] || fabs(d_ptr[j]) < 1.0e-12)
						d_ptr[j]=0;//  already  initialized as 0s

					//if(fabs(d_ptr[s-feature_id]) < 1.0e-12)
					//	continue;
					else
						d_ptr[j] = min(max(d_ptr[j],-10.0),10.0);

					group_d_norm+=fabs(d_ptr[j]);

				}
			}
#endif

			if (0 == bundle_d_norm)
				continue;
			//int g_idx=0;//
			//#pragma omp parallel for
#if 0
			for (int s=feature_id; s<feature_id+BUNDLE_SIZE; s++)
			{
				if (s<active_size)
				{
					feature_node *x;
					int j = index[s];
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int idx=x->index;
						dTx.add(idx,d_ptr[j]*x->value);//dTx(i)+= sigma_j[d_(j)*X_(i,j)]
						x++;
					}
				}

			}
#endif
			//bundle_feature_idx[g_idx]=-1;//  g_idx is its size
			//info_save("size of group_sample_idx after push %d \n",group_sample_idx.size());

			//----------------------------------line search------------------------------------------
			delta = 0;
			new_group_w_norm = 0;
			group_w_norm = 0;
			neg_sum = 0;

			for (int s=feature_id; s<s_upper_tmp; s++)
			{
				int j = index[s];
				delta +=  G_ptr[j]*d_ptr[j];
				new_group_w_norm += fabs(w[j]+d_ptr[j]);
				group_w_norm+=fabs(w[j]);
				neg_sum += d_ptr[j]*xjneg_sum[j];//   
			}

			delta += (new_group_w_norm-group_w_norm);

			num_linesearch=0;
			cond=0;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{

				cond = new_group_w_norm - group_w_norm - sigma*delta;//   !!!!!!!!!!!
				cond += neg_sum;

#if 0
				if(x_min >= 0)  
				{
					double tmp = exp(d_ptr[s-feature_id]*xj_max[j]);
					appxcond1 = log(1+sum1_ptr[s-feature_id]*(tmp-1)/xj_max[j]/C_sum[j])*C_sum[j] + cond - d_ptr[s-feature_id]*xjpos_sum[j];
					appxcond2 = log(1+sum2_ptr[s-feature_id]*(1/tmp-1)/xj_max[j]/C_sum[j])*C_sum[j] + cond + d_ptr[s-feature_id]*xjneg_sum[j];
					if(min(appxcond1,appxcond2) <= 0)
					{
						x = prob_col->x[j];
						while(x->index != -1)
						{//  feature j 关联的所有sample
							//exp_wTx[x->index] *= exp(d*x->value);
							exp_wTx.mul(x->index,exp(d_ptr[s-feature_id]*x->value));
							x++;
						}
						break;
					}
				}
#endif

#if 1
#pragma omp parallel for  reduction(+:cond) /*num_threads(linesearch_threads_num)*/
				for (int i=0;i<l;i++)//  执行 (n/bundle_size)*num_linesearch times
				{
					if(0 == dTx[i])
						continue;
					double exp_dTx = exp(dTx[i]);
					exp_wTx_new[i]=exp_wTx[i]*exp_dTx;//
					cond += C[y[i]]*log((1+exp_wTx_new[i])/(exp_dTx+exp_wTx_new[i]));
				}
#else
				memset(bundle_feature_flag,0,sizeof(bool)*l);
				//for (int idx=0;idx<group_sample_idx.size();idx++)// 
				//for (ite_idx=group_sample_idx.begin();ite_idx!=group_sample_idx.end();ite_idx++)//
				g_idx_tmp=0;
				while(-1!=bundle_feature_idx[g_idx_tmp])
				{
					int i=bundle_feature_idx[g_idx_tmp++];
					if (!bundle_feature_flag[i])
					{
						double exp_dTx = exp(dTx[i]);
						//info_save("idx %d %g ",i,exp_dTx);
						exp_wTx_new.set(i,exp_wTx[i]*exp_dTx);//  race  
						cond += C[y[i]]*log((1+exp_wTx_new[i])/(exp_dTx+exp_wTx_new[i]));
						bundle_feature_flag[i]=true;
					}
				}
				//x=prob_col->x[j];
				//while(-1 != (x->index))
				//{

				//	int i=x->index;
				//	double exp_dTx = exp(dTx[i]);
				//	//if(1!=exp_dTx)
				//	//info_save("idx %d %g ",i,exp_dTx);
				//	exp_wTx_new.set(i,exp_wTx[i]*exp_dTx);//  race  
				//	cond += C[y[i]]*log((1+exp_wTx_new[i])/(exp_dTx+exp_wTx_new[i]));
				//	x++;
				//}
#endif
				//info_save("cond2: %d\n",cond);
				if(cond <= 0)
				{
#if 1
#pragma omp parallel for /*num_threads(linesearch_threads_num)*/
					for (int i=0;i<l;i++)//  执行 (n/bundle_size)*num_linesearch times
					{
						if(0 == dTx[i])
							continue;
						exp_wTx[i]=exp_wTx_new[i];
					}
#else
					memset(bundle_feature_flag,0,sizeof(bool)*l);
					g_idx_tmp=0;
					while(-1!=bundle_feature_idx[g_idx_tmp])
					{
						int i=bundle_feature_idx[g_idx_tmp++];
						if (!bundle_feature_flag[i])
						{
							exp_wTx.set(i, exp_wTx_new[i]);
							bundle_feature_flag[i]=true;
						}
					}
					//x = prob_col->x[j];
					//while(x->index != -1)
					//{
					//	int ind = x->index;
					//	//exp_wTx[ind] = exp_wTx_new[i];// race
					//	exp_wTx.set(ind, exp_wTx_new[ind]);
					//	x++; 
					//	//i++;
					//}
#endif
					break;
				}
				else
				{
					delta *= beta;
					neg_sum*=beta;
#if 1
#pragma omp parallel for /*num_threads(linesearch_threads_num)*/
					for (int i=0;i<l;i++)//  !!!!¿ÉÄÜÖ´ÐÐ
					{
						if(0 == dTx[i])
							continue;
						//dTx.mul(i,beta);//  no race
						dTx[i]*=beta;//  no race
					}
#else
					memset(bundle_feature_flag,0,sizeof(bool)*l);// 小可能执行,先管
					g_idx_tmp=0;
					while(-1!=bundle_feature_idx[g_idx_tmp])
					{
						int i=bundle_feature_idx[g_idx_tmp++];
						if (!bundle_feature_flag[i])
						{
							dTx.mul(i,beta);
							bundle_feature_flag[i]=true;
						}
					}
					//x = prob_col->x[j];
					//while(x->index != -1)
					//{
					//	int ind = x->index;
					//	//exp_wTx[ind] = exp_wTx_new[i];// race
					//	dTx.mul(ind,beta);
					//	x++; 
					//	//i++;
					//}
#endif
					new_group_w_norm=0;
					for (int s=feature_id; s<s_upper_tmp; s++)
					{
						int j = index[s];
						d_ptr[j] *= beta;//  beta =0.5
						new_group_w_norm+= fabs(w[j]+d_ptr[j]);
					}
				}
			}//  for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)

			for (int s=feature_id; s<s_upper_tmp; s++)
			{
				int j = index[s];
				w[j] += d_ptr[j];//   no data race, 如果超出限制，无论如何接受
			}
			// recompute exp_wTx[] if line search takes too many steps
			total_num_line_search+= num_linesearch+1;
#if 1
			if (num_linesearch >= max_num_linesearch)
			{//   
				info_save("#");// 
#if 1
#pragma omp parallel for /*num_threads(linesearch_threads_num)*/
				for (int i=0;i<l;i++)//  
				{
					if(0 == dTx[i])
						continue;
					//exp_wTx[ind] = exp_wTx_new[i];// race
					exp_wTx[i]=exp_wTx[i]*exp(dTx[i]);
				}
#else
				memset(bundle_feature_flag,0,sizeof(bool)*l);// 小可能执行,先管,若执行次数过少，可以不用管
				g_idx_tmp=0;
				while(-1!=bundle_feature_idx[g_idx_tmp])
				{
					int i=bundle_feature_idx[g_idx_tmp++];
					if (!bundle_feature_flag[i])
					{
						exp_wTx.set(i, exp_wTx[i]*exp(dTx[i]));
						bundle_feature_flag[i]=true;
					}
				}
#endif
			}//if (num_linesearch >= max_num_linesearch)
			//------------------------------------------end of line search


#else				
			if(num_linesearch >= max_num_linesearch)
			{//   似乎可以再优化
				info_save("#");//   
				for(int i=0; i<l; i++)
					//exp_wTx[i] = bias;//  race
					exp_wTx.set(i, bias);

				for(int j=0; j<n; j++)
				{
					if(w[j]==0) 
						continue;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						//exp_wTx[x->index] += w[i]*x->value;
						exp_wTx.add(x->index,w[j]*x->value);
						x++;
					}
				}

				for(int i=0; i<l; i++)
					//exp_wTx[i] = exp(exp_wTx[i]);
					exp_wTx.set(i, exp(exp_wTx[i]));
			}// if(num_linesearch >= max_num_linesearch)
#endif
#endif

			//iiter2++;
		}//for (int feature_id=0;feature_id<feature_upper_id;feature_id+=NUM_THREADS)

		//   shrink after loop   same effect
		for (int s=0;s<active_size;s++)
		{
			Gmax_new = max(Gmax_new, violation_array[s]);
						Gnorm1 += violation_array[s];
		}
//#pragma  omp parallel for reduction(+: Gnorm1)
//		for (int s=0;s<active_size;s++)
//		{
//			Gnorm1 += violation_array[s];
//		}

		for (int s=0;s<active_size;s++)
		{
			if (feature_status[index[s]]==0)
			{
				active_size--;
				Swap(index[s],index[active_size]);
				s--;
			}

		}

		//---------------------------------- bias term----------------
		do
		{
			double G = 0;
			double H = 0;

			for(int i=0; i<l; i++)
			{
				double exp_wTxind = exp_wTx[i];
				double tmp1 = 1/(1+exp_wTxind);
				double tmp2 = C[y[i]]*tmp1;
				double tmp3 = tmp2*exp_wTxind;
				G += tmp2;
				H += tmp1*tmp3;
			}

			G = -G + neg_sum_bias;

			Gmax_new = max(Gmax_new, fabs(G));
			Gnorm1 += fabs(G);

			// obtain Newton direction d
			double d=0;
			d = -G/H;

			if(fabs(d) < min_d)
				break;//   为了 break才用 do while

			d = min(max(d,-10.0),10.0);

			double delta = G*d;
			int num_linesearch = 0;
			double cond=0;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				cond = -sigma*delta + d*neg_sum_bias;

				for(int i=0; i<l; i++)
				{//  must traverse all samples
					double exp_d = exp(d);
					exp_wTx_new[i] = exp_wTx[i]*exp_d;
					cond += C[y[i]]*log((1+exp_wTx_new[i])/(exp_d+exp_wTx_new[i]));
				}

				if(cond <= 0)
				{
					for(int i=0; i<l; i++)
						exp_wTx[i] = exp_wTx_new[i];
					break;
				}
				else
				{
					d *= beta;
					delta *= beta;
				}
			}

			bias += d;

			// recompute exp_wTx[] if line search takes too many steps

#if 0
			if(num_linesearch >= max_num_linesearch)
			{
				info_save("#-b");
				for(int i=0; i<l; i++)
					exp_wTx[i] = bias;
				for(int i=0; i<n; i++)
				{
					if(w[i]==0) 
						continue;
					feature_node * x = prob_col->x[i];
					while(x->index != -1)
					{
						exp_wTx[x->index] += w[i]*x->value;
						x++;
					}
				}

				for(int i=0; i<l; i++)
					exp_wTx[i] = exp(exp_wTx[i]);
			}
#else
			if(num_linesearch >= max_num_linesearch)
			{//   not test yet, but should be logically right!
				//  ms除非 diverge，极少用到，先不来严格测试
				info_save("#-b");
				for(int i=0; i<l; i++)
					exp_wTx[i] = exp_wTx[i]*exp(d);
			}

#endif

		} while(0);
		//----------------- end of bias term----------------------------------------
		if(iter == 0)
		{
			Gmax_init = Gmax_new;
			Gnorm1_init = Gnorm1;
			info_save("eps_end %lf \n\n",eps*Gnorm1_init);

		}
		total_time+= omp_get_wtime()-start;
		//---------------  bian start of in_iteration testing
		obj=0; 
		model_nnz = 0;
		for (int j=0;j<n;j++)
		{
			if (w[j] != 0)
			{
				obj += fabs(w[j]);
				model_nnz ++;
			}

		}
		for (int i = 0;i<l;i++)
		{//   
			if (y[i] == 1)
				obj+= C[y[i]]*log(1+1/exp_wTx[i]);
			else
				obj+=C[y[i]]* log(1+exp_wTx[i]);

		}
		if(probtest != NULL)
			acc = evaluate_testing_bias(w, n, bias, probtest);
		if(iter%10 ==0)
			info_save("iter %d iiter %d time %lf accuracy %lf f %lf model_nnz %d Gnorm1 %lf total_num_line_search %d\n",
			iter,iiter,total_time,acc,obj,model_nnz,Gnorm1,total_num_line_search);

		save_exp("iter %d iiter %d time %lf accuracy %lf f %lf model_nnz %d Gnorm1 %lf total_num_line_search %d\n",
			iter,iiter,total_time,acc,obj,model_nnz,Gnorm1, total_num_line_search);

		//info_save("iter %d active_size %d\n",iter,active_size);

		//-------------------  bian end of in_iteration testing
		start=omp_get_wtime();

		iter++;
		//if(Gmax_new <= eps*Gmax_init)
		if(Gnorm1 <= eps*Gnorm1_init)//  结束条件
		{
			if(active_size == n)
				break;//   
			else
			{
				active_size = n;
				info("* ");
				memset(feature_status,1,sizeof(bool)*n);
				Gmax_old = INF;
				continue;
			}
		}
		Gmax_old = Gmax_new;
	}//   while(iter<max_iter)

	total_time += omp_get_wtime()-start;


	//------------------------------- testing  --------------------------
	double gc_norm = 0;
	//if(probtest != NULL)
	//	acc = evaluate_testing_bias(w, n, bias, probtest);
	double sum2=0;
	double G=0;
	for(int j=0; j<n; j++)
	{
		sum2 = 0;
		FeatureNode *x = prob_col->x[j];
		while(x->index != -1)
		{
			int i = x->index;
			double tmp2 = C[y[i]]*x->value/(1+exp_wTx[i]);
			sum2 += tmp2;
			x++;
		}
		G = -sum2 + xjneg_sum[j];

		if(w[j] > 0)
			gc_norm += (G+1)*(G+1);
		else if(w[j] < 0)
			gc_norm += (G-1)*(G-1);
		else if(fabs(G) > 1)
			gc_norm += (fabs(G)-1)*(fabs(G)-1);
	}

	G = 0;
	for(int i=0; i<l; i++)
	{
		double exp_wTxind = exp_wTx[i];
		double tmp1 = 1/(1+exp_wTxind);
		double tmp2 = C[y[i]]*tmp1;
		G += tmp2;
	}
	G = -G + neg_sum_bias;
	gc_norm += G*G;//  
	gc_norm = sqrt(gc_norm);
	*model_bias = bias;
	info_save("bundle_size %d #iter %d #iiter %d time %lf f %lf accuracy %lf model_nnz %d l %d n %d Gcnorm %g Gnorm1 %lf eps_end %g bias %g total_num_line_search %d\n",
		bundle_size,iter, iiter, total_time, obj, acc, model_nnz, l, n, gc_norm,Gnorm1, eps*Gnorm1_init, bias,total_num_line_search);


	delete [] index;
	delete [] y;
	delete [] exp_wTx;
	delete [] exp_wTx_new;
	delete [] xj_max;
	//delete [] C_sum;
	delete [] xjneg_sum;
	delete [] xjpos_sum;
	delete [] feature_status;
	delete [] violation_array;

	delete [] sum1_ptr;
	delete [] sum2_ptr;
	delete [] G_ptr;
	delete [] d_ptr;
#if 0
	delete [] bundle_feature_flag;
	delete [] bundle_feature_idx;
#endif

#ifndef PCDN_CAS_LR
	delete [] dTx;
#endif
}
#endif





// transpose matrix X from row format to column format
static void transpose(const Problem *prob, FeatureNode **x_space_ret, Problem *prob_col)
{
	int i;
	int l = prob->l;
	int n = prob->n;
	int nnz = 0;
	int *col_ptr = new int[n+1];
	FeatureNode *x_space;
	prob_col->l = l;
	prob_col->n = n;
	prob_col->nnz = prob->nnz;
	prob_col->y = new int[l];
	prob_col->x = new FeatureNode*[n];

	for(i=0; i<l; i++)
		prob_col->y[i] = prob->y[i];

	for(i=0; i<n+1; i++)
		col_ptr[i] = 0;
	for(i=0; i<l; i++)
	{
		FeatureNode *x = prob->x[i];
		while(x->index != -1)
		{
			nnz++;
			col_ptr[x->index]++;
			x++;
		}
	}
	for(i=1; i<n+1; i++)
		col_ptr[i] += col_ptr[i-1] + 1;

	x_space = new FeatureNode[nnz+n];
	for(i=0; i<n; i++)
		prob_col->x[i] = &x_space[col_ptr[i]];

	for(i=0; i<l; i++)
	{
		FeatureNode *x = prob->x[i];
		while(x->index != -1)
		{
			int ind = x->index-1;
			x_space[col_ptr[ind]].index = i;
			x_space[col_ptr[ind]].value = x->value;
			col_ptr[ind]++;
			x++;
		}
	}
	for(i=0; i<n; i++)
		x_space[col_ptr[i]].index = -1;

	*x_space_ret = x_space;

	delete [] col_ptr;
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void group_classes(const Problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

static void train_one(const Problem *prob, const Problem *probtest, const Parameter *param, double *w, double Cp, double Cn, double *model_bias=NULL);

static void train_one(const Problem *prob, const Problem *probtest, const Parameter *param, double *w, double Cp, double Cn, double *model_bias)
{
	double eps=param->eps;
	int pos = 0;
	int neg = 0;
	for(int i=0;i<prob->l;i++)
		if(prob->y[i]==+1)
			pos++;
	neg = prob->l - pos;

	switch(param->solver_type)
	{

	case L1R_L2LOSS_SVC:
		{
			Problem prob_col;
			FeatureNode *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);


			for(int i=0; i<param->eps_n; i++)
			{
				switch (g_param.algorithm_type)
				{
				case kCDN:
					solve_l1r_l2_svc_cdn(&prob_col, probtest, w, eps*min(pos,neg)/prob->l, Cp, Cn);
					break;

				case kSCDN:
					info_save("Sorry, there are no scdn for L1-regularized L2-loss support vector classification!\n\n");
					break;

				case kPCDN:
					//solve_l1r_l2_svc_bcdn_dtx_adaptive(&prob_col, probtest, (int)g_bundle_size, w, eps*min(pos,neg)/prob->l, Cp, Cn);
					solve_l1r_l2_svc_pcdn_dtx_iiter(&prob_col, probtest, (int)g_bundle_size, w, eps*min(pos,neg)/prob->l, Cp, Cn);
					break;


				default:
					break;
				}
				eps /= param->eps_g;
			}

			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			break;
		}

	case L1R_LR_B:
		{
			Problem prob_col;
			FeatureNode *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);//  add memory of train data 
			{
				switch (g_param.algorithm_type)
				{
				case kCDN:
					solve_l1r_lr_b_cdn(&prob_col, probtest, w, eps*min(pos,neg)/prob->l, Cp, Cn, model_bias);
					break;
				case kSCDN: 
					solve_l1r_lr_b_scdn(&prob_col, probtest, prob, w, eps*min(pos,neg)/prob->l, Cp, Cn, model_bias);
					break;
				case kPCDN:
					//ocl_bcdn_solver.Solve();
					//solve_l1r_lr_b_bcdn_adaptive(&prob_col, probtest,(int)g_bundle_size, w, eps*min(pos,neg)/prob->l, Cp, Cn);
					solve_l1r_lr_b_pcdn_no_adaptive_iiter(&prob_col, probtest,(int)g_bundle_size, w, eps*min(pos,neg)/prob->l, Cp, Cn, model_bias);
					break;
				default:
					break;
				}
				eps /= param->eps_g;
			}
			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			break;
		}


	default:
		fprintf(stderr, "Error: unknown solver_type\n");
		break;
	}
}

//
// Interface functions
//
Model* train(const Problem *prob, const Problem *probtest, const Parameter *param)
{
	int i,j;
	int l = prob->l;
	int n = prob->n;
	int w_size = prob->n;
	Model *model_ = Malloc(Model,1);

	if(prob->bias != 0)
		model_->num_feature=n-1;
	else
		model_->num_feature=n;
	model_->param = *param;

	model_->bias = prob->bias;

	int nr_class;
	int *label = NULL;
	int *start = NULL;
	int *count = NULL;
	int *perm = Malloc(int,l);

	// group training data of the same class
	group_classes(prob,&nr_class,&label,&start,&count,perm);

	model_->num_class=nr_class;
	model_->label = Malloc(int,nr_class);
	for(i=0;i<nr_class;i++)
		model_->label[i] = label[i];

	// calculate weighted C
	double *weighted_C = Malloc(double, nr_class);
	for(i=0;i<nr_class;i++)
		weighted_C[i] = param->C;
	for(i=0;i<param->nr_weight;i++)
	{
		for(j=0;j<nr_class;j++)
			if(param->weight_label[i] == label[j])
				break;
		if(j == nr_class)
			fprintf(stderr,"warning: class label %d specified in weight is not found\n", param->weight_label[i]);
		else
			weighted_C[j] *= param->weight[i];
	}
	// constructing the subproblem
	//FeatureNode **x = Malloc(FeatureNode *,l);
	//for(i=0;i<l;i++)
	//	x[i] = prob->x[perm[i]];

	int k;
	Problem sub_prob;
	sub_prob.l = l;
	sub_prob.n = n;
	sub_prob.nnz = prob->nnz;
	sub_prob.x = Malloc(FeatureNode *,sub_prob.l);
	sub_prob.y = Malloc(int,sub_prob.l);

	for(k=0; k<sub_prob.l; k++)
	{
		sub_prob.x[k] = prob->x[k];
		sub_prob.y[k] = prob->y[k];
	}



	if(nr_class == 2)
	{
		model_->w=Malloc(double, w_size);
		/*
		int e0 = start[0]+count[0];
		k=0;
		for(; k<e0; k++)
		sub_prob.y[k] = +1;
		for(; k<sub_prob.l; k++)
		sub_prob.y[k] = -1;
		*/
		train_one(&sub_prob, probtest, param, &model_->w[0], weighted_C[0], weighted_C[1],&model_->bias);
	}
	else
	{
		model_->w=Malloc(double, w_size*nr_class);
		double *w=Malloc(double, w_size);
		for(i=0;i<nr_class;i++)
		{
			int si = start[i];
			int ei = si+count[i];

			k=0;
			for(; k<si; k++)
				sub_prob.y[k] = -1;
			for(; k<ei; k++)
				sub_prob.y[k] = +1;
			for(; k<sub_prob.l; k++)
				sub_prob.y[k] = -1;

			train_one(&sub_prob, NULL, param, w, weighted_C[i], param->C,&model_->bias);

			for(int j=0;j<w_size;j++)
				model_->w[j*nr_class+i] = w[j];
		}
		free(w);
	}



	//free(x);
	free(label);
	free(start);
	free(count);
	free(perm);
//	free(sub_prob.x);
//	free(sub_prob.y);
	free(weighted_C);
	return model_;
}

void destroy_model(struct Model *model_)
{
	if(model_->w != NULL)
		free(model_->w);
	if(model_->label != NULL)
		free(model_->label);
	free(model_);
}

static const char *solver_type_table[]={"L1R_LR_B", "L1R_L2LOSS_SVC", NULL};

int save_model(const char *model_file_name, const struct Model *model_)
{//   save exact number of features   exact w
	//   
	int i;
	int nr_feature=model_->num_feature;
	int n;
	const Parameter& param = model_->param;

	//if(model_->bias >=0)
	//	n=nr_feature+1;
	//else
	//	n=nr_feature;
	n=nr_feature;
	int w_size = n;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	int nr_w;
	if(model_->num_class==2)
		nr_w=1;
	else
		nr_w=model_->num_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->num_class);
	fprintf(fp, "label");
	for(i=0; i<model_->num_class; i++)
		fprintf(fp, " %d", model_->label[i]);
	fprintf(fp, "\n");

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.16g\n", model_->bias);

	fprintf(fp, "w\n");
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fprintf(fp, "%.16g ", model_->w[i*nr_w+j]);
		fprintf(fp, "\n");
	}

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else 
		return 0;
}

struct Model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	Model *model_ = Malloc(Model,1);
	Parameter& param = model_->param;

	model_->label = NULL;

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown solver type.\n");
				free(model_->label);
				free(model_);
				return NULL;
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			fscanf(fp,"%d",&nr_class);
			model_->num_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			fscanf(fp,"%d",&nr_feature);
			model_->num_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			fscanf(fp,"%lf",&bias);
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->num_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				fscanf(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			free(model_);
			return NULL;
		}
	}

	nr_feature=model_->num_feature;
	//if(model_->bias!=0)
	//	n=nr_feature+1;
	//else
		n=nr_feature;
	int w_size = n;
	int nr_w;
	if(nr_class==2)
		nr_w = 1;
	else
		nr_w = nr_class;

	model_->w=Malloc(double, w_size*nr_w);
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fscanf(fp, "%lf ", &model_->w[i*nr_w+j]);
		fscanf(fp, "\n");
	}
	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}

int predict_values(const struct Model *model_, const struct FeatureNode *x, double *dec_values)
{
	int idx;
	int n;
	//if(model_->bias!=0)
	//	n=model_->num_feature+1;
	//else
		n=model_->num_feature;
	double *w=model_->w;
	int nr_class=model_->num_class;
	int i;
	int nr_w;
	if(nr_class==2)
		nr_w = 1;
	else
		nr_w = nr_class;

	const FeatureNode *lx=x;
	for(i=0;i<nr_w;i++)
		dec_values[i] = model_->bias;
	for(; (idx=lx->index)!=-1; lx++)
	{
		// the dimension of testing data may exceed that of training
		if(idx <= n)
			for(i=0;i<nr_w;i++)
				dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
	}

	if(nr_class==2)
		return (dec_values[0]>0)?model_->label[0]:model_->label[1];
	else
	{
		int dec_max_idx = 0;
		for(i=1;i<nr_class;i++)
		{
			if(dec_values[i] > dec_values[dec_max_idx])
				dec_max_idx = i;
		}
		return model_->label[dec_max_idx];
	}
}

int infer(const Model *model_, const FeatureNode *x)
{
	double *dec_values = Malloc(double, model_->num_class);
	int label=predict_values(model_, x, dec_values);
	free(dec_values);
	return label;
}

//int predict_probability(const struct Model *model_, const struct FeatureNode *x, double* prob_estimates)
//{
//	if(model_->param.solver_type==L2R_LR)
//	{
//		int i;
//		int nr_class=model_->nun_class;
//		int nr_w;
//		if(nr_class==2)
//			nr_w = 1;
//		else
//			nr_w = nr_class;
//
//		int label=predict_values(model_, x, prob_estimates);
//		for(i=0;i<nr_w;i++)
//			prob_estimates[i]=1/(1+exp(-prob_estimates[i]));
//
//		if(nr_class==2) // for binary classification
//			prob_estimates[1]=1.-prob_estimates[0];
//		else
//		{
//			double sum=0;
//			for(i=0; i<nr_class; i++)
//				sum+=prob_estimates[i];
//
//			for(i=0; i<nr_class; i++)
//				prob_estimates[i]=prob_estimates[i]/sum;
//		}
//
//		return label;		
//	}
//	else
//		return 0;
//}

void destroy_param(Parameter* param)
{
	if(param->weight_label != NULL)
		free(param->weight_label);
	if(param->weight != NULL)
		free(param->weight);
}

const char *check_parameter(const Problem *prob, const Parameter *param)
{//   if there are no error message, return NULL
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	//XXX
	if(param->eps_g <= 0)
		return "g <= 0";
	if(param->eps_n <= 0)
		return "n <= 0";

	if(param->solver_type != L1R_L2LOSS_SVC
		&& param->solver_type != L1R_LR_B)
		return "unknown solver type";

	if(param->algorithm_type != kCDN
		&& param->algorithm_type != kSCDN
		&& param->algorithm_type != kPCDN)
		return "unknown cdn algorithm type";

	return NULL;
}

void cross_validation(const Problem *prob, const Parameter *param, int nr_fold, int *target)
{
	int i;
	int *fold_start = Malloc(int,nr_fold+1);
	int l = prob->l;
	int *perm = Malloc(int,l);

	for(i=0;i<l;i++) perm[i]=i;
	for(i=0;i<l;i++)
	{
		int j = i+rand()%(l-i);
		Swap(perm[i],perm[j]);
	}
	for(i=0;i<=nr_fold;i++)
		fold_start[i]=i*l/nr_fold;

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		struct Problem subprob;

		subprob.bias = prob->bias;
		subprob.n = prob->n;
		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct FeatureNode*,subprob.l);
		subprob.y = Malloc(int,subprob.l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct Model *submodel = train(&subprob,NULL,param);
		for(j=begin;j<end;j++)
			target[perm[j]] = infer(submodel,prob->x[perm[j]]);
		destroy_model(submodel);
		free(subprob.x);
		free(subprob.y);
	}
	free(fold_start);
	free(perm);
}

int get_nr_feature(const Model *model_)
{
	return model_->num_feature;
}

int get_nr_class(const Model *model_)
{
	return model_->num_class;
}

void get_labels(const Model *model_, int* label)
{
	if (model_->label != NULL)
		for(int i=0;i<model_->num_class;i++)
			label[i] = model_->label[i];
}

