// This file is written based on The LIBLINEAR Project.
// LIBLINEAR 1.7  http://www.csie.ntu.edu.tw/˜cjlin/liblinear/oldfiles/
/**
    This file is part of the implementation of PCDN, SCDN and CDN as described in the paper:

Parallelized Coordinate Descent Newton Method for Efficient L1-Regularized Minimization.
https://ieeexplore.ieee.org/abstract/document/8661743

    The code may be used free of charge for non-commercial and
    educational purposes, the only requirement is that this text is
    preserved within the derivative work. For any other purpose you
    must contact the authors for permission. This code may not be
    redistributed without permission from the authors.
*/



#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <ctype.h>
#include <errno.h>
#include "l1_minimization.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

void print_null(const char *s) {}

void exit_with_help()
{
	printf(
	"Usage: train [options] training_file test_file [model_file_name]\n\n"

	"options:\n\n"
	"-a algorithm: set algorithm type (default 0)\n"
	"	0 -- CDN\n"
	"	1 -- Shotgun CDN (SCDN)\n"
	"	2 -- Parallel CDN (PCDN)\n\n"

	"-s solver type : set type of solver (default 0)\n"
	"	0 -- L1-regularized logistic regression with bias term\n"
	"	1 -- L1-regularized L2-loss support vector classification\n\n"

	"-c cost : set the parameter C (default 1)\n\n"

	"-e epsilon : set tolerance of termination criterion\n"
	"	|f^S(w)|_1 <= eps*min(pos,neg)/l*|f^S(w0)|_1,\n"
	"	where f^S(w) is the minimum-norm subgradient at w \n\n"

    "-g g -n n : to generate the experimental results of CDN using a decreasing\n"
    "            epsilon values = eps/g^i, for i = 0,1,...,n-1 (default g=1.0 n=1)\n\n"

	"-q : quiet mode (no screen outputs)\n\n"
	"training_file: \n"
	"	training set file\n\n"
	"test_file: \n"
	"	test set file\n\n"
	"model_file_name: \n"
	"	model file name\n"
	"	If you do not set model_file_name, it will be set as the result file name following \".model\"\n\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *test_file_name, char *model_file_name);
void read_problem(const char *filename);
void read_problem_test(const char *filename);
void do_cross_validation();

struct FeatureNode *g_x_space;
struct Parameter g_param;
struct Problem gProb;
struct Model* gModel;
int flag_cross_validation;   // default 0
int nr_fold;

double g_bias;//  training data bias

//XXX for a test problem
struct FeatureNode *x_spacetest;
struct Problem gProbtest;

//

std::string gOutfile_name="";
std::string gOutpath = "log/";
std::string gOutfile_name_verbosity="";
std::string gCdn_algorithm_name[kNum]={"cdn","scdn","pcdn"};
////  0 "cdn", 1 "scdn",    2"pcdn"
char gData_set_name[1024];
int gNum_procs = 0;


int g_pcdn_thread_num = 0;//#threads for pcdn. default (set as 0), it is num_procs -1; otherwise, set as other positive integer
int g_bundle_size = 1250;   // bundle size  for pcdn
int g_scdn_thread_num = 8;   // #threads for scdn


int main(int argc, char **argv)
{
	char input_file_name[1024];
	char test_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;
	gNum_procs = omp_get_num_procs();
	if (0 == g_pcdn_thread_num )
	{
		g_pcdn_thread_num = gNum_procs - 1;
	}

	parse_command_line(argc, argv, input_file_name, test_file_name, model_file_name);// set outfile name

#ifdef _DEBUG
	info_save("debug version.\n");
#else
	info_save("release version.\n");
#endif
	info_save("num of procs %d\n", gNum_procs);
	//omp_set_num_threads(NUM_THREADS);
	//info_save("OMP threads = %d\n", NUM_THREADS);
	//info_save("max #threads: %d\n",omp_get_max_threads());

	read_problem(input_file_name);
	read_problem_test(test_file_name);
	error_msg = check_parameter(&gProb,&g_param);

	if(error_msg)
	{
		fprintf(stderr,"Error: %s\n",error_msg);
		exit(1);
	}

	if(flag_cross_validation)
	{
		do_cross_validation();
	}
	else
	{
		gModel = train(&gProb, &gProbtest, &g_param);
		save_model(model_file_name, gModel);
		destroy_model(gModel);
	}

	destroy_param(&g_param);
	free(gProb.y);
	free(gProb.x);
	free(g_x_space);
	free(gProbtest.y);
	free(gProbtest.x);
	free(x_spacetest);
	free(line);

	return 0;
}

void do_cross_validation()
{
	int i;
	int total_correct = 0;
	int *target = Malloc(int, gProb.l);

	cross_validation(&gProb,&g_param,nr_fold,target);

	for(i=0;i<gProb.l;i++)
		if(target[i] == gProb.y[i])
			++total_correct;
	printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/gProb.l);

	free(target);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *test_file_name, char *model_file_name)
{
	int i;
	// default values
	g_param.algorithm_type = kPCDN;
	g_param.solver_type = L1R_LR_B;
	g_param.C = 1;
	g_param.eps = INF; // see setting below

	g_param.eps_g = 1.0;
	g_param.eps_n = 1;
	g_param.nr_weight = 0;
	g_param.weight_label = NULL;
	g_param.weight = NULL;
	flag_cross_validation = 0;
	g_bias = 0;

	// parse command options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;

		if(++i>=argc)
			exit_with_help();

		switch(argv[i-1][1])
		{
		case 'a':
			g_param.algorithm_type = (Algorithm)atoi(argv[i]);
				switch (g_param.algorithm_type)
				{
				case kCDN:
					gOutfile_name+=gCdn_algorithm_name[kCDN];
					gOutfile_name+="_";
					break;
				case kSCDN:
					gOutfile_name+=gCdn_algorithm_name[kSCDN];
					gOutfile_name+="_";
			#ifdef _OPENMP
					//info_save("with omp %d.\n",_OPENMP);
					gOutfile_name+="_threads_";
					char cbuf[10];
					sprintf(cbuf,"%d_",g_scdn_thread_num);
					gOutfile_name+=cbuf;
			#else
					printf("without omp.\n");
					gOutfile_name+="no_omp_";
			#endif
					break;
				case kPCDN:
					gOutfile_name+=gCdn_algorithm_name[kPCDN];
					gOutfile_name+="_threads_";
					char cbuf2[10];
					sprintf(cbuf2,"%d",g_pcdn_thread_num);
					gOutfile_name+=cbuf2;
					gOutfile_name+="_bundle_";
					char abuf3[20];
					sprintf(abuf3,"%d_",g_bundle_size);
					gOutfile_name+=abuf3;
					break;
				default:
					break;
				}
				 break;
			case 's':
				g_param.solver_type = atoi(argv[i]);
				gOutfile_name+="s_";
				gOutfile_name+=argv[i];
				break;

			case 'c':
				g_param.C = atof(argv[i]);
				gOutfile_name+="_c_";
				gOutfile_name+=argv[i];
				break;

			case 'e':
				g_param.eps = atof(argv[i]);
				//strcpy(gOutfile_name,argv[i]);
				gOutfile_name+="_eps_";
				gOutfile_name+=argv[i];
				gOutfile_name+="_";
				break;

			case 'B':
				g_bias = atof(argv[i]);
				break;

			//XXX
			case 'g':
				g_param.eps_g = atof(argv[i]);
				break;
			case 'n':
				g_param.eps_n = atoi(argv[i]);
				break;

			case 'w':
				++g_param.nr_weight;
				g_param.weight_label = (int *) realloc(g_param.weight_label,sizeof(int)*g_param.nr_weight);
				g_param.weight = (double *) realloc(g_param.weight,sizeof(double)*g_param.nr_weight);
				g_param.weight_label[g_param.nr_weight-1] = atoi(&argv[i-1][2]);
				g_param.weight[g_param.nr_weight-1] = atof(argv[i]);
				break;

			case 'v':
				flag_cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
				break;

			case 'q':
				liblinear_print_string = &print_null;
				i--;
				break;

			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	//----------------- determine filenames
	if(i>=argc-1)
		exit_with_help();

	strcpy(input_file_name, argv[i]);
	strcpy(test_file_name, argv[i+1]);

	//----------get data set name
	int begin_,end_;
	for (int i=strlen(input_file_name);i>=0;--i)
	{
		if ('.'==input_file_name[i])
		{
			end_=i-1;
			continue;
		}
		if ('/'==input_file_name[i] )
		{
			begin_=i+1;
			break;
		}
		if (0==i)
		{
			begin_=i;
			break;
		}

	}

	int j=0;
	for (j=0;j<end_-begin_+1;j++)
	{
		gData_set_name[j]=input_file_name[j+begin_];

	}
	gData_set_name[j]='\0';
	gOutfile_name+=gData_set_name;
	if(i<argc-2)
		strcpy(model_file_name,argv[i+2]);
	//
	else
	{
		sprintf(model_file_name,"%s.model",gOutfile_name.c_str());
		//char *p = strrchr(argv[i],'/');
		//if(p==NULL)
		//	p = argv[i];
		//else
		//	++p;
		//sprintf(model_file_name,"%s.model",p);
	}
	gOutfile_name = gOutpath + gOutfile_name;
	gOutfile_name_verbosity=gOutfile_name+"_verbosity";
	//gOutfile_name+=".txt";




//	printf("model: %s\n", model_file_name);
	if(g_param.eps == INF)
	{
			g_param.eps = 0.01;
	}
	info_save("%s %s\n",__DATE__,__TIME__);
	info_save("algorithm %d solver %d C %g eps %g eps_n %d eps_g %g train %s test %s \n",g_param.algorithm_type,g_param.solver_type,g_param.C,g_param.eps,g_param.eps_n,g_param.eps_g,input_file_name,test_file_name);
}

// read in a problem (in libsvm format)
void read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	long int elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	gProb.l = 0;
	elements = 0;
	max_line_len = 1024;//   line length?
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");//  label
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		elements++; // for bias term
		gProb.l++;
	}
	rewind(fp);

	gProb.bias=g_bias;

	gProb.y = Malloc(int,gProb.l);
	gProb.x = Malloc(struct FeatureNode *,gProb.l);
	g_x_space = Malloc(struct FeatureNode,elements+gProb.l);
	gProb.nnz =elements -gProb.l;//

	max_index = 0;
	j=0;
	for(i=0;i<gProb.l;i++)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		gProb.x[i] = &g_x_space[j];
		label = strtok(line," \t");
		gProb.y[i] = (int) strtol(label,&endptr,10);
		if(endptr == label)
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			g_x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || g_x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = g_x_space[j].index;

			errno = 0;
			g_x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		if(gProb.bias != 0)
			g_x_space[j++].value = gProb.bias;

		g_x_space[j++].index = -1;
	}

	if(gProb.bias != 0)
	{
		gProb.n=max_index+1;
		for(i=1;i<gProb.l;i++)
			(gProb.x[i]-2)->index = gProb.n;
		g_x_space[j-2].index = gProb.n;
	}
	else
		gProb.n=max_index;

	fclose(fp);
}

// read in a test problem (in libsvm format)
void read_problem_test(const char *filename)
{
	int max_index, inst_max_index, i;
	long int elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	gProbtest.l = 0;
	elements = 0;
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		elements++;
		gProbtest.l++;
	}
	rewind(fp);

	gProbtest.bias=g_bias;

	gProbtest.y = Malloc(int,gProbtest.l);
	gProbtest.x = Malloc(struct FeatureNode *,gProbtest.l);
	x_spacetest = Malloc(struct FeatureNode,elements+gProbtest.l);

	max_index = 0;
	j=0;
	for(i=0;i<gProbtest.l;i++)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		gProbtest.x[i] = &x_spacetest[j];
		label = strtok(line," \t");
		gProbtest.y[i] = (int) strtol(label,&endptr,10);
		if(endptr == label)
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_spacetest[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_spacetest[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_spacetest[j].index;

			errno = 0;
			x_spacetest[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		//if(probtest.bias >= 0)
		//	x_spacetest[j++].value = probtest.bias;

		x_spacetest[j++].index = -1;
	}

	/*
	if(probtest.bias >= 0)
	{
		probtest.n=max_index+1;
		for(i=1;i<probtest.l;i++)
			(probtest.x[i]-2)->index = probtest.n;
		x_spacetest[j-2].index = probtest.n;
	}
	else
	*/
		gProbtest.n=max_index;

	fclose(fp);
}
