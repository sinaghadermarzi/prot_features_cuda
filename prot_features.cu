#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include <limits>
#include <float.h>
#include <iostream>
#include <vector>
#include <iterator>
#include <string>
#include <stdio.h>
#include <string.h>
#include <fstream>
using namespace std;




//__device__ double  dist(double* data ,int i,int j , int num_dim){
//	double dis = 0;
//	for (int attr_idx  = 0; attr_idx<num_dim; attr_idx++){
//		dis += (*(data+i*num_dim+attr_idx)-*(data+j*num_dim+attr_idx))*(*(data+i*num_dim+attr_idx)-*(data+j*num_dim+attr_idx));
//	}
//	return sqrt(dis);
//}







//double dist(ArffData* dataset,int i,int j)
//{
//	return (i-j)*(i-j);
//}


//__global__ void KNN_prediction(double* arrayData,int* predictions,int num_instances,int num_classes,int num_attributes)
//{
	//int i = blockDim.x * blockIdx.x + threadIdx.x;
	//double distances[8192];
//}

//int* KNN(ArffData* dataset)
//{
	//int num_instances = dataset->num_instances();
	//int num_classes = dataset->num_classes();
	//int num_attributes  = dataset->num_attributes();

    //int* h_predictions = (int*)malloc(num_instances * sizeof(int));
    //double * h_arrayData = (double*) malloc(num_instances*num_attributes * sizeof(double));

    //for (int i = 0; i <num_instances; i++)
    //{
////    	printf("__________%d\n",i);
    	//for (int j = 0; j <num_attributes; j++)
    	//{
 ////(double) (dataset->get_instance(i)->get(j)->operator float());
////        	printf("%d\n",j);
        	//*(h_arrayData+i*num_attributes+j) = (double) (dataset->get_instance(i)->get(j)->operator float());
    	//}
    //}
    //printf("data successfully put into array form\n");
    //int* d_predictions;
    //double * d_arrayData;
    //cudaMalloc(&d_predictions,num_instances * sizeof(int));
    //cudaMalloc(&d_arrayData,num_instances*num_attributes * sizeof(double));
    //printf("memory allocated on device\n");
    //cudaMemcpy(d_arrayData, h_arrayData, num_instances*num_attributes * sizeof(double), cudaMemcpyHostToDevice);
    //printf("data copied to device\n");
    //// Launch the Vector Add CUDA Kernel
    //int threadsPerBlock = 256;
    //int blocksPerGrid = (num_instances + threadsPerBlock - 1) / threadsPerBlock;
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    //KNN_prediction<<<blocksPerGrid, threadsPerBlock>>>(d_arrayData,d_predictions,num_instances,num_classes,num_attributes);
    //printf("kernel launched\n");

    //cudaMemcpy(h_predictions,d_predictions, num_instances * sizeof(int), cudaMemcpyDeviceToHost);
    //printf("result copied back to memory\n");
    //cudaError_t cudaError = cudaGetLastError();

    //if(cudaError != cudaSuccess)
    //{
        //fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        //exit(EXIT_FAILURE);
    //}

    //cudaFree(d_predictions);
    //cudaFree(d_arrayData);

    //// Free host memory
    //free(h_arrayData);
////    for (int i = 0 ; i<num_instances;i++)
////    	printf("%d,",h_predictions[i]);

    //return h_predictions;

//}

vector<string> split(const string& str, const string& delim)
{
	vector<string> tokens;
	size_t prev = 0, pos = 0;
	do
	{
		pos = str.find(delim, prev);
		if (pos == string::npos) pos = str.length();
		string token = str.substr(prev, pos - prev);
		if (!token.empty()) tokens.push_back(token);
		prev = pos + delim.length();
	} while (pos < str.length() && prev < str.length());
	return tokens;
}



#define MAX_ELEMENTS 10000
#define MAX_LENGTH 10000
#define FEATURE_LENGTH 26
#define TARGET_CHAR 'A'
void compute_feature(const char* seq, double * features,int i)
{
//	int count = 0;
	int all_count= 0;
	char* current_string = (char*)seq+i*MAX_LENGTH;
//	int comp;
	char* pointer = current_string;
	double* current_feaure_array = features + FEATURE_LENGTH*i;
	char ch;
	while(ch = *pointer)
	{
		char alph = ch-'A';
		current_feaure_array[alph]++ ;
//		comp = (*pointer == TARGET_CHAR);
//		count+= comp;
		all_count++;
		pointer++;
	}
	//if (all_count!=0)
		//features[i] = count/all_count;
	//else
		//features[i] = 0	;
	//features[i] = all_count;
}



__global__ void cuda_features(char* d_seq,double* d_features,int line_count)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i <line_count)
	{


		int count = 0;
		int all_count= 0;
		char* pointer = (char*)d_seq+i*MAX_LENGTH;
		int comp;
		//char* pointer = current_string;
		double* current_feaure_array = d_features + FEATURE_LENGTH*i;
		char ch;
		while(ch = *pointer)
		{
			char alph = ch-'A';
			current_feaure_array[alph]++ ;
	//		comp = (*pointer == TARGET_CHAR);
	//		count+= comp;
			all_count++;
			pointer++;
		}
	}
}








int main(int argc, char *argv[])
{
    //if(argc != 2)
    //{
        //cout << "Usage: ./main datasets/datasetFile.arff" << endl;
        //exit(0);
    //}
//    ArffParser parser(argv[1]);
//    ArffParser parser(fileaddr);
//	string filename = argv[1];
	const char* file_name = "Read.txt";
    std::ifstream file(file_name);
    std::string str;
    char* seq = (char*)malloc(MAX_LENGTH*MAX_ELEMENTS*sizeof(char));
    int line_count = 0;
    std::getline(file, str);
    while ((std::getline(file, str)&& line_count<MAX_ELEMENTS))
    {
    	vector<string> spl = split(str, "\t");
		strcpy(seq+line_count*MAX_LENGTH,spl[1].c_str());
    	line_count++;
    }
	double * features = (double*)malloc(FEATURE_LENGTH*line_count*sizeof(double));
    //for (int i = 0; i<line_count;i++)
    //{
    	//compute_feature(seq,features,i);
    	        //// Process str
    //}
    char * d_seq;
    double * d_features;

    cudaMalloc(&d_seq,MAX_LENGTH*line_count*sizeof(char));
    cudaMalloc(&d_features,FEATURE_LENGTH*line_count*sizeof(double));
    printf("memory allocated on device\n");
    cudaMemcpy(d_seq, seq, MAX_LENGTH*line_count*sizeof(char), cudaMemcpyHostToDevice);
    printf("data copied to device\n");
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (line_count + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    cuda_features<<<blocksPerGrid, threadsPerBlock>>>(d_seq,d_features,line_count);
    printf("kernel launched\n");
    cudaMemcpy(features,d_features,FEATURE_LENGTH*line_count*sizeof(double), cudaMemcpyDeviceToHost);
    printf("results copied back to memory\n");
    cudaError_t cudaError = cudaGetLastError();

    if(cudaError != cudaSuccess)
    {
        fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }
    cudaFree(d_seq);
    cudaFree(d_features);

    // Free host memory

//    for (int i = 0 ; i<num_instances;i++)
//    	printf("%d,",h_predictions[i]);




 //   cout<<line_count;
    for (int i = 0; i<line_count;i++)
    {
		for (int j = 0; j<FEATURE_LENGTH;j++)
			cout<<*(features+i*FEATURE_LENGTH+j)<<",";
		cout<<endl;
	}
	free(seq);
	free(features);
}






