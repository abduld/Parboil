/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

#define UINT8_MAX 255

/******************************************************************************
* Implementation: OpenMP
* Details:
* This implementations is a parallel version using privatized histogram. 
* The input is scanned to find the range in the histogram that is non-zero,
* and only that range is privatized.
******************************************************************************/

void get_histo_size(unsigned int * img, int img_width, int img_height, int *max_val, int *min_val) 
{
  int ii;
  int num_threads = omp_get_max_threads();
  printf("threas num = %d\n", num_threads);
  int max_temp[num_threads];
  int min_temp[num_threads];

  //calculate the temp min and max in parallel
  #pragma omp parallel 
  {
    int tid = omp_get_thread_num();
    int min = img[0];
    int max = img[0];
    #pragma omp for
    for(ii = 1; ii < img_width*img_height; ii++) {
      if(img[ii]<min) {
        min = img[ii];
      }
      if(img[ii]>max) {
        max = img[ii];
      }
    }
    min_temp[tid] = min;
    max_temp[tid] = max;
  }
  
  *max_val=max_temp[0];
  *min_val=min_temp[0];

 //combine the results
  for(ii = 1; ii < num_threads; ii++) {
    if(*min_val>min_temp[ii])
      *min_val = min_temp[ii];
    if(*max_val<max_temp[ii])
      *max_val = max_temp[ii];
  }
}

int main(int argc, char* argv[]) {
  struct pb_TimerSet timers;
  struct pb_Parameters *parameters;

  printf("Base implementation of histogramming.\n");
  printf("Maintained by Nady Obeid <obeid1@ece.uiuc.edu>\n");

  parameters = pb_ReadParameters(&argc, argv);
  if (!parameters)
    return -1;

  if(!parameters->inpFiles[0]){
    fputs("Input file expected\n", stderr);
    return -1;
  }

  int numIterations;
  if (argc >= 2){
    numIterations = atoi(argv[1]);
  } else {
    fputs("Expected at least one command line argument\n", stderr);
    return -1;
  }

  pb_InitializeTimerSet(&timers);
  
  char *inputStr = "Input";
  char *outputStr = "Output";
  
  pb_AddSubTimer(&timers, inputStr, pb_TimerID_IO);
  pb_AddSubTimer(&timers, outputStr, pb_TimerID_IO);
  
  pb_SwitchToSubTimer(&timers, inputStr, pb_TimerID_IO);  

  unsigned int img_width, img_height;
  unsigned int histo_width, histo_height;

  FILE* f = fopen(parameters->inpFiles[0],"rb");
  int result = 0;

  result += fread(&img_width,    sizeof(unsigned int), 1, f);
  result += fread(&img_height,   sizeof(unsigned int), 1, f);
  result += fread(&histo_width,  sizeof(unsigned int), 1, f);
  result += fread(&histo_height, sizeof(unsigned int), 1, f);

  if (result != 4){
    fputs("Error reading input and output dimensions from file\n", stderr);
    return -1;
  }

  unsigned int* img = (unsigned int*) malloc (img_width*img_height*sizeof(unsigned int));
  unsigned char* histo = (unsigned char*) calloc (histo_width*histo_height, sizeof(unsigned char));
  
  pb_SwitchToSubTimer(&timers, "Input", pb_TimerID_IO);

  result = fread(img, sizeof(unsigned int), img_width*img_height, f);

  fclose(f);

  if (result != img_width*img_height){
    fputs("Error reading input array from file\n", stderr);
    return -1;
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);


  //calculate the max and min value of img[]
  int max_val;
  int min_val;
  {
    // Write results into temp1 and temp2, then copy to max_val and min_val
    // This ensures that max_val and min_val are not address-taken
    int temp1;
    int temp2;
    get_histo_size(img, img_width, img_height, &temp1, &temp2);
    max_val = temp1;
    min_val = temp2;
  }
 
  int iter;
  for (iter = 0; iter < numIterations; iter++){
    memset(histo,0,histo_height*histo_width*sizeof(unsigned char));

  int num_threads = omp_get_max_threads();
  int num_histo =max_val-min_val+1;
  unsigned char * private_histo = (unsigned char*) calloc (num_threads*num_histo, sizeof(unsigned char));

  #pragma omp parallel
  {
  int i;
  int tid = omp_get_thread_num();
  int index = tid*num_histo;

  //initialize private_histo
  for(i=0;i<num_histo;i++)
    private_histo[index+i] = 0;
  #pragma omp barrier

  //accumulate the private histo
  #pragma omp for
  for(i = 0; i < img_width*img_height; i++) {
    private_histo[index+img[i]-min_val]++;
  }

  //combine the result into histo
  int t,j;
  #pragma omp for
  for(j=min_val;j<max_val+1;j++)
    for(t = 0; t < num_threads; t++) {
      unsigned char temp = histo[j];
      histo[j] += private_histo[t*num_histo+j-min_val];
      if (histo[j] < temp)
	 histo[j] = UINT8_MAX;
      }
  }
  }

//  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  pb_SwitchToSubTimer(&timers, outputStr, pb_TimerID_IO);

  if (parameters->outFile) {
    dump_histo_img(histo, histo_height, histo_width, parameters->outFile);
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  free(img);
  free(histo);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  printf("\n");
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(parameters);

  return 0;
}
