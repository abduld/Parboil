#include <iostream>
#include <TrioletData.h>
extern "C"
{
#include "util.h"
}
#include <parboil.h>
#include "hist_cxx.h"

#define UINT8_MAX 255

// countingScatter


using namespace std;
using namespace Triolet;

int main(int argc, char **argv)
{
  struct pb_TimerSet timers;
  struct pb_Parameters *parameters;
  
  Triolet_init();

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

  FILE* f = fopen(parameters->inpFiles[0], "rb");
  int result = 0;

  result += fread(&img_width,    sizeof(unsigned int), 1, f);
  result += fread(&img_height,   sizeof(unsigned int), 1, f);
  result += fread(&histo_width,  sizeof(unsigned int), 1, f);
  result += fread(&histo_height, sizeof(unsigned int), 1, f);

  if (result != 4){
    fputs("Error reading input and output dimensions from file\n", stderr);
    return -1;
  }

  int* img = (int*) malloc (img_width*img_height*sizeof(int));
  int32_t* histo_int = (int32_t*) malloc (histo_width*histo_height* sizeof(int32_t));

  pb_SwitchToSubTimer(&timers, "Input", pb_TimerID_IO);

  result = fread(img, sizeof(unsigned int), img_width*img_height, f);

  fclose(f);

  if (result != img_width*img_height){
    fputs("Error reading input array from file\n", stderr);
    return -1;
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  List<Int> img_accel = CreateIntList(img_width * img_height, img);

  for(int iter = 0; iter < numIterations-1; iter++)
  {
    // Repeat computation and ignore results.  Helps with accurate timing.
    accel_histo(img_width, img_height, img_accel);
  }
  List<Int> hist_accel = accel_histo(img_width, img_height, img_accel);
  FromIntList(histo_int, hist_accel);


  unsigned char* histo = (unsigned char*) malloc(histo_width*histo_height*sizeof(unsigned char));
  for(int i = 0; i < histo_width * histo_height; i++)
  {
    if (histo_int[i] < UINT8_MAX)
      histo[i] = (unsigned char) histo_int[i];
    else
      histo[i] = (unsigned char) UINT8_MAX;
  }

  free(histo_int);

  pb_SwitchToSubTimer(&timers, outputStr, pb_TimerID_IO);

  if (parameters->outFile) {
    dump_histo_img(histo, histo_height, histo_width, parameters->outFile);
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  free(img);
  free(histo);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  cout << endl;
  
  return 0;
}
