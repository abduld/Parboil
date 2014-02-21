// (C) Copyright 2013, University of Illinois, All Rights Reserved

#ifndef __ARG_H__
#define __ARG_H__

typedef struct _options_
{
  char *data_name;
  char *random_name;
  int random_count;
  int npoints;
  char *output_name;
} options;

void usage(char *name);
void parse_args(int argc, char **argv, options* args);

#endif
