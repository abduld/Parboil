__kernel void block2D_reg_tiling(float c0,float c1, __global float *A0,
     __global float *Anext, int nx, int ny, int nz)
{
  int k;
     int i = get_global_id(0);
 int j = get_global_id(1);

 float bottom=A0[((i)+nx*((j)+ny*(0)))];
 float current=A0[((i)+nx*((j)+ny*(1)))];
 float top;

  for(k=1;k<nz-1;k++)
  {
 if( i>0 && j>0 &&(i<nx-1) &&(j<ny-1) )
 {
   top =A0[((i)+nx*((j)+ny*(k+1)))];

   Anext[((i)+nx*((j)+ny*(k)))] = c1 *
   ( top +
     bottom +
      A0[((i)+nx*((j + 1)+ny*(k)))] +
            A0[((i)+nx*((j - 1)+ny*(k)))] +
     A0[((i + 1)+nx*((j)+ny*(k)))] +
     A0[((i - 1)+nx*((j)+ny*(k)))] )
   - current * c0;

   bottom=current;
   current=top;
  }
 }

}
