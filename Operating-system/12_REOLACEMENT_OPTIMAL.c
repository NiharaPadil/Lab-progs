// OPTIMAL 
#include&lt;stdio.h&gt; 
#include&lt;conio.h&gt;  
int fr[3], n, m; 
void 
display(); 
void main() 
{ 
int i,j,page[20],fs[10]; 
int 
max,found=0,lg[3],index,k,l,flag1=0,flag2=0,pf=0; 
float pr; 
clrscr(); 
printf(&quot;Enter length of the reference string: &quot;); 
scanf(&quot;%d&quot;,&amp;n); 
printf(&quot;Enter the reference string: &quot;); 
for(i=0;i&lt;n;i++)  
scanf(&quot;%d&quot;,&amp;page[i]); 
printf(&quot;Enter no of frames: &quot;); 
scanf(&quot;%d&quot;,&amp;m);  
for(i=0;i&lt;m;i++) 
fr[i]=-1; pf=m;
for(j=0;j&lt;n;j++) 
{ 
flag1=0; flag2=0; 
for(i=0;i&lt;m;i++) 
{ 
if(fr[i]==page[j]) 
{ 
flag1=1; flag2=1; 
break; 
} 
} 
if(flag1==0) 
{ 
for(i=0;i&lt;m;i++) 
{ 
if(fr[i]==-1) 
{ 

fr[i]=page[j]; flag2=1; 
break; 
} 
} 
} 
if(flag2==0) 
{ 
for(i=0;i&lt;m;i++) 
lg[i]=0; 
for(i=0;i&lt;m;i++) 
{ 
for(k=j+1;k&lt;=n;k++) 
{ 
if(fr[i]==page[k]) 
{ 
lg[i]=k-j; 
break; 
} 
} 
} 
found=0; 
for(i=0;i&lt;m;i++) 
{ 
if(lg[i]==0) 
{ 
index=i;  
found = 1;
break; 
} 
} 
if(found==0) 
{ 
max=lg[0]; index=0; 
for(i=0;i&lt;m;i++) 
{ 
if(max&lt;lg[i]) 
{ 
max=lg[i]; 
index=i; 
} 
} 
} 
fr[index]=page[j]; 
pf++; 
} 
display(); 
} 
printf(&quot;Number of page faults : %d\n&quot;, pf); 
pr=(float)pf/n*100; 
printf(&quot;Page fault rate = %f \n&quot;, pr); getch(); 
} 
void display() 
{ 

int i; for(i=0;i&lt;m;i++) 
printf(&quot;%d\t&quot;,fr[i]); 
printf(&quot;\n&quot;); 
}
