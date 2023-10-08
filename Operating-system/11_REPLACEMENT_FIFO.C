//////fifo
#include<stdio.h>
int main()
{
int i,j,n,a[50],frame[10],no,k,avail,fault=0;
            printf("\n ENTER THE NUMBER OF PAGES:\n");
scanf("%d",&n);
            printf("\n ENTER THE PAGE NUMBER :\n");
            for(i=1;i<=n;i++)
            scanf("%d",&a[i]);
            printf("\n ENTER THE NUMBER OF FRAMES :");
            scanf("%d",&no);
for(i=0;i<no;i++)
            frame[i]= -1;
                        j=0;
                        printf("\tinput\t page frames\n");
for(i=1;i<=n;i++)
                        {
                                    printf("%d\t\t",a[i]);
                                    avail=0;
                                    for(k=0;k<no;k++)
if(frame[k]==a[i])
                                                avail=1;
                                    if (avail==0)
                                    {
                                        frame[j]=a[i];
                                               
                                        fault++;
                                        for(k=0;k<no;k++)
                                        printf("%d\t",frame[k]);
                                        j=(j+1)%no;
}
                                    printf("\n");
}
                        printf("Page Fault Is %d",fault);
                        return 0;
}