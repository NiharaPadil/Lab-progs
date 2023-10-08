#include<stdio.h>
int main()
{
    int pn[10];
    int arr[10],bur[10],star[10],finish[10],tat[10],wt[10],i,n;
    int totwt=0,tottat=0;
    printf("Enter the number of processes :\n");
    scanf("%d",&n);
    for(i=0;i<n;i++)
    {
        printf("Enter the process name , arrival time and burst time : ");
        scanf("%d%d%d",&pn[i],&arr[i],&bur[i]);
    }
     for(i=0;i<n;i++)
     {
        if(i==0)
        {
            star[i]=arr[i];
            wt[i]=star[i]-arr[i];
            finish[i]=star[i]+bur[i];
            tat[i]=finish[i]-arr[i];     
       }
       else{
            star[i]=finish[i-1];
            wt[i]=star[i]-arr[i];
            finish[i]=star[i]+bur[i];
            tat[i]=finish[i]-arr[i];
       }
     }
     printf("\nPName \t Arrtime \t Burtime \t start \t TAT \t finish");
     for(i=0;i<n;i++)
      {
        printf("\n%d\t%6d\t\t%6d\t\t%6d\t%6d\t%6d",pn[i],arr[i],bur[i],star[i],tat[i],finish[i]);
        totwt+=wt[i];
        tottat+=tat[i];
      }
      printf("\nAverage waiting time : %f",(float)totwt/n);
      printf("\nAverage turn around time : %f", (float)tottat/n);
}

/*Enter the number of processes :
4
Enter the process name , arrival time and burst time : 1 0 35
Enter the process name , arrival time and burst time : 2 1 15
Enter the process name , arrival time and burst time : 3 2 4
Enter the process name , arrival time and burst time : 4 4 1

PName    Arrtime         Burtime         start   TAT     finish
1            0              35               0      35      35
2            1              15              35      49      50
3            2               4              50      52      54
4            4               1              54      51      55
Average waiting time : 33.000000
Average turn around time : 46.750000*/