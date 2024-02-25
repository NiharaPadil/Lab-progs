import java.io.*;
import java.util.*;
public class Leaky {
    public static int min(int x,int y){
        if(x<y) return x;
        else return y;
    }
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int drop=0,mini,n,cap,count=0,i,process;
        int inp[] = new int[25];
        System.out.println("Enter the bucket size\n");
        cap=sc.nextInt();
        System.out.println("Enter the output rate\n");
        process=sc.nextInt();
        System.out.println("Enter the number of packets\n");
        n=sc.nextInt();
        System.out.println("Enter the size of packets to be sent\n");
        for(i=0;i<n;i++){
            inp[i]=sc.nextInt();
        }
        System.out.println("\nsecond/packet received/packet senr/packet left/packet dropped/\n");
        System.out.println("____________________________________________________________________");
        for(i=0;i<n;i++)
        {
            count+=inp[i];
            if(count>cap)
            {
                drop=count-cap;
                count=cap;
            }
            System.out.print(i+1);
            System.out.print("\t"+inp[i]);
            mini=min(count,process); 
            System.out.println("\t\t"+mini);
            count=count-mini;
            System.out.println("\t\t"+count);
            System.out.println("\t\t"+drop);
            drop=0;        
    }
    System.out.print(i+1);
            System.out.print("\t0");;
            mini=min(count,process); 
            System.out.println("\t\t"+mini);
            count=count-mini;
            System.out.println("\t\t"+count);
            System.out.println("\t\t"+drop);
            drop=0;        
}
}
