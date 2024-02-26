import java.io.*;
import java.util.*;

public class Leaky {
    public static int min(int x, int y) {
        if (x < y) return x;
        else return y;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int drop = 0, mini, n, cap, count = 0, i, process;
        int inp[] = new int[25];

        System.out.println("Enter the bucket size:");
        cap = sc.nextInt();

        System.out.println("Enter the output rate:");
        process = sc.nextInt();

        System.out.println("Enter the number of packets:");
        n = sc.nextInt();

        System.out.println("Enter the size of packets to be sent:");
        for (i = 0; i < n; i++) {
            inp[i] = sc.nextInt();
        }

        System.out.println("\nSecond/Packet received/Packet sent/Packet left/Packet dropped/");
        System.out.println("____________________________________________________________________");

        for (i = 0; i < n; i++) {
            count += inp[i];
            if (count > cap) {
                drop = count - cap;
                count = cap;
            }
            System.out.print(i + 1);
            System.out.print("\t\t" + inp[i]);
            mini = min(count, process);
            System.out.print("\t\t\t" + mini);
            // Adjust the alignment for Packet left column
            System.out.print("\t\t\t" + (count - mini));
            System.out.println("\t\t\t" + drop);
            drop = 0;
        }
        System.out.print(i + 1);
        System.out.print("\t\t0");
        mini = min(count, process);
        System.out.print("\t\t\t" + mini);
        // Adjust the alignment for Packet left column
        System.out.print("\t\t\t" + (count - mini));
        System.out.println("\t\t\t" + drop);
        drop = 0;
    }
}
