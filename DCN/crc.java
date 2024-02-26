import java.util.Scanner;

public class CRC {

    void div(int a[], int k) {
        int gp[] = {1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1};
        
        int count = 0;
        for (int i = 0; i < k; i++) {
            if (a[i] == 1) {
                for (int j = i; j < 17 + i; j++) {
                    a[j] = a[j] ^ gp[count++];
                }
                count = 0;
            }
        }
    }

    public static void main(String[] args) {
        int a[] = new int[100], b[] = new int[100], len, k, flag = 0;
        CRC ob = new CRC();
        
        Scanner sc = new Scanner(System.in);
        System.out.println("Transmitter side\n------------------");
        
        System.out.print("Enter the length of the data: ");
        len = sc.nextInt();
        k = len;
        
        System.out.println("Enter the data (" + len + " bits): ");
        for (int i = 0; i < len; i++) {
            a[i] = sc.nextInt();
        }
        
        for (int i = 0; i < 16; i++) {
            a[len++] = 0;
        }
        
        for (int i = 0; i < len; i++) {
            b[i] = a[i];
        }
        
        ob.div(a, k);
        
        System.out.print("\nThe CRC at the sender: ");
        for (int i = k; i < len; i++) {
            System.out.print(a[i] + " ");
        }
        
        for (int i = 0; i < len; i++) {
            a[i] = a[i] ^ b[i];
        }
        System.out.print("\nData to be transmitted: ");
        for (int i = 0; i < len; i++) {
            System.out.print(a[i] + " ");
        }
        System.out.print("\n\nReceiver side\n------------------");
        
        System.out.println("\nEnter the Received Data: ");
        for (int i = 0; i < len; i++) {
            a[i] = sc.nextInt();
        }
        
        ob.div(a, k);
        System.out.print("\nThe CRC at the receiver: ");
        
        for (int i = k; i < len; i++) {
            System.out.print(a[i] + " ");
        }
        
        for (int i = k; i < len; i++) {
            if (a[i] != 0) {
                flag = 1;
                break;
            }
        }
        System.out.print("\nResult of CRC error detection: ");
        
        if (flag == 1) {
            System.out.println("Error in the data - Resend the data again");
        } else {
            System.out.println("Data is received successfully");
        }
    }
}
