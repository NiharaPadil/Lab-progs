import java.util.Random;
class RandomNumThread extends Thread
{
Random rand = new Random();
public void run()
{
try
{
for(int i=0;i<10;i++)
{
System.out.println("Number generated is " + rand.nextInt(100)+" ");
Thread.sleep(1000);
}
}
catch (InterruptedException e)
{
e.printStackTrace();
}
}
}
class SquareNum extends Thread
{
int num;
SquareNum(int a)
{
num = a;
}
public void run()
{
for(int i=1;i<num;i++)
System.out.println("Square of "+ i + " is:" + i*i);
}
}
class ReverseNum extends Thread
{
int n,reversed;
ReverseNum(int a)
{
n = a;
}
public void run()
{
while(n!=0) {
 
 // get last digit from num
 int digit=n%10;
 reversed=reversed*10+digit;
 // remove the last digit from num
 n/=10;
 }
System.out.println("Reversed Number:"+reversed);
 }
}
class CubeNum extends Thread
{
int n;
CubeNum(int a)
{
n = a;
}
public void run()
{
for(int i=1;i<10;i++)
System.out.println("Cube of "+ i +" is:" + i*i*i);
}
}
public class prog3b
{
public static void main(String[] args)
{
Thread thread1 = new RandomNumThread();
thread1.start();
Thread thread2 = new SquareNum(10);
thread2.start();
Thread thread3 = new ReverseNum(123);
thread3.start();
Thread thread4 = new CubeNum(10);
thread4.start();
}
}