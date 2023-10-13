/*Design and develop a Java program simulating how a lift in an 10th floor office building is used. Assume 
a there are 2lifts operating simultaneously and person calls the lift from any of the floor the lift closest 
to that floor comes to pick him up. If both the lifts are equidistance from the floor which the person 
called, then the lift from the upper floor comes down. If multiple requests are received at same time, 
then the elevator will check the available requests and then process this request depending on some 
priority. The program shall contain a thread which controls the movement of the lift and a number of 
identical threads each simulating a person travelling with the lift.*/


import java.util.LinkedList;
import java.util.Queue;
import java.util.concurrent.Semaphore;
public class LiftSimulation {
 static int currentFloor1 = 1;
 static int currentFloor2 = 10;
 static Queue<Integer> requestQueue = new LinkedList<>();
 static Semaphore semaphore = new Semaphore(1);
 public static void main(String[] args) {
 Thread lift1 = new Thread(new Lift("Lift 1", currentFloor1));
 Thread lift2 = new Thread(new Lift("Lift 2", currentFloor2));
 lift1.start();
 lift2.start();
 for (int i = 1; i <= 10; i++) {
 Thread person = new Thread(new Person(i));
 person.start();
 }
 }
 static class Lift implements Runnable {
 private String name;
 private int currentFloor;
 Lift(String name, int currentFloor) {
 this.name = name;
 this.currentFloor = currentFloor;
 }
 @Override
 public void run() {
 while (true) {
 try {
 semaphore.acquire();
 if (!requestQueue.isEmpty()) {
 int targetFloor = requestQueue.poll();
 int distance = Math.abs(currentFloor - targetFloor);
 System.out.println(name + " is on the way to floor " + targetFloor);
 Thread.sleep(distance * 1000);
 currentFloor = targetFloor;
 System.out.println(name + " has reached floor " + currentFloor);
 }
 } catch (InterruptedException e) {
 e.printStackTrace();
 } finally {
 semaphore.release();
 }
 }
 }
 }
 static class Person implements Runnable {
 private int targetFloor;
 Person(int targetFloor) {
 this.targetFloor = targetFloor;
 }
 @Override
 public void run() {
 try {
 Thread.sleep((int) (Math.random() * 10000));
 semaphore.acquire();
 requestQueue.offer(targetFloor);
 System.out.println("Person on floor " + targetFloor + " requested the lift.");
 } catch (InterruptedException e) {
 e.printStackTrace();
 } finally {
 semaphore.release();
 }
 }
 }
}
