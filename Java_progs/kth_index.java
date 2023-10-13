/*There are n people standing in a circle ready to be executed. The counting out starts at
some point in the circle and proceeds around the circle in a fixed direction. In every step, a
certain number of people are skipped and the next individual is executed. The removal
proceeds around the circle until last person remains, who is given freedom. Given the total
number of person N and a number k which indicates that k-1 persons are skipped and the kth
person is removed in a circle. The task is to choose the place in the initial circle so that you are
the last one remaining and so survive.*/



import java.util.*;
class Assign2a{
static void Compute(List<Integer> person, int k, int index)
{
// Base case , when only one person is left
if (person.size() == 1) {
System.out.println(person.get(0));
return;
}
// find the index of first person which will die
index = ((index + k) % person.size());
// remove the first person which is going to be killed
person.remove(index);
// recursive call for n-1 persons
Compute(person, k, index);
}
public static void main(String [] args)
{
int n = 14; // specific n and k values for original
// josephus problem
int k = 2;
k--; // (k-1)th person will be killed
int index= 0; // The index where the person which will die
List<Integer> person = new ArrayList<>();
// fill the person vector
for (int i = 1; i <= n; i++) {
int c=person.add(i);
System.out.println("c="+c);
}
System.out.println("ArrayList : " + person.toString());
Compute(person, k, index);
}
}