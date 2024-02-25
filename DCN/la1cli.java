
import java.io.*;
import java.net.*;

public class la1cli {
public static void main(String[] args) {
try {
Socket socket = new Socket("localhost", 9999);
BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
PrintWriter out = new PrintWriter(socket.getOutputStream(), true);

String userInput;
while ((userInput = reader.readLine()) != null) {
out.println(userInput);
String serverResponse = in.readLine();
System.out.println("Server response: " + serverResponse);
}

socket.close();//After communication terminates the connection
} catch (IOException e) {
e.printStackTrace();
}

}
}