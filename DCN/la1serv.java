
import java.io.*;
import java.net.*;
public class la1serv{
public static void main(String[] args) {
try {
ServerSocket serverSocket = new ServerSocket(9999);
System.out.println("Server started. Waiting for clients...");

while (true) {
Socket clientSocket = serverSocket.accept();
System.out.println("Client connected: " + clientSocket);
BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
//BufferedReader in reads input from the client through clientSocket.getInputStream().
PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true);
// PrintWriter out writes output to the client through clientSocket.getOutputStream().
String message;
while ((message = in.readLine()) != null) {
System.out.println("Received from client: " + message);
out.println("Server received: " + message);
}
System.out.println("Client disconnected: " + clientSocket);
clientSocket.close();
}
} catch (IOException e) {

e.printStackTrace();
}
}
}