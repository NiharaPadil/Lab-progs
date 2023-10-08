#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int main() {
pid_t pid;

// Fork a child process using vfork()
pid = vfork();

if (pid == -1) {
// Forking failed
perror("vfork");
return 1;
} else if (pid == 0) {
// Child process

printf("Child process: Hello, I'm the child!\n");
printf("Child process: My PID is %d\n", getpid());
printf("Child process: My parent's PID is %d\n", getppid());

// Terminate the child process
_exit(0);
} else {
// Parent process

printf("Parent process: Hello, I'm the parent!\n");
printf("Parent process: My PID is %d\n", getpid());
printf("Parent process: My child's PID is %d\n", pid);

// Wait for the child process to terminate
int status;
waitpid(pid, &status, 0);

if (WIFEXITED(status)) {
printf("Parent process: Child process terminated normally.\n");
} else {
printf("Parent process: Child process terminated abnormally.\n");
}
}
return 0;
}