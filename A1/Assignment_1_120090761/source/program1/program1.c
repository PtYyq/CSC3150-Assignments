#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

// the mapping fucntion
void map_signal_for_term(int signal_value)
{
	switch (signal_value) {
	case 1:
		printf("child process get SIGHUP signal\n");
		break;
	case 2:
		printf("child process get SIGINT signal\n");
		break;
	case 131:
		printf("child process get SIGQUIT signal\n");
		break;
	case 132:
		printf("child process get SIGILL signal\n");
		break;
	case 133:
		printf("child process get SIGTRAP signal\n");
		break;
	case 134:
		printf("child process get SIGABRT signal\n");
		break;
	case 135:
		printf("child process get SIGBUS signal\n");
		break;
	case 136:
		printf("child process get SIGFPE signal\n");
		break;
	case 9:
		printf("child process get SIGKILL signal\n");
		break;
	case 139:
		printf("child process get SIGSEGV signal\n");
		break;
	case 13:
		printf("child process get SIGPIPE signal\n");
		break;
	case 14:
		printf("child process get SIGALARM signal\n");
		break;
	case 15:
		printf("child process get SIGTERM signal\n");
		break;
	default:
		printf("The return signal is not in the map\n");
		break;
	}
}
int main(int argc, char *argv[])
{
	int status;
	/* fork a child process */
	printf("Process start to fork\n");
	pid_t pid = fork();
	// fork error
	if (pid == -1) {
		perror("fork");
		exit(1);
	}
	// this is the child process
	else {
		if (pid == 0) {
			char *arg[argc];
			for (int i = 0; i < argc - 1; i++) {
				arg[i] = argv[i + 1];
			};
			arg[argc - 1] = NULL;

			printf("I'm the Child Process, my pid = %d\n",
			       getpid()); // print child process pid

			printf("Child process start to execute test program:\n");

			/* execute test program */
			execve(arg[0], arg, NULL);

			// handle error
			perror("execve");
			exit(EXIT_FAILURE);
		}
		// this is the parent process
		else {
			printf("I'm the Parent Process, my pid = %d\n",
			       getpid()); // print parent process pid

			/* wait for child process terminates */
			waitpid(pid, &status, WUNTRACED);

			/* check child process'  termination status */

			// exit case
			if (WIFEXITED(status)) {
				printf("Parent process receives SIGCHLD signal\n");
				printf("Normal termination with EXIT STATUS = %d\n",
				       WEXITSTATUS(status));
				// terminate case
			} else if (WIFSIGNALED(status)) {
				printf("Parent process receives SIGCHLD signal\n");
				map_signal_for_term(status);
				// stop case
			} else if (WIFSTOPPED(status)) {
				printf("Parent process receives SIGCHLD signal\n");
				printf("child process get SIGSTOP signal\n");
				// continue case
			} else {
				printf("Parent process receives SIGCHLD signal\n");
				printf("CHILD PROCESS CONTINUED\n");
			}
			exit(0);
		}
	}
}
