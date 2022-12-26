#include <linux/err.h>
#include <linux/fs.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/kmod.h>
#include <linux/kthread.h>
#include <linux/module.h>
#include <linux/pid.h>
#include <linux/printk.h>
#include <linux/sched.h>
#include <linux/signal.h>
#include <linux/slab.h>

MODULE_LICENSE("GPL");

static struct task_struct *task;
struct wait_opts {
	enum pid_type wo_type; // It is defined in ‘/include/linux/pid.h’.
	int wo_flags; // Wait options. (0, WNOHANG, WEXITED, etc.)
	struct pid *wo_pid; // Kernel's internal notion of a process identifier.
	struct siginfo __user *wo_info; // Singal information.
	int __user
		wo_stat; // Child process’s termination status struct rusage __user
		// *wo_rusage; //Resource usage wait_queue_entry_t
		// child_wait; //Task wait queue
	struct rusage __user *wo_rusage; // Resource usage
	wait_queue_entry_t child_wait; // Task wait queue
	int notask_error;
};

// extern all the function we need
extern struct filename *getname_kernel(const char *filename);

extern pid_t kernel_clone(struct kernel_clone_args *kargs);

extern int do_execve(struct filename *filename,
		     const char __user *const __user *__argv,
		     const char __user *const __user *__envp);

extern long do_wait(struct wait_opts *wo);

int status; // a variable to record the return signal

// A function to map return signals to specified cases and print them
void map_signal(int signal_value)
{
	switch (signal_value) {
	case 0:
		printk("[program2] : child process exit normally\n");
		printk("[program2] : The return signal is 0\n");
		break;
	case 1:
		printk("[program2] : get SIGHUP signal\n");
		printk("[program2] : child process is hung up\n");
		printk("[program2] : The return signal is 1\n");
		break;
	case 2:
		printk("[program2] : get SIGINT signal\n");
		printk("[program2] : terminal interrupt\n");
		printk("[program2] : The return signal is 2\n");
		break;
	case 131:
		printk("[program2] : get SIGQUIT signal\n");
		printk("[program2] : terminal quit\n");
		printk("[program2] : The return signal is 3\n");
		break;
	case 132:
		printk("[program2] : get SIGILL signal\n");
		printk("[program2] : child process has illegal instruction error\n");
		printk("[program2] : The return signal is 4\n");
		break;
	case 133:
		printk("[program2] : get SIGTRAP signal\n");
		printk("[program2] : child process has trap error\n");
		printk("[program2] : The return signal is 5\n");
		break;
	case 134:
		printk("[program2] : get SIGABRT signal\n");
		printk("[program2] : child process has abort error\n");
		printk("[program2] : The return signal is 6\n");
		break;
	case 135:
		printk("[program2] : get SIGBUS signal\n");
		printk("[program2] : child process has bus error\n");
		printk("[program2] : The return signal is 7\n");
		break;
	case 136:
		printk("[program2] : get SIGFPE signal\n");
		printk("[program2] : child process has floating point error\n");
		printk("[program2] : The return signal is 8\n");
		break;
	case 9:
		printk("[program2] : get SIGKILL signal\n");
		printk("[program2] : child process killed\n");
		printk("[program2] : The return signal is 9\n");
		break;
	case 139:
		printk("[program2] : get SIGSEGV signal\n");
		printk("[program2] : child process has segmentation fault error\n");
		printk("[program2] : The return signal is 11\n");
		break;
	case 13:
		printk("[program2] : get SIGPIPE signal\n");
		printk("[program2] : child process has broken pipe error\n");
		printk("[program2] : The return signal is 13\n");
		break;
	case 14:
		printk("[program2] : get SIGALARM signal\n");
		printk("[program2] : child process has alarm error\n");
		printk("[program2] : The return signal is 14\n");
		break;
	case 15:
		printk("[program2] : get SIGTERM signal\n");
		printk("[program2] : child process terminated\n");
		printk("[program2] : The return signal is 15\n");
		break;
	case 4991:
		printk("[program2] : get SIGSTOP signal\n");
		printk("[program2] : child process stopped\n");
		printk("[program2] : The return signal is 19\n");
		break;
	default:
		printk("[program2] : The return signal is not in the map\n");
		break;
	}
}

// my_wait function for parent to wait the child process ends
int my_wait(pid_t pid)
{
	long a; // to receive from do_wait
	struct wait_opts wo; // copy form the kernel, see the define above
	struct pid *wo_pid = NULL;
	enum pid_type type;
	type = PIDTYPE_PID;
	wo_pid = find_get_pid(pid); // to allocate a pid

	wo.wo_type = type;
	wo.wo_pid = wo_pid;
	wo.wo_flags = WEXITED | WUNTRACED;
	wo.wo_info = NULL;
	wo.wo_stat = status;
	wo.wo_rusage = NULL;

	a = do_wait(&wo);

	put_pid(wo_pid);

	return wo.wo_stat;
}
// my_exec function for child process to execute a program

int my_exec(void)
{
	int output; // variable to check if execute sucessfully
	const char path[] =
		"/tmp/test"; // a absolute path to the file to execute
	struct filename *file = getname_kernel(
		path); // get the file parameter for do_execute function

	printk("[program2] : child process");

	output = do_execve(file, NULL,
			   NULL); // use do_execute to execute the file

	// handle output
	if (output == 0) {
		return 0;
	} else {
		do_exit(output);
	}
}

// implement fork function
int my_fork(void *argc)
{
	// set default sigaction for current process
	int i;
	int signal;
	pid_t pid;
	// define the arguments for function kernel_clone
	struct kernel_clone_args args = {
		.flags = SIGCHLD,
		.parent_tid = NULL,
		.child_tid = NULL,
		.stack = (unsigned long)&my_exec,
		.stack_size = 0,
		.tls = 0,
		.exit_signal = SIGCHLD,
	};
	struct k_sigaction *k_action = &current->sighand->action[0];
	for (i = 0; i < _NSIG; i++) {
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}

	/* fork a process using kernel_clone or kernel_thread */
	pid = kernel_clone(&args);
	printk_ratelimit();
	printk("[program2] : The child process has pid = %d\n", pid);
	printk("[program2] : This is the parent process, pid = %d\n",
	       current->pid);

	/* execute a test program in child process */

	/* wait until child process terminates */

	signal = my_wait(pid); // receive signal

	map_signal(signal); // map and print signal

	return 0;
}

static int __init program2_init(void)
{
	printk("[program2] : Module_init Yongqi_Yu 120090761\n");

	/* write your code here */

	/* create a kernel thread to run my_fork */
	printk("[program2] : Module_init create kthread start\n");
	task = kthread_create(&my_fork, NULL, "My Thread");
	// handle errror
	if (!IS_ERR(task)) {
		printk("[program2] : Module_init kthread start\n");
		wake_up_process(task);
	}

	return 0;
}

static void __exit program2_exit(void)
{
	printk("[program2] : Module_exit\n");
}

module_init(program2_init);

module_exit(program2_exit);
