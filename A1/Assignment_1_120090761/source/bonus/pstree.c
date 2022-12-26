#include <ctype.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

// funtions to trim a string with whitespace
char *ltrim(char *s)
{
	while (isspace(*s))
		s++;
	return s;
}
char *rtrim(char *s)
{
	char *back = s + strlen(s);
	while (isspace(*--back))
		;
	*(back + 1) = '\0';
	return s;
}
char *trim(char *s)
{
	return rtrim(ltrim(s));
}
/*  function to check if a string is a number, e.g. "123" is a number and
 * returns 1 while "abc" is not and returns 0*/
int is_num(char *string)
{
	int result = 1;
	for (int i = 0; i < strlen(string); i++) {
		if (!isdigit(string[i])) {
			result = 0;
			break;
		}
	}
	return result;
}
// node structure to build the tree
struct node {
	int childNum;
	char threadNum[10];
	char name[100];
	char pid[100];
	char ppid[100];
	char pgid[100];
	struct node *parent;
	struct node *children[100];
	struct node *thread[100];
	// int hasMultiple;
	// char *mulStrings[20];
};
// function to invert a /proc/PID/status file to a node
struct node make_node(char *filename)
{
	char line[100];
	char name[100];
	char pid[100];
	char ppid[100];
	char pgid[100];
	char threads[100];
	struct node result;

	FILE *f = fopen(filename, "r");
	while (fgets(line, sizeof(line), f) != NULL) {
		char *key = strtok(line, ":");
		char *value = strtok(NULL, ":");
		key = trim(key);
		value = trim(value);
		if (strcmp(key, "Pid") == 0) {
			strcpy(pid, value);
		}
		if (strcmp(key, "PPid") == 0) {
			strcpy(ppid, value);
		}
		if (strcmp(key, "Name") == 0) {
			strcpy(name, value);
		}
		if (strcmp(key, "NSpgid") == 0) {
			strcpy(pgid, value);
		}
		if (strcmp(key, "Threads") == 0) {
			strcpy(threads, value);
		}
	}
	strcpy(result.name, name);
	strcpy(result.pid, pid);
	strcpy(result.ppid, ppid);
	strcpy(result.pgid, pgid);
	strcpy(result.threadNum, threads);
	return result;
}
// similar to make_node
struct node *make_thread(char *filename)
{
	char line[100];
	char name[100];
	char pid[100];
	char pgid[100];
	// char ppid[100];
	struct node *result = (struct node *)malloc(sizeof(struct node));

	FILE *f = fopen(filename, "r");
	while (fgets(line, sizeof(line), f) != NULL) {
		char *key = strtok(line, ":");
		char *value = strtok(NULL, ":");
		key = trim(key);
		value = trim(value);
		if (strcmp(key, "Pid") == 0) {
			strcpy(pid, value);
		}
		if (strcmp(key, "NSpgid") == 0) {
			strcpy(pgid, value);
		}
		if (strcmp(key, "Name") == 0) {
			char s[20] = "{";
			strcat(s, value);
			strcat(s, "}");
			strcpy(name, s);
		}
	}
	strcpy(result->name, name);
	strcpy(result->pid, pid);
	strcpy(result->pgid, pgid);
	result->childNum = 0;
	return result;
}

// convert all process file under /proc into a node
int initialization(struct node *nodes, char *dirname)
{
	int ind = 0;
	char *path;
	DIR *folder = opendir(dirname);
	struct dirent *entry;
	while (entry = readdir(folder)) {
		if (is_num(entry->d_name)) {
			char path[100] = "/proc/";
			strcat(path, entry->d_name);
			strcat(path, "/status");
			// printf("%s\n", path);
			nodes[ind++] = make_node(path);
		}
	}
	return ind;
}
// get threads for all nodes if any
void get_threads(struct node *nodes, int length)
{
	for (int i = 0; i < length; i++) {
		char dirname[50] = "/proc/";
		strcat(dirname, nodes[i].pid);
		strcat(dirname, "/task");
		DIR *folder = opendir(dirname);
		struct dirent *entry;
		while (entry = readdir(folder)) {
			if (strcmp(entry->d_name, ".") != 0 &&
			    strcmp(entry->d_name, "..") != 0 &&
			    strcmp(entry->d_name, nodes[i].pid) != 0) {
				char filename[50] = "/proc/";
				strcat(filename, entry->d_name);
				strcat(filename, "/status");
				// printf(filename);
				struct node *n = make_thread(filename);
				nodes[i].children[nodes[i].childNum++] = n;
			}
		}
	}
	for (int i = 0; i < length; i++) {
		if (strcmp(nodes[i].pgid, "") == 0) {
			strcpy(nodes[i].pgid, nodes[i].parent->pgid);
		}
	}
}

// tried to compress the nodes but did not success
// int check_same(struct node *node1, struct node *node2) {
//   if (node1->name == node2->name) {
//     if (node1->childNum == 0 && node2->childNum == 0) {
//       return 1;
//     } else if (node1->childNum == 1 && node2->childNum == 1) {
//       return check_same(node1->children[0], node2->children[0]);
//     } else {
//       return 0;
//     }
//   } else {
//     return 0;
//   }
// }
// int exists(char **strs, char *str, int len) {
//   int exist = 0;
//   for (int i = 0; i < len; i++) {
//     if (strcmp(str, strs[i]) == 0) {
//       exist = 1;
//     }
//   }
//   return exist;
// }
// void give_multiple(struct node *node) {
//   int hasMult = 0;
//   char *s[20];
//   for (int i = 0; i < node->childNum; i++) {
//     for (int j = 0; j < node->childNum; j++) {
//       if (check_same(node->children[i], node->children[j]) &&
//           !exists(s, node->children[i]->name, hasMult)) {
//         s[hasMult++] = node->children[i]->name;
//       }
//     }
//   }
//   struct node **newchild = malloc(100 * sizeof(struct node *));
//   int ind = 0;
//   for (int i = 0; i < node->childNum; i++) {
//     if (!exists(s, node->children[i]->name, hasMult)) {
//       newchild[ind++] = node->children[i];
//     }
//   }
//   node->children = newchild;
//   node->childNum = ind;
//   node->hasMultiple = hasMult;
//   for (int i = 0; i < hasMult; i++) {
//     strcpy(node->mulStrings[i], s[i]);
//   }
// }

// build links for all the nodes so that we will have a tree
void build_tree(struct node *nodes, int length)
{
	for (int i = 0; i < length; i++) {
		int ind = 0;
		for (int j = 0; j < length; j++) {
			if (strcmp(nodes[i].pid, nodes[j].ppid) == 0) {
				nodes[j].parent = &nodes[i];
				nodes[i].children[ind++] = &nodes[j];
			}
		}
		nodes[i].childNum = ind;
	}
	// for (int i = 0; i < length; i++) {
	//   give_multiple(&nodes[i]);
	// }
}
// function to help me print the tree
void help_print(int level, int *space, int *sper)
{
	for (int i = 0; i < level; i++) {
		if (i != level - 1) {
			for (int j = 0; j < space[i]; j++) {
				printf(" ");
			}
			if (sper[i] == 1) {
				printf("|");
			} else {
				printf(" ");
			}
		} else {
			for (int j = 0; j < space[i]; j++) {
				printf(" ");
			}
		}
	}
}
// recursive function to print the tree
void print_tree(struct node *root, int level, int *space, int *sper)
{
	if (root->childNum == 0) {
		printf("%s\n", root->name);
		return;
	}
	int name_len = strlen(root->name);
	if (level == 0) {
		space[level] = name_len + 1;
	} else {
		space[level] = name_len + 2;
	}
	sper[level] = 1;
	if (root->childNum == 1) {
		printf("%s---", root->name);
		sper[level] = 0;
		print_tree(root->children[0], level + 1, space, sper);
	} else {
		for (int i = 0; i < root->childNum; i++) {
			if (i == 0) {
				printf("%s-+-", root->name);
				print_tree(root->children[i], level + 1, space,
					   sper);
			} else if (i < root->childNum - 1) {
				help_print(level + 1, space, sper);
				printf("|-");
				print_tree(root->children[i], level + 1, space,
					   sper);
			} else {
				help_print(level + 1, space, sper);
				printf("`-");
				sper[level] = 0;
				print_tree(root->children[i], level + 1, space,
					   sper);
			}
		}
	}
}
// print the pstree version
void print_version()
{
	printf("pstree (PSmisc) 22.21\nCopyright (C) 1993-2009 Werner Almesberger "
	       "and Craig Small\n\nPSmisc comes with ABSOLUTELY NO WARRANTY.\nThis "
	       "is free software, and you are welcome to redistribute it under\nthe "
	       "terms of the GNU General Public License.\nFor more information about "
	       "these matters, see the files named COPYING.\n");
}
// print tree and show the pid or pgid of the process
void print_tree_pid_gid(struct node *root, int level, int *space, int *sper,
			int option)
{
	char *id;
	if (option == 0) {
		id = root->pid;
	} else {
		id = root->pgid;
	}
	if (root->childNum == 0) {
		printf("%s(%s)\n", root->name, id);
		return;
	}
	int name_len = strlen(root->name) + strlen(id) + 2;
	if (level == 0) {
		space[level] = name_len + 1;
	} else {
		space[level] = name_len + 2;
	}
	sper[level] = 1;
	if (root->childNum == 1) {
		printf("%s(%s)---", root->name, id);
		sper[level] = 0;
		print_tree_pid_gid(root->children[0], level + 1, space, sper,
				   option);
	} else {
		for (int i = 0; i < root->childNum; i++) {
			if (i == 0) {
				printf("%s(%s)-+-", root->name, id);
				print_tree_pid_gid(root->children[i], level + 1,
						   space, sper, option);
			} else if (i < root->childNum - 1) {
				help_print(level + 1, space, sper);
				printf("|-");
				print_tree_pid_gid(root->children[i], level + 1,
						   space, sper, option);
			} else {
				help_print(level + 1, space, sper);
				printf("`-");
				sper[level] = 0;
				print_tree_pid_gid(root->children[i], level + 1,
						   space, sper, option);
			}
		}
	}
}

int main(int argc, char **argv)
{
	struct node nodes[500]; // structure to store all nodes
	struct node threads[500]; // structure to store all threads (you can
		// considerthreads as a simple node)
	int len = initialization(
		nodes,
		"/proc"); // convert all process file under /proc into a node
	int space[500]; // indicator to help me implement the print tree function
	int sper[500]; // indicator to help me implement the print tree function
	build_tree(nodes, len); // build the tree
	get_threads(nodes, len); // build the threads
	// ouput result
	if (argc == 1) {
		print_tree(&nodes[0], 0, space, sper);
	} else {
		char *arg[argc];
		for (int i = 0; i < argc - 1; i++) {
			arg[i] = argv[i + 1];
		};
		arg[argc - 1] = NULL;
		if (strcmp(arg[0], "-p") == 0) {
			print_tree_pid_gid(&nodes[0], 0, space, sper, 0);
		} else if (strcmp(arg[0], "-g") == 0) {
			print_tree_pid_gid(&nodes[0], 0, space, sper, 1);
		} else if (strcmp(arg[0], "-A") == 0) {
			print_tree(&nodes[0], 0, space, sper);
		} else if (strcmp(arg[0], "-l") == 0) {
			print_tree(&nodes[0], 0, space, sper);
		} else if (strcmp(arg[0], "-c") == 0) {
			print_tree(&nodes[0], 0, space, sper);
		} else if (strcmp(arg[0], "-n") == 0) {
			print_tree(&nodes[0], 0, space, sper);
		} else if (strcmp(arg[0], "-V") == 0) {
			print_version();
		} else {
			printf("Unknown argument, print default tree.\n");
			print_tree(&nodes[0], 0, space, sper);
		}
	}

	return 0;
}
