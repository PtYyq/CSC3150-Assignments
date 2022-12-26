#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "async.h"
#include "utlist.h"

pthread_mutex_t  mutex;
pthread_cond_t cond;

struct node{
    void (*hanlder)(int);
    int args;
    struct node* next;
};
struct queue
{
    struct node* head;
};

// define own queue
struct queue my_queue = {
    .head = NULL
};
// define queue pop and insert functions
struct node* pop(struct queue *queue){
    if (queue->head == NULL){
        return NULL;
    }
    struct node* result = queue->head;
    queue->head = queue->head->next;
    return result;
}

void insert(struct queue *queue,struct node *term){
    if (queue->head == NULL){
        queue->head = term;
        return;
    }
    struct node* temp = queue->head;
    while (temp->next != NULL)
    {
        temp = temp->next;
    }
    temp->next = term;
    return;
}

// work function
void* work(void* t){
    while (1) {
    pthread_mutex_lock(&mutex);
        while (my_queue.head == NULL)
        {
            pthread_cond_wait(&cond,&mutex); // thread wait signal if no task in the task queue
        }
    struct node* node = pop(&my_queue);
    pthread_mutex_unlock(&mutex);
    if (node != NULL){
        void(*hanlder)(int) = node->hanlder;
        int args = node->args;
        hanlder(args);
    }
    // if node is NULL, then the thread is mistakenly woken up and should go back to wait state
    else{
        continue;
    }
}
    pthread_exit(NULL);
    return NULL;
}
void async_init(int num_threads) {
    /** TODO: create num_threads threads and initialize the thread pool **/
    pthread_mutex_init(&mutex,NULL); // initialize mutex
    pthread_cond_init(&cond,NULL);  // initialize conditional variable
    for(int i=0;i<num_threads;i++){
        int res;
        pthread_t* thread = (pthread_t*)malloc(sizeof(pthread_t));
        if(thread == NULL){
            printf("Cannot allocate memory.");
        };
        res = pthread_create(thread,NULL,&work,NULL); // create thread
        if(res){
            printf("pthread_create error %d",res);
        };
    }
    return;
}

void async_run(void (*hanlder)(int), int args) {
    // hanlder(args);
    /** TODO: rewrite it to support thread pool **/
    struct node* task = malloc(sizeof(struct node)); // create the node task
    task->hanlder = hanlder;
    task->args = args;
    task->next = NULL;
    pthread_mutex_lock(&mutex);
    insert(&my_queue,task);     // insert the task to task queue
    pthread_mutex_unlock(&mutex);
    pthread_cond_signal(&cond);
}