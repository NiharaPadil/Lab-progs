//reader writer program

#include<stdio.h>
#include<stdlib.h>
#include<semaphore.h>
#include<pthread.h>


void *reader(void*);
void *writer(void*);
sem_t x,y,z,wsem,rsem;
int readcount=0;
int writecount=0;
 void main(){
     int a=1,b=1;
     system("clear");
     sem_init(&wsem,0,1);
     sem_init(&x,0,1);
     sem_init(&rsem,0,1);
     sem_init(&y,0,1);
     sem_init(&z,0,1);
     
     pthread_t r,w,r1,w1,r2,r3,w2,w3;
     pthread_create(&r,NULL,reader,(void*)a);
     a++;
    pthread_create(&r1,NULL,reader,(void*)a);
    a++;
    pthread_create(&w,NULL,writer,(void*)b);
    b++;
     pthread_create(&w1,NULL,writer,(void*)b);
    b++;
    pthread_create(&r2,NULL,reader,(void*)a);
    a++;
    pthread_create(&w2,NULL,writer,(void*)b);
    b++;
    pthread_create(&r3,NULL,reader,(void*)a);
    a++;
    pthread_create(&w3,NULL,writer,(void*)b);
    pthread_join(r,NULL);
    pthread_join(r1,NULL);
    pthread_join(w,NULL);
    pthread_join(w1,NULL);
    pthread_join(r2,NULL);
    pthread_join(w2,NULL);
    pthread_join(r3,NULL);
    pthread_join(w3,NULL);
    printf("main terminated\n");
 }
 void *reader(void *arg){
     int c=(int)arg;\
     printf("\nreader %d is created",c);
     sleep(1);
     sem_wait(&rsem);
         sem_wait(&x);
         readcount++;
         if (readcount==1)
         sem_wait(&wsem);
    sem_post(&x);
        sem_post(&rsem);
        sleep(1);
        printf("\n\nreader %d is reading\n",c);
        sleep(1);
            printf("\n\nreader %d finished reading\n",c);
            sem_wait(&x);
            readcount --;
            if (readcount==0)
            sem_post(&wsem);
        sem_post(&x);
 }
 void *writer(void *arg){
     int c=(int)arg;\
     printf("\nwriter %d iis created",c);
     sleep(1);
     sem_wait(&y);
        writecount++;
         if (writecount==0)
         sem_wait(&rsem);
    sem_post(&y);
        sem_post(&wsem);
        printf("\n\nwriter %d is writing\n",c);
        sleep(1);
            printf("\n\nwriter %d finished writing \n",c);
            sem_post(&wsem);
            sem_post(&y);
            writecount --;
            if (writecount==0)
            sem_post(&rsem);
        sem_post(&y);
 }
 
