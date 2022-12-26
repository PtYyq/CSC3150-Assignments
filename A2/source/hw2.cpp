#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>

#define ROW 10
#define COLUMN 50 
#define LOGLENGTH 15 // fixed log length 15

pthread_mutex_t mutex;

struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog ;


char map[ROW+10][COLUMN] ; 

int status; // a variable to indicate the status of the game, 0 for executing, 1 for win, 2 for lose and 3 for exit

// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}

// a function to randomly initialize the logs
void logs_initialization(void){
	srand(time(0));
	for(int i = 1; i < ROW ; ++i ){
		int start = rand() % 50;
		for (int j=0;j<LOGLENGTH;++j){
			map[i][(start+j)%49] = '=';
		}
	}
}

// print map function
void* print_map(void*t){
	while (!status)
	{
	for(int i = 0; i <= ROW; ++i)	{
		puts( map[i] );
	}
	usleep(10000);
	printf("\e[1;1H\e[2J"); // clear terminal
	}
	pthread_exit(NULL);
}

// function to capture the keyboard inputs
void* capture(void*t){
	while (!status)
	{
	pthread_mutex_lock(&mutex);
		if (kbhit()){
			char dir = getchar();
			if (dir == 'a' || dir == 'A'){
				// if gets out of logs or cross boundary, then lose
				if (frog.y==0||map[frog.x][frog.y-1]==' '){
					status = 2;
					pthread_mutex_unlock(&mutex);
					break;
				}
				// moves in the lowest row
				else if (frog.x==ROW){
					map[frog.x][frog.y] = '|';
					frog.y = frog.y-1;
					map[frog.x][frog.y] = '0';
				}
				// moves in logs
				else{
					map[frog.x][frog.y] = '=';
					frog.y = frog.y-1;
					map[frog.x][frog.y] = '0';
				}
			}
			if (dir == 's' || dir == 'S')
			{
				if (frog.x != ROW){
					// if gets out of logs, then lose
					if (map[frog.x+1][frog.y] == ' '){
						status = 2;
						pthread_mutex_unlock(&mutex);
						break;
					}
					// moves in logs
					else{
						map[frog.x][frog.y] = '=';
						frog.x = frog.x+1;
						map[frog.x][frog.y] = '0';
					}
				}
			}
			if (dir == 'd' || dir == 'D')
			{
				// if gets out of logs or cross boundary, then lose
				if (frog.y==48||map[frog.x][frog.y+1]==' '){
					status = 2;
					pthread_mutex_unlock(&mutex);
					break;
				}
				// moves in the lowest row
				else if (frog.x==ROW){
					map[frog.x][frog.y] = '|';
					frog.y = frog.y+1;
					map[frog.x][frog.y] = '0';
				}
				// moves in logs
				else{
					map[frog.x][frog.y] = '=';
					frog.y = frog.y+1;
					map[frog.x][frog.y] = '0';
				}
			}
			if (dir == 'w' || dir == 'W')
			{
				// if gets out of logs, then lose
				if (frog.x != 1 && map[frog.x-1][frog.y]==' ')
				{
					status = 2;
					pthread_mutex_unlock(&mutex);
					break;
				}
				else{
					if(frog.x==ROW){
						map[frog.x][frog.y] = '|';
						frog.x = frog.x-1;
						map[frog.x][frog.y] = '0';
					}
					else{
					map[frog.x][frog.y] = '=';
					frog.x = frog.x-1;
					map[frog.x][frog.y] = '0';
					// if reaches the other side, then win
					if (frog.x==0){
						status = 1;
						pthread_mutex_unlock(&mutex);
						break;
					}
					}
				}			
				}
			// if player press 'q', then quit
			if (dir == 'q' || dir == 'Q')
			{
				status = 3;
				pthread_mutex_unlock(&mutex);
				break;
			}
		}
		pthread_mutex_unlock(&mutex);
	}
	pthread_exit(NULL);
}

// function to move the logs
void* logs_move(void* t){
	while (!status)
	{
	usleep(50000);
	/*  Move the logs  */
	pthread_mutex_lock(&mutex);
	if (status){
		pthread_mutex_unlock(&mutex);
		break;
	}
	for(int i = 1; i < ROW; ++i ){
		int start; // start index of the log
		int end;   // end index of the log
		for (int j=0;j<COLUMN;++j){
			if((map[i][j]=='='||map[i][j]=='0')&&map[i][(j+48)%49]==' '){
				start = j;
			}
			if ((map[i][j]=='='||map[i][j]=='0')&&map[i][(j+1)%49]==' '){
				end = j;
			}
		}
		// the logs that go left
		if (i%2==1){
			// if the frog is on this log, then we move the whole log
			if(i==frog.x){
				// if the frog cross boundary, then lose 
				if (frog.y==0){
					status = 2;
					pthread_mutex_unlock(&mutex);
					break;
				}
				int j = start;
				while ((j%49)!=(end+1)%49)
				{
					map[i][(j+48)%49] = map[i][j];
					j = (j+1)%49;
				}
				map[i][end] = ' ';
				frog.y = frog.y - 1;
			}
			// otherwise we only need to move start and end pieces
			else{
			map[i][(end+48)%49] = map[i][end];
			map[i][end] = ' ';
			map[i][(start+48)%49] = map[i][start];
			}
		}
		// the logs that go right
		else{
			// if the frog is on this log, then we move the whole log
			if(i==frog.x){
				// if the frog cross boundary, then lose 
				if(frog.y==48){
					status = 2;
					pthread_mutex_unlock(&mutex);
					break;
				}
				else{
					int j = end;
					while ((j%49)!=(start+48)%49){
						map[i][(j+1)%49] = map[i][j];
						j = (j+48)%49;
					}
					map[i][start] = ' ';
					frog.y = frog.y + 1;
				}
			}
			// otherwise we only need to move start and end pieces
			else{
			map[i][(start+1)%49] = map[i][start];
			map[i][start] = ' ';
			map[i][(end+1)%49] = map[i][end];
			}
		}
	}
	pthread_mutex_unlock(&mutex);
	/*  Check keyboard hits, to change frog's position or quit the game. */
	// The keyboard hits are captured by another thread
	
	/*  Check game's status  */
	// Check game's status during the log move

	/*  Print the map on the screen  */
	// The map is printed by another thread
	}
	pthread_exit(NULL);
	
}

int main( int argc, char *argv[] ){
	int res;
	pthread_t threads[3];
	pthread_mutex_init(&mutex,NULL); //initialize the mutex
	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	int i , j ; 
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  
	}	

	for( j = 0; j < COLUMN - 1; ++j )	
		map[ROW][j] = map[0][j] = '|' ;

	for( j = 0; j < COLUMN - 1; ++j )	
		map[0][j] = map[0][j] = '|' ;

	frog = Node( ROW, (COLUMN-1) / 2 ) ; 
	map[frog.x][frog.y] = '0' ;

	logs_initialization(); // initailize logs
	
	printf("\e[?25l"); // hide the cursor

	/*  Create pthreads for wood move and frog control.  */
	res = pthread_create(&threads[0],NULL,&logs_move,NULL);
	if (res){
		printf("ERROR: return code from pthread_create() is %d.",res);
		exit(1);
	}
	res = pthread_create(&threads[1],NULL,&capture,NULL);
	if (res){
		printf("ERROR: return code from pthread_create() is %d.",res);
		exit(1);
	}
	res = pthread_create(&threads[2],NULL,&print_map,NULL);
	if (res){
		printf("ERROR: return code from pthread_create() is %d.",res);
		exit(1);
	}
	
	// join all the threads
	pthread_join(threads[0],NULL);
	pthread_join(threads[1],NULL);
	pthread_join(threads[2],NULL);

	pthread_mutex_destroy(&mutex); // destroy mutex

	// show the final result
	for(int i = 0; i <= ROW; ++i){
		puts( map[i] );
	}
	usleep(300000);
	printf("\e[1;1H\e[2J");
	
	/*  Display the output for user: win, lose or quit.  */
	if (status == 1){
		puts("You win the game!!");
	}
	else if (status == 2)
	{
		puts("You lose the game!!");
	}
	else if (status == 3)
	{
		puts("You exit the game.");
	}
	
	printf("\e[?25h"); // show the cursor again

	pthread_exit(NULL);

	return 0;

}
