// Progress Bar

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

const int BAR_LENGTH = 25;
const int MAX_TASKS = 5;

typedef struct{
  int id;
  int progress;
  int step;
}Task;


void print_bar(Task);
void clear_screen();



int main(){
  Task task[MAX_TASKS];
  srand(time(NULL));
  
  for (int i = 0; i < MAX_TASKS; i++)
  {
    task[i].id = i + 1;
    task[i].progress = 0;
    task[i].step = rand() % 5 + 1;
  }
  
  int task_incomplete = 1;

  while (task_incomplete)
  {
    task_incomplete = 0;
    clear_screen();
    for (int i = 0; i < MAX_TASKS; i++)
    {
      task[i].progress += task[i].step;
      if (task[i].progress > 100)
      {
        task[i].progress = 100;
      }
      else if(task[i].progress < 100)
      {
        task_incomplete = 1;
      }
      print_bar(task[i]);
    }
    sleep(1);
  }

  printf("\nAll tasks complete");

  return 0;
}



void clear_screen()
{
  #ifdef _WIN32
    system("cls");
  #else
    system("clear");
  #endif
}



void print_bar(Task task){
  float bars_to_show = (float) task.progress / 100 * BAR_LENGTH;

  printf("\nTask %d: [",task.id);

  for (int i = 0; i < BAR_LENGTH; i++)
  {
    if(i < (int) bars_to_show)
    {
      printf("=");
    }
    else
    {
      printf(" ");
    }
  }
  printf("] %d%%",task.progress);
}