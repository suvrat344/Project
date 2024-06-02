// Guessing Number

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(){
  int random,guess;
  int no_of_guesses = 0;
  srand(time(NULL));

  printf("Welcome to the world of Guessing Number\n");
  random = rand() % 100 + 1;   // Generating 1 to 100

  do
  {
    printf("Please enter your guess between 1 to 100 : ");
    scanf("%d",&guess);
    no_of_guesses++;
    if(guess < random){
      printf("Guess larger number.\n");   
    }
    else if(guess > random){
      printf("Guess smaller number.\n");
    }
    else{
      printf("Congratulations !!!You have successfully guessed the number in %d attempts.",no_of_guesses);

    }
  } while (guess != random);
  
  printf("\nBye Bye, Thanks for Playing.");
  printf("\nDeveloped by: Suvrat Chauhan");

  return 0;
}