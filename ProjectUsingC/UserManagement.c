// User Management

#include <stdio.h>
#include <conio.h>
#include <string.h>
#include <unistd.h>
#include <windows.h>


#define MAX_USERS 10
#define CREDENTIAL_LENGTH 30


typedef struct {
  char username[CREDENTIAL_LENGTH];
  char password[CREDENTIAL_LENGTH];
} User;


User users[MAX_USERS];
int user_count = 0;


void register_user();
int login_user();
void fix_fgets_input(char *);
void input_credential(char *,char *);


int main(){
  int option;

  while (1)
  {
    printf("\nWelcome to User Management");
    printf("\n1. Register");
    printf("\n2. Login");
    printf("\n3. Exit");
    printf("\nSelect an option : ");
    scanf("%d",&option);
    getchar(); // Consume Extra Enter

    switch (option)
    {
      case 1:
        register_user();
        break;
      case 2:
        int user_index = login_user();
        if(user_index >= 0){
          printf("\nLogin Successful! Welcome,%s!\n",users[user_index].username);
        }
        else{
          printf("\nLogin failed! Incorrect username or password\n");
        }
        break;
      case 3:
        printf("Exiting Program.\n");
        return 0;
        break;
      default:
        printf("Invalid option.Please try again.");
        break;
    }
  }
  return 0;
}



void register_user(){
  if(user_count >= MAX_USERS){
    printf("\nMaximum %d users are supported! No more registrations Allowed!!!\n",MAX_USERS);
    return;
  }

  int new_index = user_count;

  printf("\nRegister a new user\n");
  input_credential(users[new_index].username,users[new_index].password);

  user_count++;
  printf("\nRegistration Successful!\n");
}



int login_user(){
  char username[CREDENTIAL_LENGTH],password[CREDENTIAL_LENGTH];

  input_credential(username,password);

  for (int i = 0; i < user_count; i++)
  {
    if(strcmp(username,users[i].username) == 0 && strcmp(password,users[i].password)==0)
    {
      return i;
    }
  }

  return -1;
}



void input_credential(char *username,char *password){
  printf("\nEnter username : ");
  fgets(username,CREDENTIAL_LENGTH,stdin);
  fix_fgets_input(username);

  printf("Enter password (masking enabled): ");
  fflush(stdout);

  // Change Terminal Property

  HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
  DWORD mode;
  GetConsoleMode(hStdin, &mode);
  DWORD originalMode = mode;
  mode &= ~(ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT);
  SetConsoleMode(hStdin, mode);

  char ch;
  int i = 0;

  while(1){
    ch = _getch();
    if (ch == '\r' || ch == '\n') 
    {
      break;   // Exit loop on Enter key
    }

    if(ch == '\b' || ch == 127){
      if(i>0){
        i--;
        printf("\b \b");
      }
    }
    else
    {
      password[i++] = ch;
      printf("*");
    }
  }

  password[i]='\0';
  SetConsoleMode(hStdin, originalMode);
}



void fix_fgets_input(char *string){
  int index = strcspn(string,"\n");
  string[index] = '\0';
}