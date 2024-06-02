// Simple Calculator

#include <stdio.h>
#include <math.h>

void print_Menu();
double divide(double,double);
double Modulus(int,int);

int main(){

  int choice;
  double first,second,result;

  printf("Welcome to Simple Calculator\n");
  
  while(1)
  {
    printf("\n--------------------------------\n");
    print_Menu();
    scanf("%d",&choice);

    if(choice == 7){
      break;
    }

    if(choice < 1 || choice > 7){
      fprintf(stderr,"Invalid Menu Choice");
      continue;
    }

    printf("\nPlease enter first number : ");
    scanf("%lf",&first);

    printf("Now enter second number : ");
    scanf("%lf",&second);

    switch (choice)
    {
    case 1:
      result = first + second;
      break;
    case 2:
      result = first - second;
      break;
    case 3:
      result = first * second;
      break;
    case 4:
      result = divide(first,second);
      break;
    case 5:
      result = Modulus(first,second);
      break;
    case 6:
      result = pow(first,second);
      break;
    case 7:
      break;
    }

    if(!isnan(result)){
      printf("\nResult of operation is %.2f\n",result);
    }
  };

  return 0;
}


void print_Menu(){
  printf("\nChoose one of the following options:");
  printf("\n1. Addition");
  printf("\n2. Subtraction");
  printf("\n3. Multiplication");
  printf("\n4. Divide");
  printf("\n5. Modulus");
  printf("\n6. Power");
  printf("\n7. Exit");
  printf("\nNow, enter your choice : ");
}


double divide(double first,double second){
  if(second == 0){
    fprintf(stderr,"\nInvalid Argument for division");
    return NAN;
  }
  else{
    return first /second;
  }
}


double Modulus(int first,int second){
  if(second == 0){
    fprintf(stderr,"\nInvalid Argument for Modulus\n");
    return NAN;
  }
  else{
    return first % second;
  }
}