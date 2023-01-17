#include <stdio.h> 
 
 void Merge(int arr[], int left, int mid, int right)
{
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;
 
   int Left[n1], Right[n2];
 
    for (i = 0; i <n1; i++)
        Left[i] = arr[left + i];
 
    for (j = 0; j < n2; j++)
        Right[j] = arr[mid + 1 + j];
    i = 0;	 
    j = 0; 
    k = left;  
    while (i < n1 && j < n2)
    {
        if (Left[i] <= Right[j])
        {
            arr[k] = Left[i];
            i++;
        }
        else
        {
            arr[k] = Right[j];
            j++;
        }
        k++;
    }
 
   while (i < n1)
    {
        arr[k] = Left[i];
        i++;
        k++;
    }
 
    while (j < n2)
    {
        arr[k] = Right[j];
        j++;
        k++;
    }
}
 
 void Merge_Sort(int arr[], int left, int right)
{
    if (left < right)
    {
 
        int mid = (left+right) / 2;
 
       Merge_Sort(arr, left, mid);
      Merge_Sort(arr, mid + 1, right);
     Merge(arr, left, mid, right);
    }
}
  
int main()
{
    int n;
 printf("Merge Sort  \n");
      
    printf("Enter the number of elements : ");
    scanf("%d", &n);
 
    int arr[n];
    printf("Enter the elements of array: ");
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &arr[i]);
    }
 
    Merge_Sort(arr, 0, n - 1);
 
    printf("The sorted array is: ");
    for (int i = 0; i < n; i++)
    {
        printf("%d ", arr[i]);
    }
    printf("\n");
    return 0;
}
