#include <stdio.h>
#include <stdlib.h>

int st[10], top = -1, v[10], a[10][10], u[10];
int n, q[10], front = 0, rear = -1;

void dfs(int s) {
    int i;
    v[s] = 1;
    st[++top] = s;
    for (i = 1; i <= n; i++) {
        if (a[s][i] == 1 && v[i] == 0) {
            printf("Officer %d -> Officer %d\n", s, i);
            dfs(i);
        }
    }
}

void bfs(int s) {
    int m, i;
    u[s] = 1;
    q[++rear] = s;
    printf("Reachable officers using BFS method from a given officer %d are:\n", s);
    while (front <= rear) {
        m = q[front++];
        for (i = 1; i <= n; i++) {
            if (a[m][i] == 1 && u[i] == 0) {
                q[++rear] = i;
                printf("Officer %d\n", i);
                u[i] = 1;
            }
        }
    }
}

int main() {
    int s, i, j, ch;
    while (1) {
        printf("1. Create a graph using adjacency matrix indicating people who can communicate directly with each other\n");
        printf("2. DFS traversal method through which any officer can be reachable from a given node\n");
        printf("3. BFS traversal method through which any officer can be reachable from a given node\n");
        printf("4. Exit\n");
        printf("Enter the choice: ");
        scanf("%d", &ch);

        switch (ch) {
            case 1:
                printf("Enter the number of officers: ");
                scanf("%d", &n);
                printf("Enter the adjacency matrix representation:\n");
                for (i = 1; i <= n; i++) {
                    for (j = 1; j <= n; j++) {
                        scanf("%d", &a[i][j]);
                    }
                }
                break;
            case 2:
                printf("Depth First Search Traversal\n");
                printf("Enter Source Officer: ");
                scanf("%d", &s);
                printf("Reachable officers using DFS method from a given officer %d:\n", s);
                dfs(s);
                for (i = 1; i <= n; i++) {
                    if (v[i] == 0) {
                        printf("Officer %d is not visited, and it is a disconnected graph\n", i);
                    }
                }
                break;
            case 3:
                printf("Breadth First Search Traversal\n");
                printf("Enter Source Officer: ");
                scanf("%d", &s);
                bfs(s);
                for (i = 1; i <= n; i++) {
                    if (u[i] == 0) {
                        printf("Officer %d is not visited, and that officer is disconnected\n", i);
                    }
                }
                break;
            case 4:
                exit(0);
            default:
                printf("Invalid choice\n");
        }
    }
    return 0;
}
