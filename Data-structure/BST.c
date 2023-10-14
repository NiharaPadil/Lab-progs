#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct node {
    char name[25], phno[15];
    struct node* lc;
    struct node* rc;
};

typedef struct node* NODE;

int delflag;

NODE createnode() {
    NODE temp;
    temp = (NODE)malloc(sizeof(struct node));
    if (temp == NULL) {
        printf("Memory allocation failed.\n");
        exit(1);
    }
    printf("Enter the name: ");
    scanf("%s", temp->name);
    printf("Enter the phone number: ");
    scanf("%s", temp->phno);
    temp->lc = NULL;
    temp->rc = NULL;
    return temp;
}

void insertBST(NODE root, NODE newNode) {
    if (strcmp(newNode->name, root->name) == 0) {
        printf("Key already exists\n");
        free(newNode); // Free the memory allocated for newNode
        return;
    } else if (strcmp(newNode->name, root->name) < 0) {
        if (root->lc == NULL)
            root->lc = newNode;
        else
            insertBST(root->lc, newNode);
    } else {
        if (root->rc == NULL)
            root->rc = newNode;
        else
            insertBST(root->rc, newNode);
    }
}

int search(NODE root, char keyname[]) {
    if (!root)
        return -1;
    if (strcmp(keyname, root->name) == 0)
        return 1;
    else if (strcmp(keyname, root->name) < 0)
        return search(root->lc, keyname);
    else
        return search(root->rc, keyname);
}

NODE getRightMin(NODE root) {
    NODE temp = root;
    while (temp->lc != NULL) {
        temp = temp->lc;
    }
    return temp;
}

NODE deleteBST(NODE root, char keyname[]) {
    if (!root) {
        delflag = -1;
        return NULL;
    }
    if (strcmp(keyname, root->name) < 0)
        root->lc = deleteBST(root->lc, keyname);
    else if (strcmp(keyname, root->name) > 0)
        root->rc = deleteBST(root->rc, keyname);
    else {
        if (root->lc == NULL && root->rc == NULL) {
            free(root);
            return NULL;
        } else if (root->lc == NULL) {
            NODE temp = root->rc;
            free(root);
            return temp;
        } else if (root->rc == NULL) {
            NODE temp = root->lc;
            free(root);
            return temp;
        } else {
            NODE rightMin = getRightMin(root->rc);
            strcpy(root->name, rightMin->name);
            strcpy(root->phno, rightMin->phno);
            root->rc = deleteBST(root->rc, rightMin->name);
        }
    }
    return root;
}

void inorder(NODE temp) {
    if (temp != NULL) {
        inorder(temp->lc);
        printf("|%s|%s|\t", temp->name, temp->phno);
        inorder(temp->rc);
    }
}

void preorder(NODE temp) {
    if (temp != NULL) {
        printf("|%s|%s|\t", temp->name, temp->phno);
        preorder(temp->lc);
        preorder(temp->rc);
    }
}

void postorder(NODE temp) {
    if (temp != NULL) {
        postorder(temp->lc);
        postorder(temp->rc);
        printf("|%s|%s|\t", temp->name, temp->phno);
    }
}

int main() {
    int choice, n, keyFound = 0;
    char keyname[25];
    NODE root = NULL, newNode;
    printf("--------------------Creating a BST------------------\n");
    printf("Enter the number of records in the BST: ");
    scanf("%d", &n);
    for (int i = 0; i < n; i++) {
        newNode = createnode();
        if (root == NULL)
            root = newNode;
        else
            insertBST(root, newNode);
    }
    while (1) {
        choice = 0;
        printf("\n-----------------MENU----------------------\n");
        printf("1. Search a list for a specified name\n");
        printf("2. Insert a new name\n");
        printf("3. Deleting an existing name\n");
        printf("4. Traverse the phone list\n");
        printf("5. Exit\n");
        printf("-------------------------------------------\n");
        printf("Enter choice: ");
        scanf("%d", &choice);
        switch (choice) {
            case 1:
                printf("Enter the name to be searched: ");
                scanf("%s", keyname);
                keyFound = search(root, keyname);
                if (keyFound == 1)
                    printf("Name: %s is found in the BST\n", keyname);
                else
                    printf("Name: %s is not found in the BST\n", keyname);
                break;
            case 2:
                newNode = createnode();
                if (root == NULL)
                    root = newNode;
                else
                    insertBST(root, newNode);
                break;
            case 3:
                if (root == NULL) {
                    printf("Tree is empty\n");
                } else {
                    delflag = 0;
                    printf("Enter the name to be deleted: ");
                    scanf("%s", keyname);
                    root = deleteBST(root, keyname);
                    if (delflag == -1)
                        printf("Name: %s is not found in the BST\n", keyname);
                    else
                        printf("Name: %s is deleted from the BST\n", keyname);
                }
                break;
            case 4:
                if (root == NULL) {
                    printf("Tree is empty\n");
                } else {
                    printf("BST Preorder traversal\n");
                    preorder(root);
                    printf("\nBST Inorder traversal\n");
                    inorder(root);
                    printf("\nBST Postorder traversal\n");
                    postorder(root);
                }
                break;
            case 5:
                return 0;
            default:
                printf("Wrong choice\n");
                return 1;
        }
    }
    return 0;
}
