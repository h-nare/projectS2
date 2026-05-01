#include <stdio.h>
#include <string.h>

#define MAX_WORDS 200
#define MAX_LEN 50

typedef struct {
    char word[MAX_LEN];
    int tf;
} Word;

int find_word(Word words[], int size, char *token) {
    for (int i = 0; i < size; i++) {
        if (strcmp(words[i].word, token) == 0) {
            return i;
        }
    }

    return -1;
}

int compute_tf(char *text, Word words[]) {
    int word_count = 0;

    char *token = strtok(text, " ");

    while (token != NULL) {
        if (word_count >= MAX_WORDS) {
            break;
        }

        int idx = find_word(words, word_count, token);

        if (idx == -1) {
            strncpy(words[word_count].word, token, MAX_LEN - 1);
            words[word_count].word[MAX_LEN - 1] = '\0';
            words[word_count].tf = 1;
            word_count++;
        } else {
            words[idx].tf++;
        }

        token = strtok(NULL, " ");
    }

    return word_count;
}