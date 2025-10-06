// FILE: bpe_tokenizer.c
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_VOCAB_SIZE 50000
#define INITIAL_BUFFER_SIZE 1048576

typedef struct {
  int p1, p2, count;
} PairCount;
typedef struct {
  int p1, p2, new_id;
} Merge;

#define HASH_SIZE 10007 // Prime number for better distribution

typedef struct HashEntry {
  int p1, p2, count;
  struct HashEntry *next;
} HashEntry;

HashEntry *hash_table[HASH_SIZE];

unsigned int hash(int p1, int p2) {
  return (p1 * 2654435761U + p2) % HASH_SIZE; // Simple hash function
}

void get_stats_fast(int *ids, int len, PairCount **stats, int *stats_count) {
  // Reset hash table
  for (int i = 0; i < HASH_SIZE; i++) {
    HashEntry *entry = hash_table[i];
    while (entry) {
      HashEntry *next = entry->next;
      free(entry);
      entry = next;
    }
    hash_table[i] = NULL;
  }

  // Count pairs
  for (int i = 0; i < len - 1; i++) {
    unsigned int h = hash(ids[i], ids[i + 1]);
    HashEntry *entry = hash_table[h];

    while (entry) {
      if (entry->p1 == ids[i] && entry->p2 == ids[i + 1]) {
        entry->count++;
        break;
      }
      entry = entry->next;
    }

    if (!entry) {
      HashEntry *new_entry = malloc(sizeof(HashEntry));
      new_entry->p1 = ids[i];
      new_entry->p2 = ids[i + 1];
      new_entry->count = 1;
      new_entry->next = hash_table[h];
      hash_table[h] = new_entry;
    }
  }

  // Count total entries and allocate
  *stats_count = 0;
  for (int i = 0; i < HASH_SIZE; i++) {
    HashEntry *entry = hash_table[i];
    while (entry) {
      (*stats_count)++;
      entry = entry->next;
    }
  }

  *stats = malloc(*stats_count * sizeof(PairCount));
  int j = 0;
  for (int i = 0; i < HASH_SIZE; i++) {
    HashEntry *entry = hash_table[i];
    while (entry) {
      (*stats)[j++] = (PairCount){entry->p1, entry->p2, entry->count};
      entry = entry->next;
    }
  }
}

void get_stats(int *ids, int len, PairCount **stats, int *stats_count) {
  *stats_count = 0;
  *stats = NULL;
  for (int i = 0; i < len - 1; ++i) {
    bool found = false;
    for (int j = 0; j < *stats_count; ++j) {
      if ((*stats)[j].p1 == ids[i] && (*stats)[j].p2 == ids[i + 1]) {
        (*stats)[j].count++;
        found = true;
        break;
      }
    }
    if (!found) {
      *stats = realloc(*stats, (*stats_count + 1) * sizeof(PairCount));
      if (!*stats) {
        fprintf(stderr, "Fatal: realloc failed\n");
        exit(1);
      }
      (*stats)[*stats_count] = (PairCount){ids[i], ids[i + 1], 1};
      (*stats_count)++;
    }
  }
}

void merge(int *ids, int len, int p1, int p2, int new_id, int **new_ids,
           int *new_len) {
  *new_len = 0;
  *new_ids = malloc(len * sizeof(int));
  if (!*new_ids) {
    fprintf(stderr, "Fatal: malloc failed\n");
    exit(1);
  }
  int i = 0;
  while (i < len) {
    if (i < len - 1 && ids[i] == p1 && ids[i + 1] == p2) {
      (*new_ids)[(*new_len)++] = new_id;
      i += 2;
    } else {
      (*new_ids)[(*new_len)++] = ids[i];
      i++;
    }
  }
}

void vocab_init(char *vocab[MAX_VOCAB_SIZE]) {
  for (int i = 0; i < 256; i++) {
    vocab[i] = malloc(2);
    vocab[i][0] = (char)i;
    vocab[i][1] = '\0';
  }
}

void vocab_add(char *vocab[MAX_VOCAB_SIZE], int id, const char *token1,
               const char *token2) {
  vocab[id] = malloc(strlen(token1) + strlen(token2) + 1);
  strcpy(vocab[id], token1);
  strcat(vocab[id], token2);
}

const char *vocab_resolve(char *vocab[MAX_VOCAB_SIZE], int id) {
  return vocab[id];
}

void vocab_free(char *vocab[MAX_VOCAB_SIZE], int vocab_size) {
  for (int i = 0; i < vocab_size; i++) {
    free(vocab[i]);
  }
}

void train_tokenizer(int vocab_size) {
  setbuf(stdout, NULL);

  char *corpus = malloc(INITIAL_BUFFER_SIZE);
  if (!corpus) {
    fprintf(stderr, "Fatal: malloc failed\n");
    exit(1);
  }
  long corpus_size = 0, buffer_size = INITIAL_BUFFER_SIZE;
  int c;

  while ((c = getchar()) != EOF) {
    if (corpus_size >= buffer_size - 1) {
      buffer_size *= 2;
      corpus = realloc(corpus, buffer_size);
      if (!corpus) {
        fprintf(stderr, "Fatal: realloc failed\n");
        exit(1);
      }
    }
    corpus[corpus_size++] = (char)c;
  }

  corpus[corpus_size] = '\0';

  int *ids = malloc(corpus_size * sizeof(int));
  if (!ids) {
    fprintf(stderr, "Fatal: malloc failed\n");
    exit(1);
  }
  for (int i = 0; i < corpus_size; ++i)
    ids[i] = (int)(unsigned char)corpus[i];
  int ids_len = corpus_size;
  free(corpus);

  Merge merges[MAX_VOCAB_SIZE];
  char *vocab[MAX_VOCAB_SIZE];
  vocab_init(vocab);

  int num_merges = vocab_size - 256;
  if (num_merges > MAX_VOCAB_SIZE - 256)
    num_merges = MAX_VOCAB_SIZE - 256;
  int actual_merges = 0;
  for (int i = 0; i < num_merges; ++i) {
    PairCount *stats = NULL;
    int stats_count = 0;
    get_stats_fast(ids, ids_len, &stats, &stats_count);
    if (stats_count == 0) {
      break;
    }
    int max_count = -1;
    PairCount best_pair = {0, 0, 0};
    for (int j = 0; j < stats_count; ++j) {
      if (stats[j].count > max_count) {
        max_count = stats[j].count;
        best_pair = stats[j];
      }
    }
    free(stats);
    int new_id = 256 + i;
    int *new_ids;
    int new_len;
    const char *r1 = vocab_resolve(vocab, best_pair.p1);
    const char *r2 = vocab_resolve(vocab, best_pair.p2);
    vocab_add(vocab, new_id, r1, r2);
    merge(ids, ids_len, best_pair.p1, best_pair.p2, new_id, &new_ids, &new_len);
    free(ids);
    ids = new_ids;
    ids_len = new_len;
    merges[i] = (Merge){best_pair.p1, best_pair.p2, new_id};

    const char *r3 = vocab_resolve(vocab, new_id);
    fprintf(stderr, "Merge %d/%d: Merging ('%s', '%s') -> '%s' (ID: %d)\n",
            i + 1, num_merges, r1, r2, r3, new_id);

    actual_merges++;
  }
  free(ids);
  vocab_free(vocab, 256 + actual_merges);
  printf("{");
  for (int i = 0; i < actual_merges; ++i) {
    printf("\"%d,%d\":%d", merges[i].p1, merges[i].p2, merges[i].new_id);
    if (i < actual_merges - 1)
      printf(",");
  }
  printf("}");
}

// ============================================================================
//               THE FINAL, EFFICIENT, SINGLE-PASS ENCODER
// ============================================================================
void encode_text(const char *vocab_path) {
  // 1. Load merge rules into a simple lookup table.
  // For simplicity, we use a large flat array. A hash map would be more robust.
  Merge merges[MAX_VOCAB_SIZE];
  int merges_count = 0;
  FILE *file = fopen(vocab_path, "r");
  if (!file) {
    fprintf(stderr, "Error: Could not open vocab file %s\n", vocab_path);
    exit(1);
  }
  fseek(file, 0, SEEK_END);
  long file_size = ftell(file);
  fseek(file, 0, SEEK_SET);
  char *content = malloc(file_size + 1);
  if (!content) {
    fprintf(stderr, "Fatal: malloc failed\n");
    fclose(file);
    exit(1);
  }
  fread(content, 1, file_size, file);
  fclose(file);
  content[file_size] = '\0';
  char *p = content;
  while (*p && merges_count < MAX_VOCAB_SIZE) {
    while (*p && *p != '"')
      p++;
    if (!*p)
      break;
    p++;
    int p1, p2, new_id;
    if (sscanf(p, "%d,%d", &p1, &p2) == 2) {
      while (*p && *p != ':')
        p++;
      if (!*p)
        break;
      p++;
      if (sscanf(p, "%d", &new_id) == 1) {
        merges[merges_count++] = (Merge){p1, p2, new_id};
      }
    }
    while (*p && *p != ',')
      p++;
    if (!*p)
      break;
  }
  free(content);

  // 2. Load text from stdin
  char *text = malloc(INITIAL_BUFFER_SIZE);
  if (!text) {
    fprintf(stderr, "Fatal: malloc failed\n");
    exit(1);
  }
  long text_size = 0, buffer_size = INITIAL_BUFFER_SIZE;
  int c;
  while ((c = getchar()) != EOF) {
    if (text_size >= buffer_size - 1) {
      buffer_size *= 2;
      text = realloc(text, buffer_size);
      if (!text) {
        fprintf(stderr, "Fatal: realloc failed\n");
        exit(1);
      }
    }
    text[text_size++] = (char)c;
  }
  text[text_size] = '\0';

  // 3. --- SINGLE PASS ENCODING ---
  // We will build the final token list directly.
  int *tokens = malloc(text_size * sizeof(int)); // Max possible size
  if (!tokens) {
    fprintf(stderr, "Fatal: malloc failed\n");
    exit(1);
  }
  int tokens_count = 0;

  for (long i = 0; i < text_size;) {
    // Find the longest possible merge starting at position i
    int best_match_id = (int)(unsigned char)text[i];
    int best_match_len = 1;

    // This is a simplified trie search. We check for 2-char merges.
    // A full implementation would search for longer merges recursively.
    if (i < text_size - 1) {
      int current_pair_p1 = (int)(unsigned char)text[i];
      int current_pair_p2 = (int)(unsigned char)text[i + 1];
      for (int j = 0; j < merges_count; ++j) {
        if (merges[j].p1 == current_pair_p1 &&
            merges[j].p2 == current_pair_p2) {
          best_match_id = merges[j].new_id;
          best_match_len = 2; // In this simple model, merges are always 2 bytes
          break; // Since merges are ordered, the first one we find is the
                 // highest priority
        }
      }
    }

    tokens[tokens_count++] = best_match_id;
    i += best_match_len;
  }
  free(text);

  // 4. Print the final list of tokens
  for (int i = 0; i < tokens_count; ++i) {
    printf("%d ", tokens[i]);
  }
  printf("\n");
  free(tokens);
}

int main(int argc, char *argv[]) {
  // Force unbuffered output immediately
  setvbuf(stdin, NULL, _IONBF, 0);
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);

  if (argc < 2) {
    fprintf(stderr, "Usage: %s <command> [args]\n", argv[0]);
    fprintf(stderr, "Commands:\n  train <vocab_size>\n  encode <vocab_path>\n");
    return 1;
  }
  if (strcmp(argv[1], "train") == 0) {
    if (argc == 3) {
      train_tokenizer(atoi(argv[2]));
    } else {
      fprintf(stderr, "Usage: %s train <vocab_size>\n", argv[0]);
      return 1;
    }
  } else if (strcmp(argv[1], "encode") == 0) {
    if (argc != 3) {
      fprintf(stderr, "Usage: %s encode <vocab_path>\n", argv[0]);
      return 1;
    }
    encode_text(argv[2]);
  } else {
    fprintf(stderr, "Unknown command: %s\n", argv[1]);
    return 1;
  }
  return 0;
}
