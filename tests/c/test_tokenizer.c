/**
 * @file test_tokenizer.c
 * @brief Unit tests for tokenizer stub.
 */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>   /* malloc, free */
#include "ie_io.h"

int main(void) {
  ie_vocab_t v;
  assert(ie_vocab_load("models/gpt-oss-20b/vocab.json", &v) == 0);
  assert(v.vocab_size > 0);

  const char *txt = "hello world  from   engine";
  uint32_t needed = 0;
  assert(ie_tok_encode(&v, txt, NULL, &needed) == 0);
  assert(needed == 4);

  uint32_t *ids = (uint32_t*)malloc(sizeof(uint32_t) * needed);
  assert(ids != NULL);

  uint32_t got = 0;
  assert(ie_tok_encode(&v, txt, ids, &got) == 0);
  assert(got == needed);

  char buf[256];
  assert(ie_tok_decode(&v, ids, got, buf, sizeof(buf)) == 0);
  assert(strncmp(buf, "T", 1) == 0);

  free(ids);
  ie_vocab_free(&v);
  printf("ok test_tokenizer\n");
  return 0;
}
